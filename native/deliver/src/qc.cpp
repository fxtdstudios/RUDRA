#include "rudra/deliver/qc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/core/numeric.hpp"

namespace rudra {
namespace {

constexpr double kLuma[3] = {0.2627, 0.6780, 0.0593};
constexpr double kQcDiffuseWhite = 203.0;

double srgb_to_linear(double v) { return v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); }

// Python's format spec for a float, non-finite values as Python prints them.
std::string pyfmt(double v, const char* spec) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char b[64];
    std::snprintf(b, sizeof b, spec, v);
    return b;
}

// Pad to a width in characters (code points), as Python's format does.
std::size_t chars(const std::string& s) {
    return static_cast<std::size_t>(std::count_if(s.begin(), s.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}
std::string ljust(const std::string& s, std::size_t w) { return s + std::string(w > chars(s) ? w - chars(s) : 0, ' '); }
std::string rjust(const std::string& s, std::size_t w) { return std::string(w > chars(s) ? w - chars(s) : 0, ' ') + s; }

}  // namespace

const QcCheck* QcReport::first_failure() const {
    for (const auto& c : checks)
        if (!c.ok()) return &c;
    return nullptr;
}

Result<QcThresholds> load_qc_thresholds(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return make_error(ErrorCode::NotFound, "The QC thresholds file could not be opened.", path.string());
    QcThresholds t;
    try {
        const auto j = nlohmann::json::parse(in);
        for (const auto& [k, v] : j.items())
            if (!k.empty() && k[0] != '_') t[k] = v.at("value").get<double>();
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The QC thresholds file is not valid.", e.what());
    }
    return t;
}

Result<QcReport> check_frame(const NitsFrame& hdr, const SdrImage& sdr_img, const QcThresholds& t, double ceiling) {
    const int h = hdr.height(), w = hdr.width();
    if (sdr_img.height() != h || sdr_img.width() != w)
        return make_error(ErrorCode::InvalidArgument, "shape mismatch between the reconstruction and its source");
    for (const char* k : {"nan_max", "inf_max", "negative_max", "at_ceiling_max_pct", "flat_region_min_px",
                          "do_no_harm_median_max", "do_no_harm_within_1pct_min", "chroma_shift_median_max_uv",
                          "clipped_min_px", "clip_gain_min", "clip_structure_min_pct", "hf_excess_max"})
        if (!t.count(k)) return make_error(ErrorCode::InvalidArgument, "A QC threshold is missing.", k);
    auto T = [&](const char* k) { return t.at(k); };
    const std::size_t n = hdr.plane_size();
    const PlanarBuffer& s = sdr_img.buffer();

    std::vector<double> y_out(n), y_in(n), code_max(n);
    std::vector<std::array<double, 3>> src(n);
    for (std::size_t i = 0; i < n; ++i) {
        double yo = 0.0, yi = 0.0, mx = -INFINITY;
        for (int c = 0; c < 3; ++c) {
            const double sv = s.plane(c)[i];
            src[i][std::size_t(c)] = srgb_to_linear(sv) * kQcDiffuseWhite;
            yo += hdr.plane(c)[i] * kLuma[c];
            yi += src[i][std::size_t(c)] * kLuma[c];
            mx = std::max(mx, sv);
        }
        y_out[i] = yo;
        y_in[i] = yi;
        code_max[i] = mx;
    }

    QcReport r;
    // -- data integrity
    std::size_t nan = 0, inf = 0, neg = 0, pinned_n = 0;
    for (double v : hdr.span()) {
        nan += std::isnan(v);
        inf += std::isinf(v);
        neg += v < 0;
        pinned_n += v >= ceiling * 0.999;
    }
    const struct { const char* key; std::size_t count; const char* label; const char* name; } integrity[] = {
        {"nan_max", nan, "NaN", "nan"}, {"inf_max", inf, "Inf", "inf"}, {"negative_max", neg, "negative", "negative"}};
    for (const auto& it : integrity)
        r.add({it.name, double(it.count) <= T(it.key) ? kQcPass : kQcFail, double(it.count), T(it.key),
               std::to_string(it.count) + " " + it.label + " sample(s)"});
    const double pinned = 100.0 * double(pinned_n) / double(hdr.span().size());
    r.add({"at_ceiling_pct", pinned <= T("at_ceiling_max_pct") ? kQcPass : kQcFail, pinned, T("at_ceiling_max_pct"),
           pyfmt(pinned, "%.4f") + "% pinned at " + pyfmt(ceiling, "%.0f") + " nits"});

    // -- do no harm
    std::vector<std::size_t> safe;
    for (std::size_t i = 0; i < n; ++i)
        if (code_max[i] < 250.0 / 255.0) safe.push_back(i);
    if (double(safe.size()) < T("flat_region_min_px")) {
        r.add({"do_no_harm_median", kQcUnmeasured, std::nullopt, T("do_no_harm_median_max"),
               "only " + std::to_string(safe.size()) + " unclipped px -- nothing to compare"});
        r.add({"do_no_harm_within_1pct", kQcUnmeasured, std::nullopt, T("do_no_harm_within_1pct_min"), "no unclipped sample"});
        r.add({"chroma_shift_uv", kQcUnmeasured, std::nullopt, T("chroma_shift_median_max_uv"), "no unclipped sample"});
    } else {
        std::vector<double> ratio, duv;
        ratio.reserve(safe.size());
        std::size_t within_n = 0;
        for (std::size_t i : safe) {
            const double q = y_out[i] / std::max(y_in[i], 1e-9);
            ratio.push_back(q);
            within_n += std::abs(q - 1.0) < 0.01;
            double hs = 0.0, ss = 0.0;
            for (int c = 0; c < 3; ++c) { hs += hdr.plane(c)[i]; ss += src[i][std::size_t(c)]; }
            hs = std::max(hs, 1e-12);
            ss = std::max(ss, 1e-12);
            const double du = hdr.plane(0)[i] / hs - src[i][0] / ss, dv = hdr.plane(1)[i] / hs - src[i][1] / ss;
            duv.push_back(std::sqrt(du * du + dv * dv));
        }
        const double med = np_median(ratio);
        const double within = double(within_n) / double(safe.size());
        r.add({"do_no_harm_median", med <= T("do_no_harm_median_max") ? kQcPass : kQcFail, med, T("do_no_harm_median_max"),
               pyfmt(med, "%.3f") + "x on unclipped picture"});
        r.add({"do_no_harm_within_1pct", within >= T("do_no_harm_within_1pct_min") ? kQcPass : kQcFail, within,
               T("do_no_harm_within_1pct_min"), pyfmt(100 * within, "%.1f") + "% within 1%"});
        const double d = np_median(duv);
        r.add({"chroma_shift_uv", d <= T("chroma_shift_median_max_uv") ? kQcPass : kQcFail, d,
               T("chroma_shift_median_max_uv"), "median duv " + pyfmt(d, "%.4f")});
    }

    // -- clip recovery
    std::vector<unsigned char> clipped(n);
    std::size_t clipped_n = 0;
    for (std::size_t i = 0; i < n; ++i) clipped_n += (clipped[i] = code_max[i] >= 254.0 / 255.0);
    auto blur = [&](const std::vector<double>& a, double sigma) {
        std::vector<float> f(a.begin(), a.end());
        const auto b = gaussian_blur_replicate(f, h, w, sigma);
        return std::vector<double>(b.begin(), b.end());
    };
    if (double(clipped_n) < T("clipped_min_px")) {
        const std::string why = "only " + std::to_string(clipped_n) + " clipped px -- recovery cannot be judged";
        r.add({"clip_gain", kQcUnmeasured, std::nullopt, T("clip_gain_min"), why});
        r.add({"clip_structure_pct", kQcUnmeasured, std::nullopt, T("clip_structure_min_pct"), why});
    } else {
        std::vector<double> yo_c, yi_c;
        for (std::size_t i = 0; i < n; ++i)
            if (clipped[i]) { yo_c.push_back(y_out[i]); yi_c.push_back(y_in[i]); }
        const double gain = np_mean(yo_c) / std::max(np_mean(yi_c), 1e-9);
        r.add({"clip_gain", gain >= T("clip_gain_min") ? kQcPass : kQcFail, gain, T("clip_gain_min"),
               pyfmt(gain, "%.2f") + "x on " + pyfmt(100.0 * double(clipped_n) / double(n), "%.2f") + "% of px"});
        // cv2.erode, 5x5, border never erodes (morphologyDefaultBorderValue).
        std::vector<unsigned char> interior(n, 0);
        std::size_t interior_n = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                unsigned char m = 1;
                for (int dy = -2; dy <= 2 && m; ++dy)
                    for (int dx = -2; dx <= 2 && m; ++dx) {
                        const int yy = y + dy, xx = x + dx;
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
                        m = clipped[std::size_t(yy) * std::size_t(w) + std::size_t(xx)];
                    }
                interior[std::size_t(y) * std::size_t(w) + std::size_t(x)] = m;
                interior_n += m;
            }
        const auto& mask = double(interior_n) >= T("clipped_min_px") ? interior : clipped;
        const auto low = blur(y_out, 3.0);
        std::vector<double> sel;
        for (std::size_t i = 0; i < n; ++i)
            if (mask[i]) sel.push_back(low[i]);
        const double st = np_std(sel) / std::max(np_mean(sel), 1e-9) * 100;
        r.add({"clip_structure_pct", st >= T("clip_structure_min_pct") ? kQcPass : kQcFail, st,
               T("clip_structure_min_pct"), pyfmt(st, "%.1f") + "% relative variation"});
    }

    // -- manufactured noise
    const auto bi = blur(y_in, 2.0);
    std::vector<double> grad(n);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            auto at = [&](int yy, int xx) { return bi[std::size_t(yy) * std::size_t(w) + std::size_t(xx)]; };
            const double gy = h < 2 ? 0.0 : y == 0 ? at(1, x) - at(0, x) : y == h - 1 ? at(h - 1, x) - at(h - 2, x)
                                                                                      : (at(y + 1, x) - at(y - 1, x)) / 2.0;
            const double gx = w < 2 ? 0.0 : x == 0 ? at(y, 1) - at(y, 0) : x == w - 1 ? at(y, w - 1) - at(y, w - 2)
                                                                                      : (at(y, x + 1) - at(y, x - 1)) / 2.0;
            grad[std::size_t(y) * std::size_t(w) + std::size_t(x)] = std::hypot(gy, gx);
        }
    std::vector<unsigned char> flat(n, 0);
    std::vector<double> gu;
    for (std::size_t i = 0; i < n; ++i)
        if (!clipped[i] && y_in[i] > 40.0) gu.push_back(grad[i]);
    std::size_t flat_n = 0;
    if (!gu.empty()) {
        const double p25 = np_percentile(gu, 25.0);
        for (std::size_t i = 0; i < n; ++i)
            if (!clipped[i] && y_in[i] > 40.0 && grad[i] <= p25) { flat[i] = 1; ++flat_n; }
    }
    if (double(flat_n) < T("flat_region_min_px")) {
        r.add({"hf_excess", kQcUnmeasured, std::nullopt, T("hf_excess_max"),
               "only " + std::to_string(flat_n) + " flat px -- noise cannot be measured"});
        return r;
    }
    std::vector<double> xi, yo;
    for (std::size_t i = 0; i < n; ++i)
        if (flat[i]) { xi.push_back(y_in[i]); yo.push_back(y_out[i]); }
    const std::size_t bins = std::clamp<std::size_t>(xi.size() / 400, 8, 60);
    const double xmin = *std::min_element(xi.begin(), xi.end()), xmax = *std::max_element(xi.begin(), xi.end());
    const auto edges = np_linspace(xmin, xmax, bins + 1);
    std::vector<double> mid_good, med_good;
    for (std::size_t b = 0; b < bins; ++b) {
        std::vector<double> sel;
        for (std::size_t i = 0; i < xi.size(); ++i)
            if (xi[i] >= edges[b] && xi[i] < edges[b + 1]) sel.push_back(yo[i]);
        if (sel.size() > 20) {
            const double m = np_median(sel);
            if (!std::isnan(m)) {
                mid_good.push_back(0.5 * (edges[b + 1] + edges[b]));
                med_good.push_back(m);
            }
        }
    }
    if (mid_good.size() < 4) {
        r.add({"hf_excess", kQcUnmeasured, std::nullopt, T("hf_excess_max"), "transfer curve could not be estimated"});
        return r;
    }
    const auto dcurve = np_gradient(med_good, mid_good);
    const auto bo = blur(y_out, 1.6), bin_ = blur(y_in, 1.6);
    std::vector<double> lvl, pred, act;
    for (std::size_t i = 0, k = 0; i < n; ++i) {
        if (!flat[i]) continue;
        const double slope = np_interp(xi[k++], mid_good, dcurve);
        lvl.push_back(bo[i]);
        pred.push_back((y_in[i] - bin_[i]) * slope);
        act.push_back(y_out[i] - bo[i]);
    }
    const double level = np_mean(lvl);
    const double predicted = np_std(pred) / level * 100;
    const double actual = np_std(act) / level * 100;
    const double excess = actual / std::max(predicted, 1e-9);
    r.add({"hf_excess", excess <= T("hf_excess_max") ? kQcPass : kQcFail, excess, T("hf_excess_max"),
           pyfmt(actual, "%.1f") + "% actual vs " + pyfmt(predicted, "%.1f") + "% predicted"});
    return r;
}

std::string format_qc_report(const QcReport& r, const std::string& name, const std::string& resolution) {
    // The report's own dashes are the Python's text, kept byte for byte.
    const std::string dash = "\xE2\x80\x94";
    std::string out = "QC " + dash + " " + name + "\n  " + resolution + "\n  VERDICT: " + r.verdict;
    for (const auto& c : r.checks) {
        const std::string v = c.value ? pyfmt(*c.value, "%.4g") : dash;
        out += "\n    " + ljust(c.status, 10) + " " + ljust(c.name, 24) + " " + rjust(v, 10) + "   " + c.detail;
    }
    if (const QcCheck* f = r.first_failure())
        out += "\n  First failure: " + f->name + " (" + f->status + ") " + dash + " " + f->detail;
    return out;
}

}  // namespace rudra
