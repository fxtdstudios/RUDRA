#include "rudra/core/scopes.hpp"

#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rudra {
namespace {

constexpr double kPeak = 10000.0, kDiffuseWhiteNits = 203.0;

// Math.round: half toward +infinity.
double js_round(double x) { return std::floor(x + 0.5); }

// Uint8ClampedArray assignment: clamp to [0, 255], round half to even.
std::uint8_t clamped_u8(double v) {
    if (!(v > 0.0)) return 0;   // NaN and negatives
    if (v >= 255.0) return 255;
    return std::uint8_t(std::nearbyint(v));   // the default rounding mode is to nearest, ties to even
}

}  // namespace

SampleGrid sample_grid(int w, int h) {
    const double scale = std::min(1.0, double(kSampleMaxSide) / double(std::max(w, h)));
    SampleGrid g;
    g.width = std::max(1, int(js_round(double(w) * scale)));
    g.height = std::max(1, int(js_round(double(h) * scale)));
    g.index.resize(std::size_t(g.width) * std::size_t(g.height));
    for (int y = 0; y < g.height; ++y) {
        const long long sy = std::min<long long>(h - 1, (2LL * y + 1) * h / (2LL * g.height));
        for (int x = 0; x < g.width; ++x) {
            const long long sx = std::min<long long>(w - 1, (2LL * x + 1) * w / (2LL * g.width));
            g.index[std::size_t(y) * std::size_t(g.width) + std::size_t(x)] = int(sy * w + sx);
        }
    }
    return g;
}

PlanarBuffer take_sample(const PlanarBuffer& rgb, const SampleGrid& g) {
    PlanarBuffer out(rgb.channels(), g.height, g.width);
    for (int c = 0; c < rgb.channels(); ++c)
        for (std::size_t i = 0; i < g.index.size(); ++i) out.plane(c)[i] = rgb.plane(c)[std::size_t(g.index[i])];
    return out;
}

MaskCoverage mask_coverage(const std::vector<float>& hi, const std::vector<float>& sh,
                           const std::vector<std::uint8_t>& sdr8) {
    const std::size_t n = hi.size();
    std::size_t hc = 0, sc = 0, cc = 0;
    for (std::size_t i = 0; i < n; ++i) {
        hc += hi[i] > 0.5f;
        sc += sh[i] > 0.5f;
        cc += sdr8[i * 3] >= 254 || sdr8[i * 3 + 1] >= 254 || sdr8[i * 3 + 2] >= 254;
    }
    return {100.0 * double(hc) / double(n), 100.0 * double(sc) / double(n), 100.0 * double(cc) / double(n)};
}

Measured measure_view(const PlanarBuffer& ms, const PlanarBuffer& bs, const SampleGrid& g,
                      const std::vector<float>& hi_mask, const std::vector<float>& sh_mask,
                      const MaskCoverage& cov, const Reductions& model, const Reductions& base,
                      std::size_t full_pixels) {
    const std::size_t n = std::size_t(g.width) * std::size_t(g.height);
    Measured out;
    std::vector<float>& luma = out.luma;
    luma.resize(n);
    std::vector<float> base_luma(n);
    const std::size_t channels = n * 3;
    constexpr int kChBins = 2048;
    const double lo_log = std::log2(kScopeLoNits), hi_log = std::log2(kScopeHiNits), span_log = hi_log - lo_log;

    // Per pixel, in parallel chunks: the maxima, the channel histogram and the
    // two counts are integers or disjoint writes, so the chunking cannot change
    // them. The departure term is written per pixel here and summed in order below.
    std::vector<double> lr2(n);
    const int workers = detail::worker_count(n);
    std::vector<std::vector<std::uint32_t>> hists(static_cast<std::size_t>(workers), std::vector<std::uint32_t>(kChBins, 0));
    std::vector<std::size_t> dws(std::size_t(workers), 0), k1s(std::size_t(workers), 0);
    detail::parallel_chunks(n, workers, [&](int wk, std::size_t i0, std::size_t i1) {
        auto& hist = hists[std::size_t(wk)];
        std::size_t dw = 0, k1 = 0;
        for (std::size_t i = i0; i < i1; ++i) {
            const double r = double(ms.plane(0)[i]) * kPeak, gg = double(ms.plane(1)[i]) * kPeak,
                         b = double(ms.plane(2)[i]) * kPeak;
            luma[i] = float(std::max(r, std::max(gg, b)));
            base_luma[i] = float(double(std::max(bs.plane(0)[i], std::max(bs.plane(1)[i], bs.plane(2)[i]))) * kPeak);
            for (const double v : {r, gg, b}) {
                if (v > kDiffuseWhiteNits * (1 + 1e-6)) ++dw;
                if (v > 1000.0) ++k1;
                const double t = (std::log2(std::min(std::max(v, kScopeLoNits), kScopeHiNits)) - lo_log) / span_log;
                hist[std::size_t(std::min(double(kChBins - 1), std::max(0.0, std::floor(t * kChBins))))]++;
            }
            const double lr = std::log2((double(luma[i]) + 1e-4) / (double(base_luma[i]) + 1e-4));
            lr2[i] = lr * lr;
        }
        dws[std::size_t(wk)] = dw;
        k1s[std::size_t(wk)] = k1;
    });
    std::size_t above_dw = 0, above_1k = 0;
    std::vector<double> ch_hist(kChBins, 0.0);
    for (int wk = 0; wk < workers; ++wk) {
        above_dw += dws[std::size_t(wk)];
        above_1k += k1s[std::size_t(wk)];
        for (int k = 0; k < kChBins; ++k) ch_hist[std::size_t(k)] += double(hists[std::size_t(wk)][std::size_t(k)]);
    }
    auto pct = [&](double p) {
        const double want = double(channels) * p / 100.0;
        double acc = 0.0;
        for (int k = 0; k < kChBins; ++k) {
            acc += ch_hist[std::size_t(k)];
            if (acc >= want) return std::pow(2.0, lo_log + ((k + 0.5) / kChBins) * span_log);
        }
        return kScopeHiNits;
    };

    double hi_sum = 0, hi_base = 0, sh_sum = 0, sh_base = 0, rms = 0;
    std::size_t hi_count = 0, sh_count = 0;
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t src = std::size_t(g.index[j]);
        if (hi_mask[src] > 0.5f) { hi_sum += luma[j]; hi_base += base_luma[j]; ++hi_count; }
        if (sh_mask[src] > 0.5f) { sh_sum += luma[j]; sh_base += base_luma[j]; ++sh_count; }
        rms += lr2[j];
    }
    auto stops_in = [](double sum, double b, std::size_t count) {
        if (!count) return std::numeric_limits<double>::quiet_NaN();
        return std::log2(std::max(sum / double(count), 1e-6) / std::max(b / double(count), 1e-6));
    };

    ViewerMetrics& m = out.metrics;
    const double peak = double(model.peak) * kPeak, base_peak = double(base.peak) * kPeak;
    const double mean = double(model.sum) * kPeak / double(full_pixels);
    m.maxcll = std::ceil(peak);
    m.maxfall = std::ceil(mean);
    m.peak_nits = peak;
    m.baseline_peak_nits = base_peak;
    m.headroom_stops = std::log2(std::max(peak, 1e-6) / std::max(base_peak, 1e-6));
    m.headroom_highlight_stops = stops_in(hi_sum, hi_base, hi_count);
    m.headroom_shadow_stops = stops_in(sh_sum, sh_base, sh_count);
    m.departure_rms_stops = std::sqrt(rms / double(n));
    m.p99_nits = pct(99);
    m.median_nits = pct(50);
    m.above_diffuse_white_pct = 100.0 * double(above_dw) / double(channels);
    m.above_1000_nits_pct = 100.0 * double(above_1k) / double(channels);
    m.highlight_mask_pct = cov.highlight_pct;
    m.shadow_mask_pct = cov.shadow_pct;
    out.scopes = build_scopes(luma, g.width, g.height);
    return out;
}

ScopeData build_scopes(const std::vector<float>& luma, int w, int h) {
    constexpr int kColumns = 230, kHistBins = 76, kColBins = 512;
    const double lo_log10 = std::log10(kScopeLoNits), span10 = std::log10(kScopeHiNits / kScopeLoNits);
    const double lo_log2 = std::log2(kScopeLoNits), span2 = std::log2(kScopeHiNits) - lo_log2;
    const std::size_t rows = std::size_t(std::max(h, 0));
    const int workers = detail::worker_count(rows * std::size_t(std::max(w, 0)));
    std::vector<std::vector<std::uint32_t>> wcols(static_cast<std::size_t>(workers)), whist(static_cast<std::size_t>(workers));
    detail::parallel_chunks(rows, workers, [&](int wk, std::size_t y0, std::size_t y1) {
        auto& c = wcols[std::size_t(wk)];
        auto& hh = whist[std::size_t(wk)];
        c.assign(std::size_t(kColumns) * kColBins, 0);
        hh.assign(kHistBins, 0);
        for (std::size_t y = y0; y < y1; ++y)
            for (int x = 0; x < w; ++x) {
                const double v = std::min(std::max(double(luma[y * std::size_t(w) + std::size_t(x)]), kScopeLoNits), kScopeHiNits);
                const double t = (std::log10(v) - lo_log10) / span10;
                const int col = int(std::min(double(kColumns - 1), std::floor(double(x) / double(w) * kColumns)));
                c[std::size_t(col) * kColBins + std::size_t(std::min(double(kColBins - 1), std::floor(t * kColBins)))]++;
                hh[std::size_t(std::min(double(kHistBins - 1), std::floor(((std::log2(v) - lo_log2) / span2) * kHistBins)))]++;
            }
    });
    std::vector<double> cols(std::size_t(kColumns) * kColBins, 0.0), counts(kColumns, 0.0), hist(kHistBins, 0.0);
    for (int wk = 0; wk < workers; ++wk) {
        for (std::size_t k = 0; k < cols.size(); ++k) cols[k] += double(wcols[std::size_t(wk)][k]);
        for (std::size_t k = 0; k < hist.size(); ++k) hist[k] += double(whist[std::size_t(wk)][k]);
    }
    for (int c = 0; c < kColumns; ++c)
        for (int k = 0; k < kColBins; ++k) counts[std::size_t(c)] += cols[std::size_t(c) * kColBins + std::size_t(k)];
    ScopeData out;
    const double wanted[5] = {2, 25, 50, 75, 98};
    std::vector<double>* keys[5] = {&out.lo, &out.q1, &out.mid, &out.q3, &out.hi};
    for (int c = 0; c < kColumns; ++c) {
        const double total = counts[std::size_t(c)] != 0.0 ? double(counts[std::size_t(c)]) : 1.0;
        double acc = 0.0, got[5] = {0, 0, 0, 0, 0};
        int next = 0;
        for (int k = 0; k < kColBins && next < 5; ++k) {
            acc += cols[std::size_t(c) * kColBins + std::size_t(k)];
            while (next < 5 && acc >= total * wanted[next] / 100.0) {
                got[next] = (k + 0.5) / kColBins;
                ++next;
            }
        }
        while (next < 5) got[next++] = 1.0;
        for (int q = 0; q < 5; ++q) keys[q]->push_back(got[q]);
    }
    double peak = 1.0;
    for (double v : hist) peak = std::max(peak, v);
    for (double v : hist) out.histogram.push_back(v / peak);
    return out;
}

std::vector<std::uint8_t> vectorscope(const PlanarBuffer& ms) {
    const std::size_t n = ms.plane_size();
    constexpr double kr = 0.2627, kg = 0.6780, kb = 0.0593;
    const double half = kVectorSize / 2.0;
    const std::size_t cells = std::size_t(kVectorSize) * kVectorSize;
    const int workers = detail::worker_count(n);
    std::vector<std::vector<std::uint32_t>> wacc(static_cast<std::size_t>(workers));
    detail::parallel_chunks(n, workers, [&](int wk, std::size_t i0, std::size_t i1) {
        auto& a = wacc[std::size_t(wk)];
        a.assign(cells, 0);
        for (std::size_t i = i0; i < i1; ++i) {
            const double r = double(ms.plane(0)[i]) * kPeak, g = double(ms.plane(1)[i]) * kPeak,
                         b = double(ms.plane(2)[i]) * kPeak;
            const double y = r * kr + g * kg + b * kb;
            if (y < 0.02) continue;   // black has no hue to report
            const double norm = std::max(y, 1.0);
            const double cb = (b - y) / (2 * (1 - kb)) / norm;
            const double cr = (r - y) / (2 * (1 - kr)) / norm;
            const double px = js_round(half + cb * half * 0.92);
            const double py = js_round(half - cr * half * 0.92);
            if (px < 0 || py < 0 || px >= kVectorSize || py >= kVectorSize) continue;
            a[std::size_t(py) * kVectorSize + std::size_t(px)]++;
        }
    });
    std::vector<double> acc(cells, 0.0);
    double peak = 0.0;
    for (std::size_t k = 0; k < cells; ++k) {
        for (int wk = 0; wk < workers; ++wk) acc[k] += double(wacc[std::size_t(wk)][k]);
        peak = std::max(peak, acc[k]);   // the page's running maximum is the final one
    }
    std::vector<std::uint8_t> img(acc.size() * 4, 0);
    const double scale = 1.0 / std::log1p(std::max(peak, 1.0));
    for (std::size_t j = 0; j < acc.size(); ++j) {
        const double a = acc[j] != 0.0 ? std::pow(std::log1p(acc[j]) * scale, 0.6) : 0.0;
        img[j * 4] = clamped_u8(158 * a);
        img[j * 4 + 1] = clamped_u8(242 * a);
        img[j * 4 + 2] = clamped_u8(255 * a);
        img[j * 4 + 3] = clamped_u8(std::min(255.0, a * 300));
    }
    return img;
}

}  // namespace rudra
