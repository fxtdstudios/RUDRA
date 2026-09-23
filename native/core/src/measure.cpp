#include "rudra/core/measure.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

namespace rudra {
namespace {

// Neumaier-compensated sum: numpy's pairwise sum is within a few ulp of the
// exact sum, and so is this; a naive loop over 8 million pixels is not.
struct Sum {
    double s = 0.0, c = 0.0;
    void add(double v) noexcept {
        const double t = s + v;
        c += std::abs(s) >= std::abs(v) ? (s - t) + v : (v - t) + s;
        s = t;
    }
    double value() const noexcept { return s + c; }
};

// numpy's _lerp: a + (b - a) t, evaluated from b when t >= 0.5.
double lerp_np(double a, double b, double t) noexcept {
    const double d = b - a;
    return t >= 0.5 ? b - d * (1.0 - t) : a + d * t;
}

double clean_nits(double v) noexcept {
    if (std::isnan(v)) return 0.0;
    if (v == std::numeric_limits<double>::infinity()) return 10000.0;
    return std::max(v, 0.0);   // -inf -> the most negative double -> 0
}

}  // namespace

double percentile_linear(std::vector<double>& v, double q, bool already_sorted) {
    assert(!v.empty());
    if (!already_sorted) std::sort(v.begin(), v.end());
    const double idx = (q / 100.0) * static_cast<double>(v.size() - 1);
    const double lo_f = std::floor(idx);
    const std::size_t lo = static_cast<std::size_t>(lo_f);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    return lerp_np(v[lo], v[hi], idx - lo_f);
}

FrameStats analyze_frame(const NitsFrame& nits) {
    const std::size_t n = nits.plane_size();
    assert(n > 0);
    FrameStats st;
    std::vector<double> max_rgb(n);
    std::array<double, 3> maxscl{0.0, 0.0, 0.0};
    Sum sum;
    std::array<std::size_t, kLogHistBins> counts{};
    // np.histogram with uniform bins: index from the scaled value, then
    // corrected against the linspace edges it would compare with.
    std::array<double, kLogHistBins + 1> edges{};
    for (int i = 0; i <= kLogHistBins; ++i) {
        const double step = (kLogHistHi - kLogHistLo) / kLogHistBins;
        edges[static_cast<std::size_t>(i)] = i == kLogHistBins ? kLogHistHi : kLogHistLo + i * step;
    }

    double mn = std::numeric_limits<double>::infinity(), mx = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double m = 0.0;
        for (int c = 0; c < 3; ++c) {
            const double v = clean_nits(nits.plane(c)[i]);
            maxscl[static_cast<std::size_t>(c)] = std::max(maxscl[static_cast<std::size_t>(c)], v);
            m = std::max(m, v);
        }
        max_rgb[i] = m;
        mn = std::min(mn, m);
        mx = std::max(mx, m);
        sum.add(m);
        const double l = std::log2(std::max(m, 1e-4));
        if (l >= kLogHistLo && l <= kLogHistHi) {
            auto b = static_cast<int>((l - kLogHistLo) / (kLogHistHi - kLogHistLo) * kLogHistBins);
            if (b == kLogHistBins) b -= 1;
            if (l < edges[static_cast<std::size_t>(b)]) b -= 1;
            else if (b != kLogHistBins - 1 && l >= edges[static_cast<std::size_t>(b) + 1]) b += 1;
            counts[static_cast<std::size_t>(b)] += 1;
        }
    }
    st.min_nits = mn;
    st.max_nits = mx;
    st.avg_nits = sum.value() / static_cast<double>(n);
    st.maxscl_nits = maxscl;
    for (int i = 0; i < kLogHistBins; ++i)
        st.log_hist[static_cast<std::size_t>(i)] = static_cast<double>(counts[static_cast<std::size_t>(i)]) / static_cast<double>(n);
    std::sort(max_rgb.begin(), max_rgb.end());
    for (std::size_t k = 0; k < kStatPercentiles.size(); ++k)
        st.percentiles_nits[k] = percentile_linear(max_rgb, kStatPercentiles[k], true);
    return st;
}

StaticMetadata maxcll_maxfall(std::span<const FrameStats> frames) {
    assert(!frames.empty());
    double cll = 0.0, fall = 0.0;
    for (const auto& f : frames) {
        cll = std::max(cll, f.max_nits);
        fall = std::max(fall, f.avg_nits);
    }
    return {static_cast<int>(std::ceil(cll)), static_cast<int>(std::ceil(fall))};
}

StudioMeasure measure(const NetworkLinearImage& hdr, const NetworkLinearImage& baseline,
                      const PlanarBuffer& highlight_mask, const PlanarBuffer& shadow_mask) {
    const NitsFrame nits = nits_from_network(hdr);
    const NitsFrame base = nits_from_network(baseline);
    const std::size_t n = nits.plane_size();
    assert(base.plane_size() == n && highlight_mask.plane_size() == n && shadow_mask.plane_size() == n);

    StudioMeasure r;
    const FrameStats st = analyze_frame(nits);
    const StaticMetadata md = maxcll_maxfall(std::span(&st, 1));
    r.maxcll = md.maxcll;
    r.maxfall = md.maxfall;

    double peak = -std::numeric_limits<double>::infinity(), base_peak = peak;
    for (double v : nits.span()) peak = std::max(peak, v);
    for (double v : base.span()) base_peak = std::max(base_peak, v);
    r.peak_nits = peak;
    r.baseline_peak_nits = base_peak;
    r.headroom_stops = std::log2(std::max(peak, 1e-6) / std::max(base_peak, 1e-6));

    Sum hl_luma, hl_base, sh_luma, sh_base, sq;
    std::size_t hl_n = 0, sh_n = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double luma = std::max({nits.plane(0)[i], nits.plane(1)[i], nits.plane(2)[i]});
        const double bl = std::max({base.plane(0)[i], base.plane(1)[i], base.plane(2)[i]});
        if (highlight_mask.plane(0)[i] > 0.5f) { hl_luma.add(luma); hl_base.add(bl); ++hl_n; }
        if (shadow_mask.plane(0)[i] > 0.5f) { sh_luma.add(luma); sh_base.add(bl); ++sh_n; }
        const double ratio = std::log2((luma + 1e-4) / (bl + 1e-4));
        sq.add(ratio * ratio);
    }
    auto stops_in = [](const Sum& a, const Sum& b, std::size_t count) {
        if (count == 0) return std::numeric_limits<double>::quiet_NaN();
        const double c = static_cast<double>(count);
        return std::log2(std::max(a.value() / c, 1e-6) / std::max(b.value() / c, 1e-6));
    };
    r.headroom_highlight_stops = stops_in(hl_luma, hl_base, hl_n);
    r.headroom_shadow_stops = stops_in(sh_luma, sh_base, sh_n);
    r.departure_rms_stops = std::sqrt(sq.value() / static_cast<double>(n));

    std::vector<double> all(nits.span().begin(), nits.span().end());
    std::size_t above_white = 0, above_1000 = 0;
    for (double v : all) {
        above_white += v > 203.0 * (1.0 + 1e-6);
        above_1000 += v > 1000.0;
    }
    std::sort(all.begin(), all.end());
    r.p99_nits = percentile_linear(all, 99.0, true);
    // np.median, not the 50th percentile: the mean of the two middle values.
    const std::size_t mid = all.size() / 2;
    r.median_nits = all.size() % 2 ? all[mid] : (all[mid - 1] + all[mid]) / 2.0;
    const double total = static_cast<double>(all.size());
    r.above_diffuse_white_pct = 100.0 * static_cast<double>(above_white) / total;
    r.above_1000_nits_pct = 100.0 * static_cast<double>(above_1000) / total;
    r.highlight_mask_pct = 100.0 * static_cast<double>(hl_n) / static_cast<double>(n);
    r.shadow_mask_pct = 100.0 * static_cast<double>(sh_n) / static_cast<double>(n);
    return r;
}

}  // namespace rudra
