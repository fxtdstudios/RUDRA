#include "rudra/core/grade.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {
// rudra/hdr10.py REC2020_LUMA: float32 weights, widened when multiplied by
// float64 nits.
constexpr double kLuma2020f[3] = {double(0.2627f), double(0.6780f), double(0.0593f)};

double luma(const double px[3]) noexcept {
    return (px[0] * kLuma2020f[0] + px[1] * kLuma2020f[1]) + px[2] * kLuma2020f[2];
}
}  // namespace

Result<void> shoulder_to_peak(NitsFrame& nits, double peak, std::optional<double> knee_nits) {
    const double knee = knee_nits ? *knee_nits : 0.75 * peak;
    if (!(knee >= 0.0 && knee < peak))
        return make_error(ErrorCode::InvalidArgument, "knee_nits must be >= 0 and below peak_nits");
    const double span = peak - knee;
    const std::size_t n = nits.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        double px[3];
        for (int c = 0; c < 3; ++c) px[c] = std::max(nits.plane(c)[i], 0.0);
        const double lum = luma(px);
        const double compressed = lum <= knee ? lum : knee + span * (1.0 - std::exp(-(lum - knee) / span));
        const double k = compressed / std::max(lum, 1e-6);
        for (double& v : px) v = v * k;
        const double cmax = std::max({px[0], px[1], px[2]});
        const double s = std::min(1.0, peak / std::max(cmax, 1e-6));
        for (int c = 0; c < 3; ++c) nits.plane(c)[i] = std::clamp(px[c] * s, 0.0, peak);
    }
    return {};
}

Result<PlanarBuffer> apply_grade(const NitsFrame& in, const GradeControls& g) {
    if (!(g.peak_nits >= 100.0 && g.peak_nits <= 10000.0))
        return make_error(ErrorCode::InvalidArgument, "peak_nits must be between 100 and 10000");
    NitsFrame f = in;
    for (double& v : f.span()) v = std::max(v, 0.0);
    const std::size_t n = f.plane_size();

    if (g.exposure_ev != 0.0) {
        const double k = std::pow(2.0, g.exposure_ev);
        for (double& v : f.span()) v = v * k;
    }
    for (const auto& r : g.regions) {
        if (r.mask.height() != f.height() || r.mask.width() != f.width())
            return make_error(ErrorCode::InvalidArgument, "region mask does not match the frame", r.label);
        for (std::size_t i = 0; i < n; ++i) {
            const double m = std::clamp(double(r.mask.plane(0)[i]), 0.0, 1.0);
            const double k = std::pow(2.0, r.ev * m);
            for (int c = 0; c < 3; ++c) f.plane(c)[i] *= k;
        }
    }
    if (g.highlight_desat > 0.0) {
        const double knee = g.knee_nits ? *g.knee_nits : 0.75 * g.peak_nits;
        const double amt0 = std::clamp(g.highlight_desat, 0.0, 1.0);
        const double denom = std::max(g.peak_nits - knee, 1e-6);
        for (std::size_t i = 0; i < n; ++i) {
            double px[3] = {f.plane(0)[i], f.plane(1)[i], f.plane(2)[i]};
            const double lum = luma(px);
            const double above = std::clamp((lum - knee) / denom, 0.0, 1.0);
            const double amount = amt0 * above;
            for (int c = 0; c < 3; ++c) f.plane(c)[i] = px[c] + amount * (lum - px[c]);
        }
    }
    if (auto s = shoulder_to_peak(f, g.peak_nits, g.knee_nits); !s) return s.error();
    PlanarBuffer out(3, f.height(), f.width());
    auto o = out.span();
    const auto src = f.span();
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = static_cast<float>(src[i]);
    return out;
}

Result<PlanarBuffer> itm_strength_map(int h, int w, double base, const std::vector<MaskRegion>& regions) {
    std::vector<double> s(std::size_t(h) * w, base);
    for (const auto& r : regions) {
        if (r.mask.height() != h || r.mask.width() != w)
            return make_error(ErrorCode::InvalidArgument, "region mask does not match the frame", r.label);
        for (std::size_t i = 0; i < s.size(); ++i)
            s[i] = s[i] * std::pow(2.0, r.ev * std::clamp(double(r.mask.plane(0)[i]), 0.0, 1.0));
    }
    PlanarBuffer out(1, h, w);
    for (std::size_t i = 0; i < s.size(); ++i) out.plane(0)[i] = static_cast<float>(std::clamp(s[i], 0.0, 2.0));
    return out;
}

}  // namespace rudra
