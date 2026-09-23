#include "rudra/core/hdr10.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {

// rudra/radiometry.py, as numpy uses them against float32 arrays: the Python
// floats are demoted to float32 in each operation (NEP 50).
constexpr double kM1 = 2610.0 / 16384.0, kM2 = 2523.0 / 32.0;
constexpr double kC1 = 3424.0 / 4096.0, kC2 = 2413.0 / 128.0, kC3 = 2392.0 / 128.0;
constexpr float kLuma2020f[3] = {0.2627f, 0.6780f, 0.0593f};
constexpr double kHlgA = 0.17883277;
constexpr double kHlgB = 1 - 4 * kHlgA;
const double kHlgC = 0.5 - kHlgA * std::log(4 * kHlgA);
constexpr double kLuma2020[3] = {0.2627, 0.6780, 0.0593};

}  // namespace

float pq_oetf(float nits) noexcept {
    const float v = std::clamp(nits / 10000.0f, 0.0f, 1.0f);
    const float p = std::pow(v, float(kM1));
    return std::pow((float(kC1) + float(kC2) * p) / (1.0f + float(kC3) * p), float(kM2));
}

float pq_eotf(float code) noexcept {
    const float v = std::clamp(code, 0.0f, 1.0f);
    const float p = std::pow(v, float(1.0 / kM2));
    const float ratio = std::max(p - float(kC1), 0.0f) / std::max(float(kC2) - float(kC3) * p, 1e-9f);
    return 10000.0f * std::pow(ratio, float(1.0 / kM1));
}

int pq12(double nits) noexcept {
    // np.round(pq_oetf(float64 -> float32) * 4095.0): a float32 product, rounded half to even.
    const float code = pq_oetf(static_cast<float>(nits)) * 4095.0f;
    const float r = std::nearbyint(code);
    return static_cast<int>(std::clamp(r, 0.0f, 4095.0f));
}

Result<PlanarBuffer> master_to_peak(const PlanarBuffer& in, double peak, std::optional<double> knee_nits) {
    if (!(peak >= 100.0 && peak <= 10000.0))
        return make_error(ErrorCode::InvalidArgument, "peak_nits must be between 100 and 10000");
    const double knee = knee_nits ? *knee_nits : 0.75 * peak;
    if (!(knee >= 0.0 && knee < peak))
        return make_error(ErrorCode::InvalidArgument, "knee_nits must be >= 0 and below peak_nits");
    const float kf = float(knee), span = float(peak - knee), pf = float(peak);
    PlanarBuffer out(3, in.height(), in.width());
    const std::size_t n = in.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        float px[3];
        for (int c = 0; c < 3; ++c) px[c] = std::max(in.plane(c)[i], 0.0f) * 10000.0f;
        const float lum = (px[0] * kLuma2020f[0] + px[1] * kLuma2020f[1]) + px[2] * kLuma2020f[2];
        const float compressed = lum <= kf ? lum : kf + span * (1.0f - std::exp(-(lum - kf) / span));
        const float k = compressed / std::max(lum, 1e-6f);
        for (float& v : px) v *= k;
        const float cmax = std::max({px[0], px[1], px[2]});
        const float s = std::min(1.0f, pf / std::max(cmax, 1e-6f));
        for (int c = 0; c < 3; ++c) out.plane(c)[i] = std::clamp(px[c] * s, 0.0f, pf);
    }
    return out;
}

Result<std::pair<PlanarBuffer, PlanarBuffer>> master_to_pq(const PlanarBuffer& in, double peak,
                                                           std::optional<double> knee) {
    auto m = master_to_peak(in, peak, knee);
    if (!m) return m.error();
    PlanarBuffer pq(3, in.height(), in.width());
    auto o = pq.span();
    const auto s = m->span();
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = pq_oetf(s[i]);
    return std::pair{std::move(pq), std::move(*m)};
}

double hlg_oetf(double scene) noexcept {
    scene = std::max(scene, 0.0);
    return scene <= 1.0 / 12.0 ? std::sqrt(3.0 * scene)
                               : kHlgA * std::log(std::max(12.0 * scene - kHlgB, 1e-12)) + kHlgC;
}

namespace {
double hlg_inverse(double code) noexcept {
    return code <= 0.5 ? code * code / 3.0 : (std::exp((code - kHlgC) / kHlgA) + kHlgB) / 12.0;
}
}  // namespace

const std::vector<DeliveryProfile>& delivery_profiles() {
    static const std::vector<DeliveryProfile> p{
        {"hdr10", "hevc", "libx265", "smpte2084", "yuv420p10le", -1, ""},
        {"hlg", "hevc", "libx265", "arib-std-b67", "yuv420p10le", -1, ""},
        {"prores422", "prores", "prores_ks", "smpte2084", "yuv422p10le", 2, "apcn"},
        {"prores422hq", "prores", "prores_ks", "smpte2084", "yuv422p10le", 3, "apch"},
        {"prores4444", "prores", "prores_ks", "smpte2084", "yuv444p10le", 4, "ap4h"},
    };
    return p;
}

Result<std::pair<PlanarBuffer, PlanarBuffer>> encode_master(const PlanarBuffer& in, const std::string& profile,
                                                            double peak, std::optional<double> knee) {
    auto m = master_to_peak(in, peak, knee);
    if (!m) return m.error();
    if (profile != "hlg") {
        PlanarBuffer pq(3, in.height(), in.width());
        auto o = pq.span();
        const auto s = m->span();
        for (std::size_t i = 0; i < o.size(); ++i) o[i] = pq_oetf(s[i]);
        return std::pair{std::move(pq), std::move(*m)};
    }
    // Display light -> scene light, gamma on luminance, then HLG; and the
    // display light that code decodes back to.
    const double gamma = 1.2 + 0.42 * std::log10(peak / 1000.0);
    PlanarBuffer code(3, in.height(), in.width()), disp(3, in.height(), in.width());
    const std::size_t n = in.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        double d[3];
        for (int c = 0; c < 3; ++c) d[c] = double(m->plane(c)[i]) / peak;
        const double lum = (d[0] * kLuma2020[0] + d[1] * kLuma2020[1]) + d[2] * kLuma2020[2];
        const double k = std::pow(std::max(lum, 1e-12), (1.0 - gamma) / gamma);
        double sc[3];
        for (int c = 0; c < 3; ++c) sc[c] = d[c] * k;
        const double mx = std::max({sc[0], sc[1], sc[2]});
        for (double& v : sc) v /= std::max(1.0, mx);
        double cv[3], back[3];
        for (int c = 0; c < 3; ++c) {
            cv[c] = std::clamp(hlg_oetf(sc[c]), 0.0, 1.0);
            back[c] = hlg_inverse(cv[c]);
        }
        const double blum = (back[0] * kLuma2020[0] + back[1] * kLuma2020[1]) + back[2] * kLuma2020[2];
        const double bk = std::pow(std::max(blum, 1e-12), gamma - 1.0);
        for (int c = 0; c < 3; ++c) {
            code.plane(c)[i] = static_cast<float>(cv[c]);
            disp.plane(c)[i] = static_cast<float>(back[c] * bk * peak);   // (scene * pow) * peak
        }
    }
    return std::pair{std::move(code), std::move(disp)};
}

}  // namespace rudra
