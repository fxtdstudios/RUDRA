#include "rudra/core/view.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rudra {
namespace {

constexpr float kPeak = 10000.0f;   // network units to nits

// Zone upper bounds (nits) and colours, docs/view.spec.md section 2.
constexpr std::array<float, 9> kZoneBelow{0.1f, 1.0f, 10.0f, 100.0f, 160.0f, 250.0f, 1000.0f, 4000.0f, 10000.0f};
constexpr std::array<std::array<float, 3>, 10> kZoneColour{{
    {0.169f, 0.122f, 0.239f}, {0.184f, 0.294f, 0.561f}, {0.184f, 0.561f, 0.722f},
    {0.200f, 0.627f, 0.416f}, {0.604f, 0.655f, 0.698f}, {0.914f, 0.929f, 0.945f},
    {0.910f, 0.765f, 0.290f}, {0.910f, 0.529f, 0.227f}, {0.816f, 0.263f, 0.184f},
    {0.780f, 0.290f, 0.780f},
}};

float lum2020(float r, float g, float b) noexcept {
    return r * kRec2020Luma[0] + g * kRec2020Luma[1] + b * kRec2020Luma[2];
}

}  // namespace

int false_colour_zone(float nits) noexcept {
    for (int i = 0; i < int(kZoneBelow.size()); ++i)
        if (nits < kZoneBelow[std::size_t(i)]) return i;
    return int(kZoneBelow.size());   // NaN lands here too, as in the shader
}

std::array<float, 3> false_colour(int zone) noexcept {
    return kZoneColour[std::size_t(std::clamp(zone, 0, int(kZoneColour.size()) - 1))];
}

float linear_to_srgb(float x) noexcept {
    x = std::clamp(x, 0.0f, 1.0f);
    return x > 0.0031308f ? 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f : 12.92f * x;
}

PlanarBuffer render_view(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                         const ViewParams& p) {
    assert(model.width() == baseline.width() && model.height() == baseline.height());
    const int w = model.width(), h = model.height();
    const PlanarBuffer& m = model.buffer();
    const PlanarBuffer& b = baseline.buffer();
    const bool wiping = p.wipe >= 0.0;
    const PlanarBuffer& source = (!wiping && p.show == ViewSource::Baseline) ? b : m;
    // Uniforms reach the shader as fp32.
    const float scale = float(double(kPeak) / std::max(p.display_nits, 1e-3));
    const float wipe = wiping ? float(std::clamp(p.wipe, 0.0, 1.0)) : -1.0f;
    const float half_width = float(p.wipe_half_width);
    const float log_gain = std::log2(1.0f + float(std::max(p.diff_gain, 1.0)));

    PlanarBuffer out(3, h, w);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float u = (float(x) + 0.5f) / float(w);
            const PlanarBuffer& pic = (wiping && u < wipe) ? b : source;
            const float hr = pic.at(0, y, x), hg = pic.at(1, y, x), hb = pic.at(2, y, x);
            float c[3];
            switch (p.mode) {
                case ViewMode::FalseColour: {
                    const auto fc = false_colour(false_colour_zone(lum2020(hr, hg, hb) * kPeak));
                    c[0] = fc[0], c[1] = fc[1], c[2] = fc[2];
                    break;
                }
                case ViewMode::Difference: {
                    const float d = lum2020(std::abs(source.at(0, y, x) - b.at(0, y, x)),
                                            std::abs(source.at(1, y, x) - b.at(1, y, x)),
                                            std::abs(source.at(2, y, x) - b.at(2, y, x)));
                    const float v = std::clamp(std::log2(1.0f + d * kPeak) / log_gain, 0.0f, 1.0f);
                    c[0] = v * 0.95f, c[1] = v * 0.62f, c[2] = v * 0.28f;
                    break;
                }
                case ViewMode::Image:
                default:
                    c[0] = linear_to_srgb(hr * scale);
                    c[1] = linear_to_srgb(hg * scale);
                    c[2] = linear_to_srgb(hb * scale);
                    break;
            }
            if (wiping && std::abs(u - wipe) < half_width)
                for (float& v : c) v = 1.0f - v;
            for (int k = 0; k < 3; ++k) out.at(k, y, x) = c[k];
        }
    }
    return out;
}

Rgb8Image render_view_rgb8(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                           const ViewParams& p) {
    const PlanarBuffer f = render_view(model, baseline, p);
    Rgb8Image out{f.width(), f.height(), std::vector<std::uint8_t>(f.plane_size() * 3)};
    const std::size_t n = f.plane_size();
    for (std::size_t i = 0; i < n; ++i)
        for (int k = 0; k < 3; ++k) {
            const float v = std::clamp(f.plane(k)[i], 0.0f, 1.0f);
            out.rgb[i * 3 + std::size_t(k)] = std::uint8_t(std::lround(v * 255.0f));
        }
    return out;
}

}  // namespace rudra
