#include "rudra/core/composite.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "rudra/core/baseline.hpp"

namespace rudra {
namespace {

// rudra/hdr10.py REC2020_LUMA is a float32 array; controls.py multiplies
// float64 nits by it, so the weights are the float32-rounded values, widened.
constexpr double kLuma2020f[3] = {static_cast<double>(0.2627f), static_cast<double>(0.6780f),
                                  static_cast<double>(0.0593f)};

inline float sigmoidf(float x) noexcept { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

std::vector<RegionBand> default_region_bands() {
    return {{400.0, 2000.0, 0.0}, {2000.0, 8000.0, 0.0}, {0.05, 12.0, 0.0}};
}

bool any_graded(std::span<const RegionBand> bands) noexcept {
    return std::any_of(bands.begin(), bands.end(), [](const RegionBand& b) { return b.ev != 0.0; });
}

float qualifier_mask(double luma_nits, double low_nits, double high_nits, double softness_stops) noexcept {
    const double y = std::max(luma_nits, 1e-6);
    const double log_y = std::log2(y);
    const double soft = std::max(softness_stops, 1e-3);
    const double rise = std::clamp((log_y - (std::log2(low_nits) - soft)) / soft, 0.0, 1.0);
    const double fall = std::clamp(((std::log2(high_nits) + soft) - log_y) / soft, 0.0, 1.0);
    const double m = std::min(rise, fall);
    return static_cast<float>(m * m * (3.0 - 2.0 * m));
}

double region_ev_gain(const double rgb_nits[3], std::span<const RegionBand> bands, double softness_stops) noexcept {
    double y = 0.0;
    for (int c = 0; c < 3; ++c) y += std::max(rgb_nits[c], 0.0) * kLuma2020f[c];
    double total = 0.0;
    for (const auto& b : bands) {
        if (b.ev == 0.0) continue;
        // numpy: a Python float times a float32 array is a float32 product
        // (NEP 50), widened only when it is added to the float64 total.
        const float term = static_cast<float>(b.ev) * qualifier_mask(y, b.low_nits, b.high_nits, softness_stops);
        total += static_cast<double>(term);
    }
    return std::exp2(total);
}

NetworkLinearImage composite(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                             const ModelConstants& model, const CompositeParams& params) {
    const int h = sdr.height(), w = sdr.width();
    assert(fields.residual.channels() == 3 && fields.residual.height() == h && fields.residual.width() == w);
    assert(fields.highlight.height() == h && fields.shadow.height() == h);

    // The baseline reads the clamped SDR, as forward() does.
    PlanarBuffer clamped = sdr.buffer();
    for (float& v : clamped.span()) v = std::clamp(v, 0.0f, 1.0f);
    const SdrImage sdr_c(std::move(clamped));
    const NetworkLinearImage base = corrected_baseline(sdr_c, model.corpus_ev, scalars.curve_params);

    const float log_scale = model.log_scale;
    const float log_ceiling = std::log1p(model.max_hdr * log_scale);
    const bool graded = any_graded(params.regions);
    const double ceiling_nits = static_cast<double>(model.max_hdr) * 10000.0;

    NetworkLinearImage out(h, w);
    const std::size_t n = static_cast<std::size_t>(h) * w;
    const PlanarBuffer& s = sdr_c.buffer();
    const PlanarBuffer& b = base.buffer();
    const PlanarBuffer& r = fields.residual;
    const float* hl = fields.highlight.plane(0);
    const float* sh = fields.shadow.plane(0);
    PlanarBuffer& o = out.buffer();

    for (std::size_t i = 0; i < n; ++i) {
        const float y = 0.2126f * s.plane(0)[i] + 0.7152f * s.plane(1)[i] + 0.0722f * s.plane(2)[i];
        const float hp = sigmoidf((y - 0.82f) * 24.0f);
        const float sp = sigmoidf((0.10f - y) * 24.0f) * scalars.shadow_weight;
        float gate = 0.0f;
        switch (params.mode) {
            case RecoveryMode::All: gate = std::max(hp, sp); break;
            case RecoveryMode::Highlights: gate = hp; break;
            case RecoveryMode::Shadows: gate = sp; break;
            case RecoveryMode::Off: gate = 0.0f; break;
        }
        gate *= params.strength;
        const float recovery = std::max(hl[i], sh[i]);
        float px[3];
        for (int c = 0; c < 3; ++c) {
            const float base_c = b.plane(c)[i];
            const float pred_log = std::clamp(std::log1p(base_c * log_scale) + r.plane(c)[i] * gate, 0.0f, log_ceiling);
            float pred = std::expm1(pred_log) / log_scale;
            if (params.preserve_outside) pred = base_c + recovery * (pred - base_c);
            px[c] = pred;
        }
        if (graded) {
            double nits[3] = {px[0] * 10000.0, px[1] * 10000.0, px[2] * 10000.0};
            const double g = region_ev_gain(nits, params.regions, params.region_softness_stops);
            for (int c = 0; c < 3; ++c)
                px[c] = static_cast<float>(std::clamp(nits[c] * g, 0.0, ceiling_nits) / 10000.0);
        }
        for (int c = 0; c < 3; ++c) o.plane(c)[i] = px[c];
    }
    return out;
}

}  // namespace rudra
