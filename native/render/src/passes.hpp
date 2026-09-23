#pragma once
// Internal to render/: the uniform blocks of the viewer's shaders (std140) and
// how they are filled, shared by GpuCompositor (the parity path) and
// ViewerWindow (the live viewer), so the two cannot drift apart.

#include <QFile>
#include <rhi/qrhi.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/view.hpp"

namespace rudra::detail {

inline QShader load_shader(const QString& name) {
    QFile f(name);
    return f.open(QIODevice::ReadOnly) ? QShader::fromSerialized(f.readAll()) : QShader();
}

// The Composite block of shaders/composite.frag.
struct CompositeUbo {
    float model[4];
    float control[4];
    float counts[4];
    float curve[36];
    float bands[32];
};
static_assert(sizeof(CompositeUbo) == (3 * 4 + 36 + 32) * sizeof(float));

// The View block of shaders/display.frag.
struct ViewUbo {
    float view[4];
    float extra[4];
    float target[4];
    float pic[12];   // three vec4 rows
    float gfx[12];
};
static_assert(sizeof(ViewUbo) == 36 * sizeof(float));

// The Reduce block of shaders/reduce.frag.
struct ReduceUbo {
    float sizes[4];
};

inline void rows_of(const Mat3& m, float* out) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) out[r * 4 + c] = float(m[std::size_t(r)][std::size_t(c)]);
}

inline CompositeUbo composite_ubo(const FrameScalars& scalars, const ModelConstants& model,
                                  const CompositeParams& params) {
    CompositeUbo u{};
    u.model[0] = model.log_scale;
    u.model[1] = model.max_hdr;
    u.model[2] = static_cast<float>(std::exp2(-static_cast<double>(model.corpus_ev)) * (203.0 / 10000.0));
    u.model[3] = params.strength;
    u.control[0] = float(int(params.mode));
    u.control[1] = params.preserve_outside ? 1.0f : 0.0f;
    u.control[2] = scalars.shadow_weight;
    u.control[3] = static_cast<float>(params.region_softness_stops);
    const std::size_t np = std::min<std::size_t>(scalars.curve_params.size(), 36);
    u.counts[0] = np >= 3 ? float(np - 1) : 0.0f;
    std::copy(scalars.curve_params.begin(), scalars.curve_params.begin() + std::ptrdiff_t(np), u.curve);
    int bands = 0;
    for (const auto& band : params.regions) {
        if (band.ev == 0.0 || bands == 8) continue;
        u.bands[bands * 4 + 0] = static_cast<float>(std::log2(band.low_nits));
        u.bands[bands * 4 + 1] = static_cast<float>(std::log2(band.high_nits));
        u.bands[bands * 4 + 2] = static_cast<float>(band.ev);
        ++bands;
    }
    u.counts[1] = float(bands);
    return u;
}

// The uniforms as core/view.cpp rounds them to fp32.
inline ViewUbo view_ubo(const ViewParams& params, int width) {
    ViewUbo u{};
    u.view[0] = float(int(params.mode));
    u.view[1] = float(10000.0 / std::max(params.display_nits, 1e-3));
    u.view[2] = params.wipe >= 0.0 ? float(std::clamp(params.wipe, 0.0, 1.0)) : -1.0f;
    u.view[3] = float(params.wipe_half_width);
    u.extra[0] = std::log2(1.0f + float(std::max(params.diff_gain, 1.0)));
    u.extra[1] = float(width);
    u.extra[2] = params.show == ViewSource::Baseline ? 1.0f : 0.0f;
    u.target[0] = float(int(params.target.path));
    u.target[1] = float(std::min(params.display_nits, params.target.peak_nits));
    u.target[2] = float(params.target.unit_nits);
    rows_of(rgb_to_rgb_matrix(params.source, params.target.primaries), u.pic);
    rows_of(rgb_to_rgb_matrix(Primaries::Rec709, params.target.primaries), u.gfx);
    return u;
}

// sdr rgb + shadow in a, and residual rgb + highlight in a: the two inputs
// composite.frag fetches.
inline void interleave_inputs(const SdrImage& sdr, const Fields& fields, std::vector<float>& a, std::vector<float>& b) {
    const std::size_t n = std::size_t(sdr.width()) * std::size_t(sdr.height());
    a.assign(n * 4, 0.0f);
    b.assign(n * 4, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            a[i * 4 + std::size_t(c)] = sdr.buffer().plane(c)[i];
            b[i * 4 + std::size_t(c)] = fields.residual.plane(c)[i];
        }
        a[i * 4 + 3] = fields.shadow.plane(0)[i];
        b[i * 4 + 3] = fields.highlight.plane(0)[i];
    }
}

// A composite target as an RGBA32F upload: rgb, alpha 1.
inline std::vector<float> rgba_of(const PlanarBuffer& rgb) {
    const std::size_t n = rgb.plane_size();
    std::vector<float> out(n * 4, 1.0f);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) out[i * 4 + std::size_t(c)] = rgb.plane(c)[i];
    return out;
}

}  // namespace rudra::detail
