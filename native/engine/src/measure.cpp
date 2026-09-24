#include "rudra/engine/measure.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "rudra/core/baseline.hpp"
#include "rudra/core/view.hpp"

namespace rudra {

FrameMetrics FrameMeasure::frame_metrics() const {
    const ViewerMetrics& m = measured.metrics;
    FrameMetrics f;
    f.maxcll = m.maxcll;
    f.maxfall = m.maxfall;
    f.peak_nits = m.peak_nits;
    f.p99_nits = m.p99_nits;
    f.median_nits = m.median_nits;
    f.above_diffuse_white_pct = m.above_diffuse_white_pct;
    f.above_1000_nits_pct = m.above_1000_nits_pct;
    f.headroom_highlight_stops = m.headroom_highlight_stops;
    f.headroom_shadow_stops = m.headroom_shadow_stops;
    f.departure_rms_stops = m.departure_rms_stops;
    f.highlight_mask_pct = m.highlight_mask_pct;
    f.shadow_mask_pct = m.shadow_mask_pct;
    f.compose_ms = compose_ms;
    return f;
}

std::optional<ProbeInput> FrameMeasure::probe_at(double fx, double fy) const {
    const int x = int(std::floor(fx)), y = int(std::floor(fy));
    if (x < 0 || y < 0 || x >= width || y >= height) return std::nullopt;
    const Probe p = probe_pixel(model, baseline, x, y);
    ProbeInput in;
    in.x = x;
    in.y = y;
    in.model_nits = p.model.nits;
    in.baseline_nits = p.baseline.nits;
    const std::size_t i = std::size_t(y) * std::size_t(width) + std::size_t(x);
    if (sdr8.size() >= (i + 1) * 3) in.sdr = std::array<int, 3>{sdr8[i * 3], sdr8[i * 3 + 1], sdr8[i * 3 + 2]};
    if (i < highlight.size()) in.hi_mask = highlight[i];
    if (i < shadow.size()) in.sh_mask = shadow[i];
    return in;
}

FrameMeasure measure_frame(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                           const ModelConstants& model, const CompositeParams& params,
                           const NetworkLinearImage* baseline) {
    const auto t0 = std::chrono::steady_clock::now();
    FrameMeasure out;
    out.width = sdr.width();
    out.height = sdr.height();
    const std::size_t n = std::size_t(out.width) * std::size_t(out.height);
    out.baseline = baseline ? *baseline : corrected_baseline(sdr, model.corpus_ev, scalars.curve_params);
    out.model = composite(sdr, fields, scalars, model, params);
    out.highlight.assign(fields.highlight.plane(0), fields.highlight.plane(0) + n);
    out.shadow.assign(fields.shadow.plane(0), fields.shadow.plane(0) + n);
    out.sdr8.resize(n * 3);
    for (std::size_t i = 0; i < n; ++i)
        for (int c = 0; c < 3; ++c) {
            const float v = sdr.buffer().plane(c)[i];
            out.sdr8[i * 3 + std::size_t(c)] = std::uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
        }
    out.coverage = mask_coverage(out.highlight, out.shadow, out.sdr8);
    const SampleGrid grid = sample_grid(out.width, out.height);
    const PlanarBuffer model_sample = take_sample(out.model.buffer(), grid);
    const PlanarBuffer base_sample = take_sample(out.baseline.buffer(), grid);
    out.measured = measure_view(model_sample, base_sample, grid, out.highlight, out.shadow, out.coverage,
                                reduce_ladder(out.model.buffer()), reduce_ladder(out.baseline.buffer()), n);
    out.vector_rgba = vectorscope(model_sample);
    out.compose_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace rudra
