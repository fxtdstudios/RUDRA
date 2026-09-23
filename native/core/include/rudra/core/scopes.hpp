#pragma once
// The viewer's measurements and scopes (docs/view.spec.md sections 5 to 7):
// the capped point sample and its index, computeStats, buildScopes and the
// vectorscope of ui/app.js, ported so that they give the browser's numbers
// and bins exactly. Everything distributional runs on the sample; MaxCLL and
// MaxFALL come from the exact reductions (view.hpp reduce_ladder).

#include <array>
#include <cstdint>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/core/view.hpp"

namespace rudra {

inline constexpr int kSampleMaxSide = 768;
inline constexpr double kScopeLoNits = 0.05, kScopeHiNits = 4000.0;

// The sample grid and, per sample pixel, its source pixel (row-major index).
struct SampleGrid {
    int width = 0, height = 0;
    std::vector<int> index;
};
SampleGrid sample_grid(int width, int height);

// A composite target at the sample grid: 3 x sh x sw, the texels themselves.
PlanarBuffer take_sample(const PlanarBuffer& rgb, const SampleGrid& grid);

// Per-frame mask coverage at full resolution (adopt): shares in percent.
struct MaskCoverage {
    double highlight_pct = 0.0, shadow_pct = 0.0, clipped_pct = 0.0;
};
// `highlight`, `shadow`: H x W masks; `sdr8`: H x W x 3 interleaved 8-bit SDR.
MaskCoverage mask_coverage(const std::vector<float>& highlight, const std::vector<float>& shadow,
                           const std::vector<std::uint8_t>& sdr8);

struct ViewerMetrics {
    double maxcll = 0, maxfall = 0, peak_nits = 0, baseline_peak_nits = 0, headroom_stops = 0;
    double headroom_highlight_stops = 0, headroom_shadow_stops = 0;   // NaN when the mask is empty
    double departure_rms_stops = 0, p99_nits = 0, median_nits = 0;
    double above_diffuse_white_pct = 0, above_1000_nits_pct = 0;
    double highlight_mask_pct = 0, shadow_mask_pct = 0;
};

struct ScopeData {
    std::vector<double> lo, q1, mid, q3, hi;   // 230 columns, fractions of the log range
    std::vector<double> histogram;             // 76 bins, normalised to the tallest
};

struct Measured {
    ViewerMetrics metrics;
    ScopeData scopes;
    std::vector<float> luma;   // max(R, G, B) in nits per sample pixel, fp32 (the page's `luma`)
};

// computeStats and buildScopes on one frame's sample. `model_sample` and
// `base_sample` come from take_sample; the masks are full resolution (H x W);
// `model`, `base` are the exact reductions of the full targets.
Measured measure_view(const PlanarBuffer& model_sample, const PlanarBuffer& base_sample, const SampleGrid& grid,
                      const std::vector<float>& highlight, const std::vector<float>& shadow,
                      const MaskCoverage& coverage, const Reductions& model, const Reductions& base,
                      std::size_t full_pixels);

ScopeData build_scopes(const std::vector<float>& luma, int width, int height);

// drawVector: 256 x 256 RGBA8 of the model sample.
inline constexpr int kVectorSize = 256;
std::vector<std::uint8_t> vectorscope(const PlanarBuffer& model_sample);

}  // namespace rudra
