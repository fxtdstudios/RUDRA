#pragma once
// Measurements: what the QC panel, the sidecar and the HDR10 metadata report.
// Port of rudra/delivery/metadata.py (analyze_frame, maxcll_maxfall) and of
// ui/server.py measure(). Raw values; rounding is the page's business.

#include <array>
#include <span>

#include "rudra/core/image.hpp"
#include "rudra/core/master.hpp"

namespace rudra {

inline constexpr std::array<double, 9> kStatPercentiles{1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.98};
inline constexpr int kLogHistBins = 64;
inline constexpr double kLogHistLo = -14.0, kLogHistHi = 21.0;

// MaxRGB statistics of one frame in absolute nits.
struct FrameStats {
    double min_nits = 0.0, avg_nits = 0.0, max_nits = 0.0;
    std::array<double, 3> maxscl_nits{};
    std::array<double, kStatPercentiles.size()> percentiles_nits{};
    std::array<double, kLogHistBins> log_hist{};    // fraction of pixels per bin
};

FrameStats analyze_frame(const NitsFrame& nits);

// CTA-861.3 static metadata: ceil of the brightest frame's MaxRGB peak and of
// the brightest frame-average MaxRGB.
struct StaticMetadata {
    int maxcll = 0;
    int maxfall = 0;
};

StaticMetadata maxcll_maxfall(std::span<const FrameStats> frames);

// np.percentile(values, q) with the default linear method. Sorts `values`.
double percentile_linear(std::vector<double>& sorted_or_not, double q, bool already_sorted = false);

// The Studio's QC numbers for one composite against its baseline.
struct StudioMeasure {
    int maxcll = 0, maxfall = 0;
    double peak_nits = 0.0, baseline_peak_nits = 0.0;
    double headroom_stops = 0.0;
    double headroom_highlight_stops = 0.0;   // NaN when the mask is empty
    double headroom_shadow_stops = 0.0;      // NaN when the mask is empty
    double departure_rms_stops = 0.0;
    double p99_nits = 0.0, median_nits = 0.0;
    double above_diffuse_white_pct = 0.0, above_1000_nits_pct = 0.0;
    double highlight_mask_pct = 0.0, shadow_mask_pct = 0.0;
};

StudioMeasure measure(const NetworkLinearImage& hdr, const NetworkLinearImage& baseline,
                      const PlanarBuffer& highlight_mask, const PlanarBuffer& shadow_mask);

}  // namespace rudra
