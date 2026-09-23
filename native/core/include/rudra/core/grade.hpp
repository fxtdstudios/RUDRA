#pragma once
// Artist grade controls on absolute nits. Port of rudra/delivery/controls.py
// apply_grade and itm_strength_map, in float64 like the Python, same fixed
// order: exposure, region EV (soft masks), highlight desaturation, shoulder.

#include <optional>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/core/master.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// A soft mask (1 x h x w, [0, 1]) and the EV push applied where it is 1.
struct MaskRegion {
    PlanarBuffer mask;
    double ev = 0.0;
    std::string label;
};

struct GradeControls {
    double exposure_ev = 0.0;
    double peak_nits = 1000.0;
    std::optional<double> knee_nits;   // default: 0.75 * peak
    double highlight_desat = 0.0;      // 0..1, above the knee
    std::vector<MaskRegion> regions;
};

// Hue-preserving exponential shoulder to peak (controls._shoulder_to_peak),
// in place, float64.
Result<void> shoulder_to_peak(NitsFrame& nits, double peak_nits, std::optional<double> knee_nits);

// apply_grade: 3 x h x w float32 nits, as the Python's .astype(np.float32).
Result<PlanarBuffer> apply_grade(const NitsFrame& nits, const GradeControls& controls);

// Per-pixel residual strength for the network: base * prod 2^(ev * mask),
// clamped to [0, 2], float32 (1 x h x w).
Result<PlanarBuffer> itm_strength_map(int height, int width, double base_strength,
                                      const std::vector<MaskRegion>& regions);

}  // namespace rudra
