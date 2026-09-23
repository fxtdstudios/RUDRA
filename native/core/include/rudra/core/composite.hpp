#pragma once
// The composite: picture = composite(sdr, fields, scalars, params). A pure
// function, the one every user control lives in (NATIVE_ARCHITECTURE.md 4 and
// docs/composite.spec.md). It exists once as a spec and twice as code: this
// file (fp32, the CPU path and the reference) and the render layer's shader.
//
// Port of the tail of rudra/sdr2hdr.py SDR2HDRNet.forward (as fed by
// training/infer_sdr2hdr.py predict_fields) plus Region EV from
// rudra/delivery/controls.py, which is also what ui/compositor.js runs.

#include <span>
#include <vector>

#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

enum class RecoveryMode : std::uint8_t { All, Highlights, Shadows, Off };

// The three constants a model package carries (manifest.json).
struct ModelConstants {
    float log_scale = 16.0f;
    float max_hdr = 4.0f;       // network units: 4.0 = 40 000 nits
    float corpus_ev = -1.0f;
};

// One luminance-qualified EV band, in absolute nits (Rec.2020 luma).
struct RegionBand {
    double low_nits = 0.0;
    double high_nits = 0.0;
    double ev = 0.0;
};

// The bands the Region EV panel opens with. Neutral: ev 0 everywhere.
std::vector<RegionBand> default_region_bands();

bool any_graded(std::span<const RegionBand> bands) noexcept;

struct CompositeParams {
    RecoveryMode mode = RecoveryMode::All;
    float strength = 1.0f;
    bool preserve_outside = true;
    std::vector<RegionBand> regions;       // empty or all ev 0: no grade
    double region_softness_stops = 1.0;
};

// qualifier_mask for one pixel's Rec.2020 luminance in nits. float32 result,
// exactly as the Python casts it before the gain sums it.
float qualifier_mask(double luma_nits, double low_nits, double high_nits, double softness_stops) noexcept;

// region_ev_gain for one pixel in absolute nits: 2^(sum ev * mask).
double region_ev_gain(const double rgb_nits[3], std::span<const RegionBand> bands, double softness_stops) noexcept;

// The full composite into the network convention (1.0 = 10 000 nits). With a
// grade, Region EV is applied and the result clamped to [0, max_hdr].
NetworkLinearImage composite(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                             const ModelConstants& model, const CompositeParams& params);

}  // namespace rudra
