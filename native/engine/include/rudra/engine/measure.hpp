#pragma once
// The frame's measurements (Phase 3 step 8): the page's computeStats, run for
// the app on its own thread after a frame arrives or the grade changes. The
// same chain the viewer draws (composite.cpp, the corrected baseline) at full
// resolution, the exact reductions for MaxCLL and MaxFALL, the 768-side
// sample for the distributions, the scopes and the vectorscope, the mask and
// clip coverage -- every part of it held to the browser in Phase 2 -- and the
// full-resolution pictures kept for the probe.

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/readouts.hpp"
#include "rudra/core/scopes.hpp"

namespace rudra {

struct FrameMeasure {
    int width = 0, height = 0;
    NetworkLinearImage model, baseline;          // the composite and the baseline, full size
    std::vector<float> highlight, shadow;        // the masks, H x W
    std::vector<std::uint8_t> sdr8;              // the SDR codes, H x W x 3
    MaskCoverage coverage;                       // state.maskPct
    Measured measured;                           // metrics, scopes, the sample's luma
    std::vector<std::uint8_t> vector_rgba;       // drawVector's 256 x 256
    double compose_ms = 0.0;                     // the grade's cost, as the page times it

    FrameMetrics frame_metrics() const;          // showMetrics' input
    // The probe at a frame pixel (probeAt, floor of the position): nullopt
    // outside the frame.
    std::optional<ProbeInput> probe_at(double x, double y) const;
};

// `baseline` may be given when the caller has it already (it depends on the
// frame only, never on the grade).
FrameMeasure measure_frame(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                           const ModelConstants& model, const CompositeParams& params,
                           const NetworkLinearImage* baseline = nullptr);

}  // namespace rudra
