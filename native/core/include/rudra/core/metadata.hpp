#pragma once
// Dynamic HDR metadata from per-frame MaxRGB statistics. Port of
// rudra/delivery/metadata.py: shot detection, Dolby Vision L1 per shot, the
// dovi_tool generate config, HDR10+ scene statistics and the RUDRA sidecar,
// each built as the same JSON value the Python builds (byte-identical when
// dumped with rudra::pyjson).

#include <span>
#include <utility>
#include <vector>

#include "rudra/core/measure.hpp"
#include "rudra/platform/pyjson.hpp"

namespace rudra {

using Shot = std::pair<int, int>;   // (start, length)

// Histogram distance between consecutive frames above `threshold` is a cut.
std::vector<Shot> detect_shots(std::span<const FrameStats> stats, double threshold = 0.35);

pyjson::Value l1_per_shot(std::span<const FrameStats> stats, std::span<const Shot> shots);
pyjson::Value to_dovi_generate_json(std::span<const FrameStats> stats, std::span<const Shot> shots,
                                    double mastering_peak_nits = 1000.0, double mastering_min_nits = 0.0001);
pyjson::Value to_hdr10plus_json(std::span<const FrameStats> stats, std::span<const Shot> shots,
                                int target_display_nits = 400);
pyjson::Value to_rudra_sidecar(std::span<const FrameStats> stats, std::span<const Shot> shots);

}  // namespace rudra
