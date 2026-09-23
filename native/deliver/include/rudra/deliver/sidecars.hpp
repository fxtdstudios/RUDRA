#pragma once
// The three metadata sidecars next to a delivery (metadata.write_all_sidecars):
// <stem>_hdr_analysis.json, <stem>_dovi_generate.json,
// <stem>_hdr10plus_scenes.json, each byte-identical to the Python's.

#include <filesystem>
#include <span>

#include "rudra/core/measure.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct SidecarPaths {
    std::filesystem::path rudra, dovi, hdr10plus;
};

Result<SidecarPaths> write_all_sidecars(std::span<const FrameStats> stats, const std::filesystem::path& output_stem,
                                        double mastering_peak_nits = 1000.0, double shot_threshold = 0.35);

// Writes text exactly (UTF-8, no newline translation).
Result<void> write_text_file(const std::filesystem::path& path, const std::string& text);

}  // namespace rudra
