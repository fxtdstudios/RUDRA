#pragma once
// check_alpha (Phase 4, step 6): every decoded alpha pixel of a ProRes 4444
// master against the untouched spool it was encoded from. The master's alpha
// is extracted by ffmpeg as 16-bit grey on a pipe, frame by frame, and each
// frame compared with the spool PNG's fourth channel; the largest difference
// in 16-bit codes is returned (the verdict, at most 128, is deliver's).

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

std::vector<std::string> alpha_extract_command(const std::string& ffmpeg, const std::filesystem::path& output);

// "Missing decoded alpha frame", "Unexpected extra alpha frames" and "Alpha
// decode failed" as the Python says them; stderr goes to spool/alpha_qc.log.
Result<int> alpha_max_error(const std::filesystem::path& output, const std::filesystem::path& spool, int frame_count,
                            int width, int height);

}  // namespace rudra
