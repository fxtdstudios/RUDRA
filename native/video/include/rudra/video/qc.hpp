#pragma once
// quality_check and check_alpha as convert_video runs them: the master probed
// (media), the checks (deliver/video_qc), the complete decode, then the
// verdict; and the alpha of a ProRes 4444 master against its spool.

#include <filesystem>
#include <string_view>

#include "rudra/deliver/video_qc.hpp"

namespace rudra {

Result<pyjson::Value> run_quality_check(std::string_view source_streams_json, const VideoClock& clock,
                                        const std::filesystem::path& output, const VideoQcRequest& request);

#ifdef RUDRA_HAVE_STILL_DECODE
Result<pyjson::Value> run_alpha_check(const std::filesystem::path& output, const std::filesystem::path& spool,
                                      int frame_count, int width, int height);
#endif

}  // namespace rudra
