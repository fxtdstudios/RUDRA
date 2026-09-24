#include "rudra/engine/video_qc.hpp"

#include "rudra/media/video_probe.hpp"
#include "rudra/platform/process.hpp"
#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/video_alpha.hpp"
#endif

namespace rudra {

Result<pyjson::Value> run_quality_check(std::string_view source_streams_json, const VideoClock& clock,
                                        const std::filesystem::path& output, const VideoQcRequest& request) {
    auto probe = probe_video(output);
    if (!probe) return probe.error();
    auto findings = evaluate_video_qc(source_streams_json, clock, probe->streams_json, probe->frames_json, request);
    if (!findings) return findings.error();
    // Decode the complete result: valid tags alone do not establish decodability.
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    auto decoded = run_tool(qc_decode_command(ffmpeg->string(), output));
    if (!decoded) return decoded.error();
    return qc_verdict(*findings);
}

#ifdef RUDRA_HAVE_STILL_DECODE
Result<pyjson::Value> run_alpha_check(const std::filesystem::path& output, const std::filesystem::path& spool,
                                      int frame_count, int width, int height) {
    auto max = alpha_max_error(output, spool, frame_count, width, height);
    if (!max) return max.error();
    return alpha_verdict(*max);
}
#endif

}  // namespace rudra
