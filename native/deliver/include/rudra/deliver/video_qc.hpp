#pragma once
// Export QC for a video master (Phase 4, step 6): rudra/video.py
// quality_check on ffprobe's JSON of the source and the master. Every check
// the Python makes, with its words: codec, pixel format, colour tags and size;
// the frame count and every presentation timestamp; HDR10's mastering display
// and content light SEI and their values; HLG without static metadata; the
// ProRes tag; the duration; the audio streams, their channels, rate, codec in
// copy mode and start and end offsets. The complete decode is run by the
// caller (qc_decode_command), before the verdict, as the Python runs it.

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/core/video_clock.hpp"
#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct ExpectedLight {
    double max_cll = 0, max_fall = 0, peak = 0, minimum = 0;
};

struct VideoQcRequest {
    std::string audio_mode = "copy";          // copy, aac, none
    std::optional<ExpectedLight> expected;    // expected_hdr
    std::string format = "hdr10";             // delivery_format
    bool alpha = false;
};

struct VideoQcFindings {
    std::vector<std::string> errors;          // in the Python's order
    int video_frames = 0, audio_streams = 0;
    pyjson::Value video;                      // the master's video stream, ffprobe's keys in its order
    pyjson::Value side_data;                  // the first frame's side data list
    std::string range_contract;
};

// The checks on ffprobe's JSON: -show_streams of the source; -show_streams and
// -show_frames of the master (video.py probe's two argument lists).
Result<VideoQcFindings> evaluate_video_qc(std::string_view source_streams_json, const VideoClock& clock,
                                          std::string_view output_streams_json, std::string_view output_frames_json,
                                          const VideoQcRequest& request);

// "Export QC failed: a; b" when there are findings, else the record the
// sidecar carries: passed, video_frames, audio_streams, video, hdr_side_data,
// decoded_without_errors, range_contract.
Result<pyjson::Value> qc_verdict(const VideoQcFindings& findings);

// The complete decode of the master that QC runs.
std::vector<std::string> qc_decode_command(const std::string& ffmpeg, const std::filesystem::path& output);

// check_alpha's record, or "Alpha QC failed: maximum 16-bit code error N exceeds 128".
inline constexpr int kAlphaToleranceCodes = 128;
Result<pyjson::Value> alpha_verdict(int max_error_16bit_codes);

}  // namespace rudra
