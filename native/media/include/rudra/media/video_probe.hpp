#pragma once
// Video probe and the input contract (Phase 4, step 1): rudra/video.py probe,
// timing, input_contract, has_alpha and decoder_filter. ffprobe is run as a
// program, as the Python does; its JSON is read here, and every refusal
// carries the Python's own message.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

// The video.py arguments the contract reads (argparse defaults).
struct VideoArgs {
    std::string format = "hdr10";              // PROFILES key
    std::optional<std::string> alpha_mode;     // "straight" or unset
    std::string input_transfer = "auto";       // auto, srgb, rec709, gamma22, gamma24
    std::string input_primaries = "auto";      // auto, rec709, rec2020
    std::string input_matrix = "auto";         // auto, bt709, bt2020nc, gbr
    std::string input_range = "auto";          // auto, full, limited
};

// One video stream as ffprobe reports it; absent keys take video.py's .get defaults.
struct VideoStreamInfo {
    int index = 0;
    int width = 0, height = 0;
    std::string codec_name;
    std::string pix_fmt;                       // "" when absent
    std::optional<std::string> color_transfer, color_primaries, color_space, color_range;
    std::string field_order = "progressive";
    std::string sample_aspect_ratio = "1:1";
    std::string avg_frame_rate = "0/1";
    std::string time_base = "1/1000000";
    double rotation = 0;                       // tags.rotate, then any side data rotation
};

struct VideoFrameInfo {
    std::optional<double> pts;                 // best_effort_timestamp_time; nullopt when absent
    int width = 0, height = 0;
};

struct VideoProbe {
    std::vector<VideoStreamInfo> video_streams;
    std::vector<VideoFrameInfo> frames;        // of v:0, when probed with frames
    std::string streams_json;                  // ffprobe's own output, for later steps (audio, format)
};

struct InputContract {
    std::string transfer, primaries, matrix, range;
    bool operator==(const InputContract&) const = default;
};

// timing(): the clip's clock.
struct VideoClock {
    std::string fps;                           // str(Fraction): "24000/1001" or "25"
    int frames = 0;
    double start = 0, duration = 0, tolerance = 0;
};

// Everything convert_video learns before it predicts a frame.
struct VideoSource {
    VideoStreamInfo stream;
    InputContract contract;
    VideoClock clock;
    bool alpha = false;
    std::string streams_json;
};

// A reduced fraction as Python's Fraction reads "n/d", "n" or a decimal.
struct Fraction {
    std::int64_t num = 0, den = 1;
    std::string str() const;                   // str(Fraction)
    double value() const;                      // float(Fraction), correctly rounded
};
Result<Fraction> parse_fraction(std::string_view text);

// Reads ffprobe -of json output: -show_streams (and -show_frames for v:0).
Result<VideoProbe> parse_video_probe(std::string_view streams_json, std::string_view frames_json = {});
// Runs ffprobe on the file (twice, as video.py does: streams, then frames).
Result<VideoProbe> probe_video(const std::filesystem::path& file);

bool has_alpha(std::string_view pix_fmt);
Result<InputContract> input_contract(const VideoStreamInfo& stream, const VideoArgs& args);
Result<VideoClock> video_timing(const VideoStreamInfo& stream, const std::vector<VideoFrameInfo>& frames);
// The zscale chain that turns decoded frames into full-range 16-bit RGB(A).
std::string decoder_filter(const InputContract& contract, bool alpha);

// convert_video from the probe to the frame-size check, in its order:
// one video stream, the contract, the clock, alpha, constant frame size.
Result<VideoSource> open_video_source(const VideoProbe& probe, const VideoArgs& args);
Result<VideoSource> open_video_source(const std::filesystem::path& file, const VideoArgs& args);

// shutil.which or "<name> is required on PATH".
Result<std::filesystem::path> require_executable(const std::string& name);
// video.run: stdout, or "<program> failed: <last 4000 bytes of stderr>".
Result<std::string> run_tool(const std::vector<std::string>& argv);

}  // namespace rudra
