#pragma once
// Video decode on a pipe (Phase 4, step 2): convert_video's decoder. ffmpeg
// turns the source into full-range 16-bit RGB (RGBA with alpha) through
// decoder_filter and writes raw frames to stdout; frames are read one at a
// time, and a short, missing or extra frame is refused with the Python's
// message.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

class Process;

// One decoded frame as ffmpeg wrote it: height x width x channels, interleaved,
// little-endian 16-bit, channels 3 (rgb48le) or 4 (rgba64le).
struct RawFrame16 {
    int width = 0, height = 0, channels = 3;
    std::vector<std::uint16_t> data;
    std::size_t bytes() const noexcept { return data.size() * 2; }
};

// decoded[..., :3].astype(float32) / 65535, as the planar image the predictor takes.
SdrImage rgb_from_frame(const RawFrame16& frame);

// The decoder's command, argument for argument (video.py convert_video).
std::vector<std::string> decoder_command(const std::string& ffmpeg, const std::filesystem::path& source,
                                         const InputContract& contract, bool alpha);

class VideoDecoder {
public:
    ~VideoDecoder();
    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    // Starts ffmpeg on the probed source; stderr goes to `log` (decode.log).
    static Result<std::unique_ptr<VideoDecoder>> open(const std::filesystem::path& source, const VideoSource& video,
                                                      const std::filesystem::path& log);
    // Any program writing raw frames of this shape to stdout (tests use it).
    static Result<std::unique_ptr<VideoDecoder>> open_command(const std::vector<std::string>& argv, int width,
                                                              int height, int channels, int frames,
                                                              const std::filesystem::path& log);

    int frames() const noexcept { return frames_; }
    int next_index() const noexcept { return index_; }
    // The next frame: "Decoder ended before expected frame count" when the
    // stream ends early, "Truncated decoded frame" on a partial one.
    Result<RawFrame16> next();
    // After the last frame: "Decoder produced unexpected extra frames", or
    // "Decoder failed; <the last 2000 bytes of the log>" on a non-zero exit.
    Result<void> finish();
    // Ends the program now (convert_video's finally).
    void close();

private:
    VideoDecoder() = default;
    std::unique_ptr<Process> process_;
    std::filesystem::path log_;
    int width_ = 0, height_ = 0, channels_ = 3, frames_ = 0, index_ = 0;
};

}  // namespace rudra
