#pragma once
// Video mastering and the spool frame (Phase 4, step 4): rudra/video.py
// convert_video between the predictor and the encoder. Each HDR frame goes
// through encode_master for the delivery format, is checked finite, gives its
// MaxCLL and frame average, and becomes the 16-bit spool frame: code values
// rounded to 16 bits, BGR for the PNG writer, the decoded alpha appended.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct VideoFrameLight {
    double max_cll = 0;         // float(maxrgb.max()), nits
    double frame_average = 0;   // float(maxrgb.mean()), nits
};

struct SpoolFrame {
    int width = 0, height = 0, channels = 3;
    std::vector<std::uint16_t> packed;   // height x width x channels, BGR(A)
    VideoFrameLight light;
};

// `hdr` in the network convention (1.0 = 10 000 nits). `alpha` is the decoded
// frame (height x width x 4, RGBA) when the source carries alpha; its fourth
// channel is appended as it was decoded. "Invalid HDR pixels at frame N" and
// "Invalid PQ pixels at frame N" as the Python says them.
Result<SpoolFrame> master_video_frame(const NetworkLinearImage& hdr, int index, const std::string& format,
                                      double peak_nits, std::optional<double> knee_nits,
                                      std::span<const std::uint16_t> alpha = {});

// The encoder's ceilings: math.ceil of the largest MaxCLL and frame average.
struct LightCeilings {
    int max_cll = 0, max_fall = 0;
};
LightCeilings light_ceilings(std::span<const VideoFrameLight> frames);

// The spool's frame name: f'{index:08d}.png'.
std::string spool_frame_name(int index);

// shutil.disk_usage(path).free < frame_bytes * 2 + 64 MiB.
bool spool_space_low(const std::filesystem::path& dir, std::uint64_t frame_bytes);

}  // namespace rudra
