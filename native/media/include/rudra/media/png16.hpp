#pragma once
// 16-bit PNG frames through OpenCV imgcodecs, as the Python's spool writes and
// reads them (cv2.imwrite with no parameters; IMREAD_UNCHANGED).

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "rudra/platform/result.hpp"

namespace rudra {

struct Png16 {
    int width = 0, height = 0, channels = 0;
    std::vector<std::uint16_t> data;   // height x width x channels, as stored (BGR(A))
};

// "Failed writing signal spool" when the writer refuses, as convert_video says it.
Result<void> write_png16(const std::filesystem::path& path, std::span<const std::uint16_t> hwc, int width,
                         int height, int channels);
Result<Png16> read_png16(const std::filesystem::path& path);

// cv2.__version__ of the OpenCV this build writes with ("4.6.0").
std::string opencv_version();

}  // namespace rudra
