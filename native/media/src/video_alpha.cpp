#include "rudra/media/video_alpha.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "rudra/media/png16.hpp"
#include "rudra/platform/process.hpp"

namespace rudra {
namespace fs = std::filesystem;

std::vector<std::string> alpha_extract_command(const std::string& ffmpeg, const fs::path& output) {
    return {ffmpeg, "-v", "error", "-nostdin", "-i", output.string(), "-map", "0:v:0", "-vf",
            "alphaextract,format=gray16le", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray16le",
            "pipe:1"};
}

Result<int> alpha_max_error(const fs::path& output, const fs::path& spool, int frame_count, int width, int height) {
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    auto p = Process::start(alpha_extract_command(ffmpeg->string(), output), spool / "alpha_qc.log");
    if (!p) return p.error();
    auto& process = **p;
    auto fail = [&](const char* message) -> Error {
        if (process.running()) process.kill();
        process.wait();
        return make_error(ErrorCode::IntegrityError, message);
    };
    int maximum = 0;
    std::vector<std::uint16_t> frame(static_cast<std::size_t>(width) * height);
    for (int index = 0; index < frame_count; ++index) {
        const auto bytes = std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(frame.data()), frame.size() * 2);
        const std::size_t got = process.read(bytes);
        if (got == 0) return fail("Missing decoded alpha frame");
        if (got != bytes.size()) return fail("Truncated decoded frame");
        char name[32];
        std::snprintf(name, sizeof name, "%08d.png", index);
        auto png = read_png16(spool / name);
        if (!png) return fail("Failed reading signal spool");
        if (png->channels != 4 || png->width != width || png->height != height) return fail("Spool frame has no alpha");
        for (std::size_t i = 0; i < frame.size(); ++i)
            maximum = std::max(maximum, std::abs(int(frame[i]) - int(png->data[i * 4 + 3])));
    }
    std::uint8_t one[1];
    if (process.read(one) != 0) return fail("Unexpected extra alpha frames");
    if (process.wait() != 0) return make_error(ErrorCode::IntegrityError, "Alpha decode failed");
    return maximum;
}

}  // namespace rudra
