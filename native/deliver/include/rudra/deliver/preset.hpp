#pragma once
// deliver: presets as data (NATIVE_ARCHITECTURE.md 5.5). Ported from
// rudra/delivery/profiles.py in Phase 4; each preset is a JSON file reviewed
// against the Python profile, not new C++.

#include <string>
#include <vector>

namespace rudra {

enum class Container { Mp4, Mov, Mkv, Exr };

struct Preset {
    std::string id;              // hdr10, hlg, prores422, prores422hq, prores4444, exr
    Container container = Container::Mp4;
    std::string codec;           // libx265, prores_ks, ...
    std::string pixel_format;    // yuv420p10le, yuv422p10le, yuva444p10le
    std::string transfer_tag;    // smpte2084, arib-std-b67
    std::string primaries_tag = "bt2020";
    std::string matrix_tag = "bt2020nc";   // swscale takes "bt2020" (the 23 Sep 2026 ffmpeg 7 fix)
    bool alpha = false;
    std::vector<std::string> qc;           // checks run before the file is published
};

}  // namespace rudra
