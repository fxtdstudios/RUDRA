#pragma once
// media: readers and writers (NATIVE_ARCHITECTURE.md 3.2, 5.4). Implementations
// arrive in Phase 4 (libav with hardware decode, EXR/PNG/TIFF sequences); the
// interfaces are fixed now so the engine can be written against them.

#include <cstdint>
#include <string>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct Rational {
    std::int64_t num = 24, den = 1;
};

struct ShotInfo {
    int width = 0, height = 0;
    std::int64_t frame_count = 0;
    Rational rate;
    ColorEncoding source;       // what the file's tags (or the user's overrides) say
    std::string path;
};

struct SourceFrame {
    SdrImage pixels;            // canonicalised sRGB code values, what the network reads
    std::int64_t index = 0;
    std::uint64_t content_hash = 0;   // xxh64 of the decoded pixels: the cache key root
};

class Reader {
public:
    virtual ~Reader() = default;
    virtual const ShotInfo& info() const = 0;
    virtual Result<SourceFrame> decode(std::int64_t index) = 0;
};

class Writer {
public:
    virtual ~Writer() = default;
    virtual Result<void> write(const PlanarBuffer& scene_linear_2020, std::int64_t index) = 0;
    virtual Result<void> finish() = 0;
};

}  // namespace rudra
