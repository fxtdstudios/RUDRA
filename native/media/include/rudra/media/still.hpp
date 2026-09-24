#pragma once
// Still SDR frames at full precision. Port of rudra/decode.py decode_sdr: the
// same decoder (OpenCV imgcodecs, IMREAD_UNCHANGED, so no silent 8-bit
// conversion and no EXIF rotation), the same normalisation by the container's
// maximum, the same refusal of scene-linear float, the same honest bit depth.

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct DecodedStill {
    SdrImage rgb;              // display-encoded code values in [0, 1]
    int bits = 8;              // effective bits: an 8-bit frame padded into 16 reports 8
    int distinct_codes = 0;    // distinct values in the green channel as stored
    std::string source;        // which decoder read it
    bool truncated_to_8bit() const noexcept { return bits <= 8; }
};

// PNG (8/16-bit, grey, alpha, palette), JPEG, TIFF (8/16-bit, float), BMP,
// WebP; whatever this build's imgcodecs reads. Alpha is dropped, grey is
// replicated. Float input above 1 is refused: it is scene-linear HDR.
Result<DecodedStill> decode_sdr(std::span<const std::uint8_t> bytes);
Result<DecodedStill> decode_sdr_file(const std::filesystem::path& path);

// The Studio's preview size (ui/server.py _fit): a frame larger than
// `max_side` on its long side is scaled down to it with OpenCV's INTER_AREA in
// float, so no depth is lost; smaller frames and max_side <= 0 are returned
// as they are. The Studio previews at 1600 and masters at full size.
inline constexpr int kPreviewMaxSide = 1600;
SdrImage fit_max_side(const SdrImage& rgb, int max_side);

}  // namespace rudra
