#pragma once
// EXR, ACES and OCIO writers. Port of rudra/delivery/exr.py and aces.py,
// byte for byte: uncompressed scanline OpenEXR 2.0 written directly (no
// OpenEXR library, like the Python), so the file the native app writes is
// the file the Python writes, header included.

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

using ExrAttributes = std::vector<std::pair<std::string, std::string>>;   // string attributes, in order
using Chromaticities = std::array<double, 8>;   // rx ry gx gy bx by wx wy

extern const Chromaticities kAp0Chromaticities;
extern const Chromaticities kAp1Chromaticities;
extern const Chromaticities kRec2020Chromaticities;

// float32 -> IEEE half, round to nearest even (numpy's astype(np.float16)).
std::uint16_t float_to_half(float f) noexcept;

// write_exr as bytes: 3 or 4 planes, NaN -> 0, +inf -> 3.4e38, -inf -> 0,
// half clipped to +-65504.
Result<std::vector<std::uint8_t>> exr_bytes(const PlanarBuffer& rgb, bool half = true,
                                            const std::optional<Chromaticities>& chromaticities = std::nullopt,
                                            const ExrAttributes& attributes = {});
Result<void> write_exr(const std::filesystem::path& path, const PlanarBuffer& rgb, bool half = true,
                       const std::optional<Chromaticities>& chromaticities = std::nullopt,
                       const ExrAttributes& attributes = {});

// write_aces_exr / write_acescg_exr: scene linear in `source`, diffuse white
// 1.0; `provenance` goes in as one sorted-keys JSON string attribute.
Result<void> write_aces_exr(const std::filesystem::path& path, const PlanarBuffer& rgb_linear,
                            Primaries source = Primaries::Rec2020, double exposure_scale = 1.0,
                            const ExrAttributes& provenance = {});
Result<void> write_acescg_exr(const std::filesystem::path& path, const PlanarBuffer& rgb_linear,
                              Primaries source = Primaries::Rec2020, double exposure_scale = 1.0,
                              const ExrAttributes& provenance = {});

// read_exr for uncompressed scanline files (exr.py's reader): the planes in
// R, G, B(, A) order, and the header's attributes in file order as raw bytes.
struct ExrImage {
    PlanarBuffer pixels;
    std::vector<std::pair<std::string, std::string>> attribute_types;   // name -> type
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> attributes;   // name -> payload
    std::vector<int> pixel_types;   // per channel, file order: 1 half, 2 float
    std::vector<std::uint16_t> half_bits;   // raw halves, R, G, B(, A) planes, when every channel is half
};
Result<ExrImage> read_exr(const std::filesystem::path& path);

// generate_ocio_config's text.
std::string ocio_config_text();

// The name Python uses for a primaries set ("rec709", "rec2020", ...).
const char* primaries_name(Primaries p) noexcept;

}  // namespace rudra
