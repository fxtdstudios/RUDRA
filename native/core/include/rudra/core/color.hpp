#pragma once
// What the numbers in a buffer mean (NATIVE_ARCHITECTURE.md 3.1, principle P1).

#include <cstdint>

namespace rudra {

// ---- units ------------------------------------------------------------------
// Strong types so a stop is never added to a nit by accident.
struct Nits  { float v; constexpr auto operator<=>(const Nits&) const = default; };
struct Stops { float v; constexpr auto operator<=>(const Stops&) const = default; };

// Diffuse white. Scene-linear 1.0 means this many nits everywhere in RUDRA.
inline constexpr Nits kDiffuseWhite{203.0f};
// The network's own PQ-normalised convention: 1.0 = 10 000 nits.
inline constexpr Nits kPqPeak{10000.0f};
// scRGB (the Windows HDR swapchain): linear Rec.709, 1.0 = 80 nits.
inline constexpr Nits kScRgbUnit{80.0f};

// ---- encodings ----------------------------------------------------------------
enum class Transfer : std::uint8_t { Srgb, Rec709, Gamma22, Gamma24, Linear, Pq, Hlg };
enum class Primaries : std::uint8_t { Rec709, Rec2020, P3D65, Ap0, Ap1 };
enum class Range : std::uint8_t { Full, Limited };

struct ColorEncoding {
    Transfer transfer = Transfer::Srgb;
    Primaries primaries = Primaries::Rec709;
    Range range = Range::Full;
    float nits_per_unit = 0.0f;   // 0 for display-referred code values
    constexpr bool operator==(const ColorEncoding&) const = default;
};

// ---- compile-time tags for the spaces the pipeline passes between stages ------
namespace space {
// sRGB-encoded display code values in [0,1], canonicalised the way
// rudra.sdr2hdr.canonicalize_sdr does. What the network reads.
struct SdrDisplay {
    static constexpr ColorEncoding encoding{Transfer::Srgb, Primaries::Rec709, Range::Full, 0.0f};
};
// Scene-linear, Rec.709 primaries, 1.0 = 203 nits (the composite before gamut).
struct SceneLinear709 {
    static constexpr ColorEncoding encoding{Transfer::Linear, Primaries::Rec709, Range::Full, 203.0f};
};
// Scene-linear, Rec.2020 primaries, 1.0 = 203 nits (what masters are written from).
struct SceneLinear2020 {
    static constexpr ColorEncoding encoding{Transfer::Linear, Primaries::Rec2020, Range::Full, 203.0f};
};
// The network's HDR convention: linear Rec.709, 1.0 = 10 000 nits. The analytic
// baseline and the network output live here (rudra/sdr2hdr.py).
struct NetworkLinear {
    static constexpr ColorEncoding encoding{Transfer::Linear, Primaries::Rec709, Range::Full, 10000.0f};
};
}  // namespace space

inline constexpr float kLumaRec709[3] = {0.2126f, 0.7152f, 0.0722f};

}  // namespace rudra
