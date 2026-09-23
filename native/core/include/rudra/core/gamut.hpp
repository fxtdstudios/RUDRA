#pragma once
// RGB primary conversions, derived from primaries and white points at run time
// (normalised primary matrices, Bradford adaptation). Port of
// rudra/delivery/colorspace.py; no hard-coded 3x3 tables that can drift.

#include <array>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

using Mat3 = std::array<std::array<double, 3>, 3>;

// Linear RGB (primaries) -> CIE XYZ at the space's own white.
Mat3 normalized_primary_matrix(Primaries p) noexcept;

// Exact linear RGB src -> dst, Bradford-adapting between white points.
Mat3 rgb_to_rgb_matrix(Primaries src, Primaries dst) noexcept;

// Every pixel of a 3-channel planar buffer through the matrix, in double,
// written back as float32 (colorspace.convert).
PlanarBuffer convert_primaries(const PlanarBuffer& rgb, Primaries src, Primaries dst);

}  // namespace rudra
