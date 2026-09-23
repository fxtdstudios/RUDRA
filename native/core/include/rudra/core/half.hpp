#pragma once
// IEEE 754 binary16 to float, exact (every half is a float). The wire format
// of the Studio's /api/frame carries the fields as halves, and the viewer
// goldens are fed to the native code in that form.

#include <bit>
#include <cstdint>

namespace rudra {

constexpr float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) return std::bit_cast<float>(sign);
        int e = -1;   // subnormal: normalise
        do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
        return std::bit_cast<float>(sign | (std::uint32_t(127 - 15 - e) << 23) | ((mant & 0x3FFu) << 13));
    }
    if (exp == 31) return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13));
    return std::bit_cast<float>(sign | ((exp + 112) << 23) | (mant << 13));
}

}  // namespace rudra
