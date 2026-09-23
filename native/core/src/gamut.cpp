#include "rudra/core/gamut.hpp"

#include <cassert>
#include <cmath>

namespace rudra {
namespace {

struct Chromaticities {
    double rx, ry, gx, gy, bx, by, wx, wy;
};

constexpr Chromaticities chroma(Primaries p) noexcept {
    switch (p) {
        case Primaries::Rec709:  return {0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290};
        case Primaries::Rec2020: return {0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290};
        case Primaries::P3D65:   return {0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290};
        case Primaries::Ap0:     return {0.7347, 0.2653, 0.0000, 1.0000, 0.0001, -0.0770, 0.32168, 0.33767};
        case Primaries::Ap1:     return {0.7130, 0.2930, 0.1650, 0.8300, 0.1280, 0.0440, 0.32168, 0.33767};
    }
    return {};
}

constexpr Mat3 kBradford{{{0.8951, 0.2664, -0.1614}, {-0.7502, 1.7135, 0.0367}, {0.0389, -0.0685, 1.0296}}};

std::array<double, 3> xy_to_xyz(double x, double y) noexcept { return {x / y, 1.0, (1.0 - x - y) / y}; }

Mat3 mul(const Mat3& a, const Mat3& b) noexcept {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r[i][j] += a[i][k] * b[k][j];
    return r;
}

std::array<double, 3> mul(const Mat3& a, const std::array<double, 3>& v) noexcept {
    std::array<double, 3> r{};
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) r[i] += a[i][k] * v[k];
    return r;
}

Mat3 inverse(const Mat3& m) noexcept {
    const double a = m[0][0], b = m[0][1], c = m[0][2];
    const double d = m[1][0], e = m[1][1], f = m[1][2];
    const double g = m[2][0], h = m[2][1], i = m[2][2];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    assert(det != 0.0);
    const double s = 1.0 / det;
    return {{{A * s, -(b * i - c * h) * s, (b * f - c * e) * s},
             {B * s, (a * i - c * g) * s, -(a * f - c * d) * s},
             {C * s, -(a * h - b * g) * s, (a * e - b * d) * s}}};
}

Mat3 identity() noexcept { return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; }

Mat3 bradford(double swx, double swy, double dwx, double dwy) noexcept {
    if (std::abs(swx - dwx) <= 1e-9 && std::abs(swy - dwy) <= 1e-9) return identity();
    const auto src = mul(kBradford, xy_to_xyz(swx, swy));
    const auto dst = mul(kBradford, xy_to_xyz(dwx, dwy));
    Mat3 gain{};
    for (int k = 0; k < 3; ++k) gain[k][k] = dst[k] / src[k];
    return mul(inverse(kBradford), mul(gain, kBradford));
}

}  // namespace

Mat3 normalized_primary_matrix(Primaries p) noexcept {
    const auto c = chroma(p);
    const auto r = xy_to_xyz(c.rx, c.ry), g = xy_to_xyz(c.gx, c.gy), b = xy_to_xyz(c.bx, c.by);
    const Mat3 prim{{{r[0], g[0], b[0]}, {r[1], g[1], b[1]}, {r[2], g[2], b[2]}}};
    const auto scale = mul(inverse(prim), xy_to_xyz(c.wx, c.wy));
    Mat3 out = prim;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i][j] *= scale[j];
    return out;
}

Mat3 rgb_to_rgb_matrix(Primaries src, Primaries dst) noexcept {
    if (src == dst) return identity();
    const auto s = chroma(src), d = chroma(dst);
    return mul(inverse(normalized_primary_matrix(dst)),
               mul(bradford(s.wx, s.wy, d.wx, d.wy), normalized_primary_matrix(src)));
}

PlanarBuffer convert_primaries(const PlanarBuffer& rgb, Primaries src, Primaries dst) {
    assert(rgb.channels() == 3);
    const Mat3 m = rgb_to_rgb_matrix(src, dst);
    PlanarBuffer out(3, rgb.height(), rgb.width());
    const std::size_t n = rgb.plane_size();
    for (std::size_t i = 0; i < n; ++i) {
        const double v[3] = {rgb.plane(0)[i], rgb.plane(1)[i], rgb.plane(2)[i]};
        for (int c = 0; c < 3; ++c)
            out.plane(c)[i] = static_cast<float>(m[c][0] * v[0] + m[c][1] * v[1] + m[c][2] * v[2]);
    }
    return out;
}

}  // namespace rudra
