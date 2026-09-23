#include "rudra/core/tiling.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace rudra {

std::vector<int> tile_starts(int length, int tile_size, int overlap) {
    if (length <= tile_size) return {0};
    const int stride = tile_size - overlap;
    assert(stride > 0);
    std::vector<int> starts;
    for (int s = 0; s <= length - tile_size; s += stride) starts.push_back(s);
    if (starts.back() != length - tile_size) starts.push_back(length - tile_size);
    return starts;
}

std::vector<Tile> plan_tiles(int height, int width, const TileConfig& cfg) {
    if (cfg.tile_size <= 0 || (height <= cfg.tile_size && width <= cfg.tile_size))
        return {Tile{0, 0, height, width}};
    std::vector<Tile> tiles;
    for (int y : tile_starts(height, cfg.tile_size, cfg.overlap))
        for (int x : tile_starts(width, cfg.tile_size, cfg.overlap))
            tiles.push_back(Tile{y, x, std::min(cfg.tile_size, height - y), std::min(cfg.tile_size, width - x)});
    return tiles;
}

std::vector<float> linspace_f32(float start, float end, int steps) {
    std::vector<float> v(static_cast<std::size_t>(std::max(steps, 0)));
    if (steps <= 0) return v;
    if (steps == 1) { v[0] = start; return v; }
    // ATen's CPU kernel: a float32 step, applied with one fused multiply-add
    // from whichever end is nearer. A plain multiply then add differs from it
    // in the last bit on ~10% of elements (measured 23 Sep 2026, torch 2.14).
    const float step = (end - start) / static_cast<float>(steps - 1);
    const int halfway = steps / 2;
    for (int i = 0; i < steps; ++i)
        v[static_cast<std::size_t>(i)] = i < halfway ? std::fma(step, static_cast<float>(i), start)
                                                     : std::fma(-step, static_cast<float>(steps - i - 1), end);
    return v;
}

PlanarBuffer tile_weight(const Tile& t, int overlap, int full_h, int full_w) {
    std::vector<float> wy(static_cast<std::size_t>(t.h), 1.0f), wx(static_cast<std::size_t>(t.w), 1.0f);
    const int fy = std::min(overlap, t.h / 2), fx = std::min(overlap, t.w / 2);
    if (t.y > 0 && fy) { auto r = linspace_f32(1e-3f, 1.0f, fy); std::copy(r.begin(), r.end(), wy.begin()); }
    if (t.y + t.h < full_h && fy) { auto r = linspace_f32(1.0f, 1e-3f, fy); std::copy(r.begin(), r.end(), wy.end() - fy); }
    if (t.x > 0 && fx) { auto r = linspace_f32(1e-3f, 1.0f, fx); std::copy(r.begin(), r.end(), wx.begin()); }
    if (t.x + t.w < full_w && fx) { auto r = linspace_f32(1.0f, 1e-3f, fx); std::copy(r.begin(), r.end(), wx.end() - fx); }
    PlanarBuffer w(1, t.h, t.w);
    for (int r = 0; r < t.h; ++r)
        for (int c = 0; c < t.w; ++c) w.at(0, r, c) = wy[static_cast<std::size_t>(r)] * wx[static_cast<std::size_t>(c)];
    return w;
}

Stitcher::Stitcher(int channels, int height, int width) : acc_(channels, height, width), wsum_(1, height, width) {}

void Stitcher::add(const Tile& t, const PlanarBuffer& values, const PlanarBuffer& weight) {
    assert(values.height() == t.h && values.width() == t.w && values.channels() == acc_.channels());
    for (int c = 0; c < acc_.channels(); ++c)
        for (int r = 0; r < t.h; ++r)
            for (int x = 0; x < t.w; ++x)
                acc_.at(c, t.y + r, t.x + x) += values.at(c, r, x) * weight.at(0, r, x);
    for (int r = 0; r < t.h; ++r)
        for (int x = 0; x < t.w; ++x) wsum_.at(0, t.y + r, t.x + x) += weight.at(0, r, x);
}

PlanarBuffer Stitcher::finish() && {
    for (int c = 0; c < acc_.channels(); ++c)
        for (int r = 0; r < acc_.height(); ++r)
            for (int x = 0; x < acc_.width(); ++x)
                acc_.at(c, r, x) /= std::max(wsum_.at(0, r, x), 1e-6f);
    return std::move(acc_);
}

}  // namespace rudra
