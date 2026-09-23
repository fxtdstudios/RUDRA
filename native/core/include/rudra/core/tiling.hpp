#pragma once
// Tile geometry and feathering, ported from training/infer_sdr2hdr.py
// (_tile_starts, _tile_weight, predict_fields). The native tiler must stitch
// fields exactly as the Python does, or tiled frames disagree in the overlap
// bands (the few-percent effect documented in predict_fields).

#include <vector>

#include "rudra/core/image.hpp"

namespace rudra {

struct TileConfig {
    int tile_size = 512;   // <= 0 means untiled
    int overlap = 64;
};

struct Tile {
    int y, x, h, w;
};

// Start offsets along one axis. Equivalent to _tile_starts.
std::vector<int> tile_starts(int length, int tile_size, int overlap);

// Every tile of a frame, rows then columns (the Python's loop order, which is
// also the accumulation order the stitched result depends on).
std::vector<Tile> plan_tiles(int height, int width, const TileConfig& cfg);

// torch.linspace(start, end, steps) in float32, element by element the way
// ATen computes it (symmetric from both ends), so weights match bit for bit.
std::vector<float> linspace_f32(float start, float end, int steps);

// Feather weights for one tile, h x w, as _tile_weight builds them: a 1-D ramp
// on each edge that has a neighbour, outer product of the two axes.
PlanarBuffer tile_weight(const Tile& t, int overlap, int full_h, int full_w);

// Accumulates weighted tiles and normalises, as predict_fields does:
// sum(f * w) / clamp_min(sum(w), 1e-6).
class Stitcher {
public:
    Stitcher(int channels, int height, int width);
    void add(const Tile& t, const PlanarBuffer& values, const PlanarBuffer& weight);
    PlanarBuffer finish() &&;

private:
    PlanarBuffer acc_;
    PlanarBuffer wsum_;
};

}  // namespace rudra
