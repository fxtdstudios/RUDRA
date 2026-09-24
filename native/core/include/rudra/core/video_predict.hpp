#pragma once
// The pure parts of the video predictor (Phase 4, step 3): rudra/video.py
// Predictor.predict before and after the network, and ShadowSmoother. All in
// float32 as the Python's tensors are; the smoother's weight is a Python float
// (double). The network itself runs through infer (engine/video_predictor).

#include <string_view>
#include <utility>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// rudra.sdr2hdr.canonicalize_sdr: code values in the clip's transfer and range
// to the model's canonical sRGB code ("srgb", "rec709", "gamma22", "gamma24";
// "full" or "limited").
Result<PlanarBuffer> canonicalize_sdr(const PlanarBuffer& rgb, std::string_view transfer, std::string_view range);

// linear_to_srgb(M @ srgb_to_linear(x)) with M = rgb_to_rgb_matrix(src,
// Rec.2020) in float32, as the predictor does when the primaries are not 2020.
PlanarBuffer srgb_code_to_rec2020(const PlanarBuffer& code, Primaries src);

// F.interpolate(x, size, mode="area"): adaptive average pooling.
PlanarBuffer area_resize(const PlanarBuffer& x, int height, int width);

// torch.linspace(start, end, steps) in float32, as the CPU kernel computes it.
std::vector<float> torch_linspace(float start, float end, int steps);

// Scalar gate smoothing that resets at hard cuts; never blends pixels.
class ShadowSmoother {
public:
    explicit ShadowSmoother(double retention = 0.0, double cut_threshold = 0.15)
        : retention_(retention), cut_threshold_(cut_threshold) {}
    // (the weight to use, whether this frame starts a new shot)
    std::pair<double, bool> update(const PlanarBuffer& thumbnail, double weight);
    double retention() const noexcept { return retention_; }
    double cut_threshold() const noexcept { return cut_threshold_; }

private:
    double retention_, cut_threshold_;
    PlanarBuffer previous_;
    bool has_weight_ = false;
    double weight_ = 0.0;
};

// The size of the thumbnail the cut detector compares.
inline constexpr int kCutThumbHeight = 36, kCutThumbWidth = 64;

// The video path's tile origins along one axis: every size - overlap, the last
// moved back to end at the edge.
std::vector<int> video_tile_starts(int length, int size, int overlap);

// The feather for a tile at (y, x) of (th, tw) in a frame of (h, w): th x tw,
// row-major, (wy[:, None] * wx[None, :]) with linear ramps from 0.001 to 1 on
// the sides that have a neighbour.
std::vector<float> video_tile_blend(int y, int x, int th, int tw, int h, int w, int overlap);

}  // namespace rudra
