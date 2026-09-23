#pragma once
// The master pixel chain: what lands in the file after the composite. Port of
// the stage sequence in ui/server.py _render_master, which runs in float64:
//
//   network (1.0 = 10 000 nits) -> nits
//   -> Region EV, clipped to [0, max_hdr]      rudra/delivery/controls.py
//   -> anchor to the SDR (level)               rudra/anchor.py
//   -> carry the source chroma (hue)           rudra/chroma.py
//   -> scene linear (nits / 203, float32)
//   -> container primaries (AP0 for ACES)      rudra/delivery/colorspace.py
//
// Every stage is its own function so each has its own golden and a failure
// names the stage that disagreed.

#include <cstddef>
#include <span>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

// Absolute nits, linear, planar 3 x h x w, double: the working precision of
// the master chain. Kept out of Image<Space> on purpose: it exists only inside
// this chain and the measurements, never on a GPU or in a file.
class NitsFrame {
public:
    NitsFrame() = default;
    NitsFrame(int height, int width) : h_(height), w_(width), data_(3 * static_cast<std::size_t>(height) * width, 0.0) {}
    int height() const noexcept { return h_; }
    int width() const noexcept { return w_; }
    std::size_t plane_size() const noexcept { return static_cast<std::size_t>(h_) * w_; }
    double* plane(int c) noexcept { return data_.data() + c * plane_size(); }
    const double* plane(int c) const noexcept { return data_.data() + c * plane_size(); }
    std::span<double> span() noexcept { return data_; }
    std::span<const double> span() const noexcept { return data_; }

private:
    int h_ = 0, w_ = 0;
    std::vector<double> data_;
};

inline constexpr double kAnchorKnee = 0.9, kAnchorSoftness = 0.04;
inline constexpr double kChromaKnee = 0.99, kChromaSoftness = 0.01, kChromaMaskSigma = 2.0;

enum class MasterContainer : std::uint8_t { Aces2065, SceneLinear };

struct MasterParams {
    std::vector<RegionBand> regions;       // empty or all ev 0: no grade
    double region_softness_stops = 1.0;
    bool anchor = true;
    double anchor_knee = kAnchorKnee;
    bool carry_chroma = true;
    double chroma_knee = kChromaKnee;
    Primaries source_primaries = Primaries::Rec709;
    MasterContainer container = MasterContainer::Aces2065;
};

// network * 10 000, widened.
NitsFrame nits_from_network(const NetworkLinearImage& network);

// apply_region_ev, then np.clip(0, ceiling). A no-op without a grade.
void apply_region_ev(NitsFrame& nits, std::span<const RegionBand> bands, double softness_stops, double ceiling_nits);

// rudra.anchor.anchor_gain / anchor_to_sdr.
void anchor_to_sdr(NitsFrame& nits, const SdrImage& sdr, double knee = kAnchorKnee,
                   double softness = kAnchorSoftness);

// cv2.GaussianBlur(src, (0,0), sigma, BORDER_REPLICATE) for one float32 plane:
// OpenCV's kernel size (round(8 sigma + 1) | 1) and its float32 kernel.
std::vector<float> gaussian_blur_replicate(std::span<const float> plane, int height, int width, double sigma);

// rudra.chroma.carry_source_chroma.
void carry_source_chroma(NitsFrame& nits, const SdrImage& sdr, double knee = kChromaKnee,
                         double softness = kChromaSoftness, double mask_sigma = kChromaMaskSigma);

// (nits / 203) as float32.
PlanarBuffer scene_linear(const NitsFrame& nits);

struct MasterPixels {
    NitsFrame nits;          // after the chain, before scene-linear: what is measured
    PlanarBuffer pixels;     // what is written: scene linear in the container's primaries
    Primaries primaries = Primaries::Rec709;
};

MasterPixels render_master_pixels(const NetworkLinearImage& network, const SdrImage& sdr,
                                  const ModelConstants& model, const MasterParams& params);

}  // namespace rudra
