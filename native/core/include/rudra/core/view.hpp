#pragma once
// The viewer's display pass on the CPU (docs/view.spec.md section 2): the
// reference the display shader is held to, and the port of the DISPLAY shader
// in ui/compositor.js. It reads the two composite targets and never changes
// them; switching view or moving the wipe is a new call on the same inputs.

#include <array>
#include <cstdint>
#include <vector>

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"

namespace rudra {

enum class ViewMode : std::uint8_t { Image = 0, FalseColour = 1, Difference = 2 };

// Which path the picture takes to the glass. The pipe bar shows it verbatim.
enum class OutputPath {
    SdrPqSimulation,   // SDR swapchain: exposure to the view peak, then clip (the browser Studio's view)
    ScRgb,             // FP16 linear Rec.709, 1.0 = 80 nits (Windows D3D12, Linux Vulkan)
    Hdr10,             // PQ Rec.2020 10-bit (Windows option)
    Edr,               // linear, 1.0 = the SDR white (macOS Metal; P3 or Rec.709 primaries)
};

// The swapchain the display pass writes for (docs/view.spec.md section 9).
struct DisplayTarget {
    OutputPath path = OutputPath::SdrPqSimulation;
    Primaries primaries = Primaries::Rec709;   // of the swapchain
    double peak_nits = 203.0;                  // what the display can show right now (EDR headroom is live)
    double unit_nits = 203.0;                  // linear paths: nits written as 1.0

    static DisplayTarget sdr() { return {}; }
    static DisplayTarget scrgb(double peak) { return {OutputPath::ScRgb, Primaries::Rec709, peak, 80.0}; }
    static DisplayTarget hdr10(double peak) { return {OutputPath::Hdr10, Primaries::Rec2020, peak, 10000.0}; }
    // EDR: 1.0 is the SDR white, which RUDRA holds at diffuse white (203 nits).
    static DisplayTarget edr(double peak, Primaries p = Primaries::P3D65) { return {OutputPath::Edr, p, peak, 203.0}; }
};
enum class ViewSource : std::uint8_t { Model, Baseline };

struct ViewParams {
    ViewMode mode = ViewMode::Image;
    double display_nits = 203.0;     // the view peak: exposure to it, then clip
    ViewSource show = ViewSource::Model;
    double wipe = -1.0;              // < 0 off; else [0, 1] across the frame, baseline on the left
    double wipe_half_width = 0.0012;
    double diff_gain = 2000.0;       // nits at which the difference ramp saturates
    DisplayTarget target;            // SDR unless the swapchain is HDR
    Primaries source = Primaries::Rec709;   // of the composite: the network keeps the input's primaries
};

// 8-bit RGB, interleaved, image order (row 0 the top).
struct Rgb8Image {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgb;
};

inline constexpr std::array<float, 3> kRec2020Luma{0.2627f, 0.6780f, 0.0593f};

// The false-colour zone of a Rec.2020 luminance in nits, 0..9, and its colour.
int false_colour_zone(float nits) noexcept;
std::array<float, 3> false_colour(int zone) noexcept;

// sRGB OETF on [0, 1], fp32 as the shader computes it.
float linear_to_srgb(float x) noexcept;

// The display pass as the values the swapchain is written with (3 x H x W,
// image order): SDR codes in [0, 1] before the 8-bit conversion; scRGB and EDR
// linear in the target's unit; HDR10 PQ codes. `model` and `baseline` are
// network units.
PlanarBuffer render_view(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                         const ViewParams& params);

// The SDR picture quantised as the 8-bit framebuffer does: round(255 c).
Rgb8Image render_view_rgb8(const NetworkLinearImage& model, const NetworkLinearImage& baseline,
                           const ViewParams& params);

}  // namespace rudra
