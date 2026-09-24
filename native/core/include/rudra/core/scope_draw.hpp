#pragma once
// The scopes as the page draws them (Phase 3 step 5): drawScopes in ui/app.js
// writes the waveform and the histogram as SVG, and this builds the same
// elements with the same attributes, as the same strings, from the same scope
// data (tests/golden/scopes, tools/emit_scopes_golden.py). The app's scope
// widgets paint the result; nothing is drawn here.
//
// The colours are the page's: its zone hues (Z_SHADOW, Z_MID, Z_HI,
// Z_HI_BRIGHT) and the greys of its gridlines and labels, kept as it writes
// them ("#777" stays "#777"), and parse_colour reads them.

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rudra/core/scopes.hpp"

namespace rudra {

// One SVG element: its tag, its attributes in the page's order, its text
// (for <text>), its children (for <g>).
struct SvgElement {
    std::string tag;
    std::vector<std::pair<std::string, std::string>> attrs;
    std::string text;
    std::vector<SvgElement> children;
    const std::string* attr(const std::string& name) const;
};

struct SvgDrawing {
    std::string view_box;   // "min-x min-y width height"
    std::vector<SvgElement> elements;
};

// The page's constants: the waveform's W x H and the histogram's HW x HH.
inline constexpr int kWaveW = 460, kWaveH = 132, kHistW = 304, kHistH = 96;
inline constexpr double kScopeLo = 0.05, kScopeHi = 4000.0;

// drawScopes' waveform (#wave): gridlines, diffuse white, the envelope and
// interquartile bands, the median, the ceiling band, and MaxCLL when the
// frame has one (NaN or none: the diffuse-white label instead).
SvgDrawing waveform_svg(const ScopeData& s, std::optional<double> maxcll);
// drawScopes' histogram (#hist): the zone-coloured bars in a group, the base
// line, the diffuse-white mark and the nits scale. Its box is 310 x 116.
SvgDrawing histogram_svg(const ScopeData& s);

struct Rgb8 {
    std::uint8_t r = 0, g = 0, b = 0;
};
// "#rgb" or "#rrggbb"; nullopt for "none" or anything else.
std::optional<Rgb8> parse_colour(const std::string& s);

// The vectorscope's frame (the page's markup and theme.css): the picture is
// shown 176 px across on #0a0a0a, in a ring of #2e2e2e with an inner ring of
// #262626 inset 27 %, and the six hue labels at the page's places in #5c5c5c.
struct VectorLabel {
    const char* text;
    double left, top;   // fractions of the plot box (.plot.vs), the label's centre
};
struct VectorFrame {
    int display = 176;
    Rgb8 background{10, 10, 10}, ring{46, 46, 46}, inner_ring{38, 38, 38}, label{92, 92, 92};
    double inner_inset = 0.27;
    double label_px = 8.5;
    std::vector<VectorLabel> labels;
};
const VectorFrame& vector_frame();

}  // namespace rudra
