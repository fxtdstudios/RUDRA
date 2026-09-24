#pragma once
// The HDR card (Gate B, and the first-run wizard, Phase 3 step 10): six
// patches of known luminance, 10, 100, 203, 600, 1 000 and 2 000 nits, across
// the top 60 % of a 1280 x 720 picture on an 18 % grey of 203 nits. Shown
// through the display pass at the display's own peak, each patch reads as its
// luminance up to that peak and as the peak above it (clipped, ADR-005), so
// the card says on the glass what the swapchain says in numbers.

#include <array>
#include <string>

#include "rudra/core/image.hpp"
#include "rudra/core/view.hpp"

namespace rudra {

inline constexpr std::array<float, 6> kCardNits{10, 100, 203, 600, 1000, 2000};
inline constexpr int kCardWidth = 1280, kCardHeight = 720;

// Network-linear (1.0 = 10 000 nits), the same picture as model and baseline.
NetworkLinearImage hdr_card(int width = kCardWidth, int height = kCardHeight);

// The patch a card column falls in.
constexpr int card_patch(int x, int width) noexcept { return x * int(kCardNits.size()) / width; }

// What the wizard says about the display it opened on.
struct DisplayReport {
    bool hdr = false;            // an HDR swapchain with a real peak above a stop over SDR white
    std::string headline;        // "HDR display, 418 nits"
    std::string detail;          // what the card will show and what to do
};
// `swapchain` is the viewer's (SDR, scRGB, HDR10, EDR); `peak_from` is where
// the peak came from (swapchain, DXGI, placeholder).
DisplayReport display_report(const DisplayTarget& target, const std::string& swapchain, const std::string& peak_from);

}  // namespace rudra
