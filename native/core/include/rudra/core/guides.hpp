#pragma once
// Guides over the picture (docs/view.spec.md section 11, Phase 2 step 11):
// the Studio's two safe areas (ui/style.css .guides: 90 % and 80 % boxes,
// dashed, white at 46/255), a centre cross and an aspect mask. They are drawn
// in screen space by the blit, one device pixel wide at every zoom, and blend
// over the picture as graphics at the SDR white. This is the reference the
// shader is held to (rudra-viewer-check).

namespace rudra {

struct GuideOptions {
    bool action_safe = false;   // 90 %: inset 5 % each side
    bool title_safe = false;    // 80 %: inset 10 % each side
    bool centre = false;        // a cross at the centre
    double aspect = 0.0;        // > 0: darken outside this width / height ratio
    bool any() const noexcept { return action_safe || title_safe || centre || aspect > 0.0; }
};

inline constexpr float kGuideLineAlpha = 46.0f / 255.0f;   // #ffffff2e
inline constexpr float kGuideMaskAlpha = 0.6f;             // outside the aspect
inline constexpr int kGuideDash = 3;                       // device pixels on, then off

struct GuideSample {
    float line = 0.0f;   // alpha of white
    float mask = 0.0f;   // alpha of black
};

// For the device pixel (px, py) (top-left origin) and the picture's placed
// rectangle in device pixels.
GuideSample guide_at(const GuideOptions& g, double left, double top, double width, double height, int px, int py) noexcept;

// One encoded channel under the guides: the mask darkens, then the line
// blends toward `white` (the SDR white in the swapchain's encoding).
inline float apply_guides(float c, float white, const GuideSample& s) noexcept {
    c = c * (1.0f - s.mask);
    return c * (1.0f - s.line) + white * s.line;
}

}  // namespace rudra
