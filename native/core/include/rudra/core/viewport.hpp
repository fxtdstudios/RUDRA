#pragma once
// Where the picture lands in the viewer (docs/view.spec.md section 10): fit,
// 1:1, zoom about the cursor, pan. A port of fitScale, applyViewport and
// zoomAbout in ui/app.js together with the CSS box model they drive
// (ui/style.css .viewer / .plate), held to the browser's own layout by
// tools/emit_viewport_golden.py. Units are the viewer's logical pixels, origin
// its top-left corner; the renderer multiplies by the device pixel ratio.

#include <optional>

namespace rudra {

inline constexpr double kViewerPadding = 14.0;   // .viewer padding, each side
inline constexpr double kMinZoom = 0.05, kMaxZoom = 32.0, kWheelStep = 1.12;

struct ViewSize {
    double width = 0.0, height = 0.0;
};

struct ViewportState {
    std::optional<double> scale;   // empty: fit
    double pan_x = 0.0, pan_y = 0.0;
};

struct PlacedRect {
    double left = 0.0, top = 0.0, width = 0.0, height = 0.0;
    double scale = 1.0;   // displayed pixels per frame pixel
};

// The scale fit mode shows: the whole frame inside the padded viewer, never above 1:1.
double fit_scale(ViewSize viewer, ViewSize frame) noexcept;
double displayed_scale(const ViewportState& v, ViewSize viewer, ViewSize frame) noexcept;
PlacedRect place(const ViewportState& v, ViewSize viewer, ViewSize frame) noexcept;

// The wheel: zoom by `factor` keeping the frame point under (x, y) still.
void zoom_about(ViewportState& v, ViewSize viewer, ViewSize frame, double x, double y, double factor) noexcept;
// A middle drag of (dx, dy); fit mode first becomes a zoom at the fit scale.
void pan_by(ViewportState& v, ViewSize viewer, ViewSize frame, double dx, double dy) noexcept;
inline void zoom_fit(ViewportState& v) noexcept { v = {}; }
inline void zoom_actual(ViewportState& v) noexcept { v = {1.0, 0.0, 0.0}; }

// The readout: round(100 s) percent.
int zoom_percent(const ViewportState& v, ViewSize viewer, ViewSize frame) noexcept;

// The frame pixel under (x, y) (probeAt), or nothing outside the frame.
struct FramePixel {
    int x = 0, y = 0;
};
std::optional<FramePixel> pixel_at(const PlacedRect& r, ViewSize frame, double x, double y) noexcept;
// The wipe position for a pointer at x (wipeFromEvent), in [0, 1].
double wipe_at(const PlacedRect& r, double x) noexcept;

}  // namespace rudra
