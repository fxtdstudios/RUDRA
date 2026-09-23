#include "rudra/core/viewport.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {

// applyViewport writes the pan as translate(panX.toFixed(1) px, ...).
double to_fixed_1(double v) {
    const double r = std::round(std::abs(v) * 10.0) / 10.0;
    return v < 0 ? -r : r;
}

}  // namespace

double fit_scale(ViewSize viewer, ViewSize frame) noexcept {
    if (frame.width <= 0 || frame.height <= 0 || viewer.width <= 0) return 1.0;
    const double pad = 2 * kViewerPadding;
    return std::min({1.0, (viewer.width - pad) / frame.width, (viewer.height - pad) / frame.height});
}

double displayed_scale(const ViewportState& v, ViewSize viewer, ViewSize frame) noexcept {
    return v.scale ? *v.scale : fit_scale(viewer, frame);
}

PlacedRect place(const ViewportState& v, ViewSize viewer, ViewSize frame) noexcept {
    // The plate is centred in the viewer's content box (grid, place-items:
    // center), and scaled about its own centre, then translated by the pan.
    const double s = displayed_scale(v, viewer, frame);
    const double cx = viewer.width / 2 + (v.scale ? to_fixed_1(v.pan_x) : 0.0);
    const double cy = viewer.height / 2 + (v.scale ? to_fixed_1(v.pan_y) : 0.0);
    const double w = frame.width * s, h = frame.height * s;
    return {cx - w / 2, cy - h / 2, w, h, s};
}

void zoom_about(ViewportState& v, ViewSize viewer, ViewSize frame, double x, double y, double factor) noexcept {
    const double from = displayed_scale(v, viewer, frame);
    const double to = std::max(kMinZoom, std::min(kMaxZoom, from * factor));
    const PlacedRect r = place(v, viewer, frame);
    const double cx = x - (r.left + r.width / 2);
    const double cy = y - (r.top + r.height / 2);
    const double k = to / from;
    const double px = v.scale ? v.pan_x : 0.0, py = v.scale ? v.pan_y : 0.0;
    v.pan_x = (px - cx) * k + cx;
    v.pan_y = (py - cy) * k + cy;
    v.scale = to;
}

void pan_by(ViewportState& v, ViewSize viewer, ViewSize frame, double dx, double dy) noexcept {
    if (!v.scale) {
        v.scale = fit_scale(viewer, frame);
        v.pan_x = v.pan_y = 0.0;
    }
    v.pan_x += dx;
    v.pan_y += dy;
}

int zoom_percent(const ViewportState& v, ViewSize viewer, ViewSize frame) noexcept {
    return int(std::floor(displayed_scale(v, viewer, frame) * 100.0 + 0.5));   // Math.round
}

std::optional<FramePixel> pixel_at(const PlacedRect& r, ViewSize frame, double x, double y) noexcept {
    if (r.width <= 0 || r.height <= 0) return std::nullopt;
    const int px = int(std::floor((x - r.left) / r.width * frame.width));
    const int py = int(std::floor((y - r.top) / r.height * frame.height));
    if (px < 0 || py < 0 || px >= int(frame.width) || py >= int(frame.height)) return std::nullopt;
    return FramePixel{px, py};
}

double wipe_at(const PlacedRect& r, double x) noexcept {
    if (r.width <= 0) return 0.5;
    return std::clamp((x - r.left) / r.width, 0.0, 1.0);
}

}  // namespace rudra
