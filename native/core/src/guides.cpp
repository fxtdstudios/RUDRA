#include "rudra/core/guides.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {

// A dashed box: columns floor(x0) and ceil(x1) - 1, rows likewise; the dash
// counts from the box's first row or column.
bool on_box(double l, double t, double w, double h, double inset, int px, int py) {
    const double x0 = l + inset * w, x1 = l + (1.0 - inset) * w;
    const double y0 = t + inset * h, y1 = t + (1.0 - inset) * h;
    const int c0 = int(std::floor(x0)), c1 = int(std::ceil(x1)) - 1;
    const int r0 = int(std::floor(y0)), r1 = int(std::ceil(y1)) - 1;
    auto dash = [](int k) { return (k / kGuideDash) % 2 == 0; };
    if ((px == c0 || px == c1) && py >= r0 && py <= r1) return dash(py - r0);
    if ((py == r0 || py == r1) && px >= c0 && px <= c1) return dash(px - c0);
    return false;
}

}  // namespace

GuideSample guide_at(const GuideOptions& g, double l, double t, double w, double h, int px, int py) noexcept {
    GuideSample s;
    if (g.aspect > 0.0 && w > 0.0 && h > 0.0) {
        const double cx = px + 0.5, cy = py + 0.5;
        if (g.aspect > w / h) {   // letterbox
            const double mh = w / g.aspect, top = t + (h - mh) / 2;
            if (cy < top || cy >= top + mh) s.mask = kGuideMaskAlpha;
        } else {                  // pillarbox
            const double mw = h * g.aspect, left = l + (w - mw) / 2;
            if (cx < left || cx >= left + mw) s.mask = kGuideMaskAlpha;
        }
    }
    bool line = (g.action_safe && on_box(l, t, w, h, 0.05, px, py)) || (g.title_safe && on_box(l, t, w, h, 0.10, px, py));
    if (g.centre) {
        const int cc = int(std::floor(l + w / 2)), cr = int(std::floor(t + h / 2));
        const int arm = std::max(8, int(std::floor(0.02 * std::min(w, h))));
        if ((px == cc && std::abs(py - cr) <= arm) || (py == cr && std::abs(px - cc) <= arm)) line = true;
    }
    if (line) s.line = kGuideLineAlpha;
    return s;
}

}  // namespace rudra
