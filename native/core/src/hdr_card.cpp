#include "rudra/core/hdr_card.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

#include "rudra/core/js_format.hpp"

namespace rudra {

NetworkLinearImage hdr_card(int width, int height) {
    PlanarBuffer px(3, height, width, 0.18f * 203.0f / 10000.0f);
    for (int y = 0; y < int(float(height) * 0.6f); ++y)
        for (int x = 0; x < width; ++x) {
            const float v = kCardNits[std::size_t(card_patch(x, width))] / 10000.0f;
            for (int c = 0; c < 3; ++c) px.at(c, y, x) = v;
        }
    return NetworkLinearImage(std::move(px));
}

namespace {

std::string nits(double v) { return js_fmt(std::round(v), 0) + " nits"; }

}  // namespace

DisplayReport display_report(const DisplayTarget& t, const std::string& swapchain, const std::string& peak_from) {
    DisplayReport r;
    if (t.path == OutputPath::SdrPqSimulation) {
        r.headline = "SDR display";
        r.detail = "The viewer shows HDR simulated on a 203-nit display: the 600, 1,000 and 2,000 nit patches "
                   "clip to white. Turn on HDR for this display in the system's settings and open the wizard again "
                   "to judge highlights on the glass.";
        return r;
    }
    if (peak_from == "placeholder") {
        r.headline = "HDR display (" + swapchain + "), peak unknown";
        r.detail = "The swapchain is HDR but the system reports no peak for this display, so the viewer assumes " +
                   nits(t.peak_nits) + ". Check where the 1,000 and 2,000 nit patches stop getting brighter.";
        return r;
    }
    r.hdr = t.peak_nits > 2.0 * 203.0;
    r.headline = (r.hdr ? "HDR display, " : "HDR display with little headroom, ") + nits(t.peak_nits);
    std::vector<std::string> over;
    for (float n : kCardNits)
        if (n > t.peak_nits) over.push_back(js_fmt(n, 0));
    std::string clip;
    for (std::size_t i = 0; i < over.size(); ++i)
        clip += (i == 0 ? "" : i + 1 == over.size() ? " and " : ", ") + over[i];
    r.detail = swapchain + ", peak from " + peak_from + ". Each patch shows its own luminance up to " +
               nits(t.peak_nits) + (clip.empty() ? "." : "; the " + clip + " nit patches clip there, never tone-mapped.");
    if (!r.hdr) r.detail += " Less than a stop above SDR white: highlights will look much as they do in SDR.";
    return r;
}

}  // namespace rudra
