#include "rudra/core/copy_texts.hpp"

#include <algorithm>
#include <cmath>

namespace rudra {
namespace {

JsValue numbers(const std::vector<double>& v) {
    std::vector<JsValue> out;
    out.reserve(v.size());
    for (double d : v) out.emplace_back(d);
    return JsValue::array(std::move(out));
}

}  // namespace

JsValue metrics_value(const ViewerMetrics& m, double compose_ms) {
    JsValue o = JsValue::object();
    o.add("maxcll", m.maxcll)
        .add("maxfall", m.maxfall)
        .add("peak_nits", m.peak_nits)
        .add("baseline_peak_nits", m.baseline_peak_nits)
        .add("headroom_stops", m.headroom_stops)
        .add("headroom_highlight_stops", m.headroom_highlight_stops)
        .add("headroom_shadow_stops", m.headroom_shadow_stops)
        .add("departure_rms_stops", m.departure_rms_stops)
        .add("p99_nits", m.p99_nits)
        .add("median_nits", m.median_nits)
        .add("above_diffuse_white_pct", m.above_diffuse_white_pct)
        .add("above_1000_nits_pct", m.above_1000_nits_pct)
        .add("highlight_mask_pct", m.highlight_mask_pct)
        .add("shadow_mask_pct", m.shadow_mask_pct)
        .add("compose_ms", compose_ms);
    return o;
}

JsValue scopes_value(const ScopeData& s) {
    JsValue o = JsValue::object();
    o.add("lo", numbers(s.lo))
        .add("q1", numbers(s.q1))
        .add("mid", numbers(s.mid))
        .add("q3", numbers(s.q3))
        .add("hi", numbers(s.hi))
        .add("histogram", numbers(s.histogram));
    return o;
}

JsValue delivery_value(const DeliveryRecord& d) {
    // graded(): some region moved by more than 1e-9 EV.
    const bool graded = std::any_of(d.regions.begin(), d.regions.end(), [](const auto& r) { return std::abs(r.ev) > 1e-9; });
    JsValue regions;
    if (graded) {
        regions = JsValue::array();
        for (const auto& r : d.regions) {
            JsValue o = JsValue::object();
            o.add("label", r.label).add("low_nits", r.low_nits).add("high_nits", r.high_nits).add("ev", r.ev);
            regions.push(std::move(o));
        }
    }
    JsValue o = JsValue::object();
    o.add("checkpoint", d.checkpoint)
        .add("step", d.step)
        .add("frame", d.frame)
        .add("resolution", d.resolution)
        .add("container", d.aces ? "ACES 2065-1 (AP0)" : "linear Rec.2020")
        .add("transfer", "linear")
        .add("diffuse_white_nits", 203)
        .add("recovery_mode", d.mode)
        .add("residual_strength", d.strength)
        .add("preserve_outside", d.preserve)
        .add("region_ev", std::move(regions))
        .add("measurements", d.metrics ? metrics_value(d.metrics->first, d.metrics->second) : JsValue());
    return o;
}

}  // namespace rudra
