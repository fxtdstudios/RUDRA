#pragma once
// What the page's Measure > Copy and Deliver > Copy put on the clipboard
// (Phase 3 step 11): JSON.stringify(value, null, 2) of state.metrics,
// state.scopeData and deliveryRecord(), key for key (tests/golden/copy).

#include <optional>
#include <string>
#include <vector>

#include "rudra/core/js_json.hpp"
#include "rudra/core/scopes.hpp"

namespace rudra {

// computeStats' object, compose_ms last as the page adds it.
JsValue metrics_value(const ViewerMetrics& m, double compose_ms);
// buildScopes' object: lo, q1, mid, q3, hi, histogram.
JsValue scopes_value(const ScopeData& s);

struct DeliveryRegion {
    std::string label;
    double low_nits = 0, high_nits = 0, ev = 0;
};
struct DeliveryRecord {
    std::optional<std::string> checkpoint;      // state.header.checkpoint
    std::optional<double> step;
    std::optional<std::string> frame;           // current().name
    std::optional<std::string> resolution;
    bool aces = true;                           // state.container
    std::string mode = "all";
    double strength = 1.0;
    bool preserve = true;
    std::vector<DeliveryRegion> regions;        // region_ev only when one is graded
    std::optional<std::pair<ViewerMetrics, double>> metrics;   // with compose_ms
};
JsValue delivery_value(const DeliveryRecord& d);

}  // namespace rudra
