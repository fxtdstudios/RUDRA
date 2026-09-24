// Phase 3 step 11: the clipboard texts of Measure > Copy and Deliver > Copy.
//
// tests/golden/copy/texts.json is the Studio page's own copy actions run
// headless on the state each case sets, the clipboard recorded
// (tools/emit_copy_golden.py). The same state here must give the same bytes.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/copy_texts.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/copy/texts.json");
        return json::parse(f);
    }();
    return g;
}

double num(const json& v) {
    if (v.is_string() && v.get<std::string>() == "NaN") return std::numeric_limits<double>::quiet_NaN();
    return v.get<double>();
}

std::pair<ViewerMetrics, double> metrics(const json& j) {
    ViewerMetrics m;
    m.maxcll = num(j["maxcll"]);
    m.maxfall = num(j["maxfall"]);
    m.peak_nits = num(j["peak_nits"]);
    m.baseline_peak_nits = num(j["baseline_peak_nits"]);
    m.headroom_stops = num(j["headroom_stops"]);
    m.headroom_highlight_stops = num(j["headroom_highlight_stops"]);
    m.headroom_shadow_stops = num(j["headroom_shadow_stops"]);
    m.departure_rms_stops = num(j["departure_rms_stops"]);
    m.p99_nits = num(j["p99_nits"]);
    m.median_nits = num(j["median_nits"]);
    m.above_diffuse_white_pct = num(j["above_diffuse_white_pct"]);
    m.above_1000_nits_pct = num(j["above_1000_nits_pct"]);
    m.highlight_mask_pct = num(j["highlight_mask_pct"]);
    m.shadow_mask_pct = num(j["shadow_mask_pct"]);
    return {m, num(j["compose_ms"])};
}

std::vector<double> nums(const json& a) {
    std::vector<double> v;
    for (const auto& x : a) v.push_back(num(x));
    return v;
}

}  // namespace

TEST(CopyTexts, EveryCopyIsThePagesBytes) {
    ASSERT_GE(golden()["cases"].size(), 5u);
    for (const auto& c : golden()["cases"]) {
        const std::string act = c["act"], name = c["name"];
        const json& st = c["state"];
        std::string text;
        if (act == "copy-metrics") {
            const auto [m, ms] = metrics(st["metrics"]);
            text = metrics_value(m, ms).stringify();
            EXPECT_EQ(c["log"], "copied measurements to the clipboard");
        } else if (act == "copy-scopes") {
            ScopeData s;
            s.lo = nums(st["scopeData"]["lo"]);
            s.q1 = nums(st["scopeData"]["q1"]);
            s.mid = nums(st["scopeData"]["mid"]);
            s.q3 = nums(st["scopeData"]["q3"]);
            s.hi = nums(st["scopeData"]["hi"]);
            s.histogram = nums(st["scopeData"]["histogram"]);
            text = scopes_value(s).stringify();
            EXPECT_EQ(c["log"], "copied scope data to the clipboard");
        } else {
            DeliveryRecord d;
            // The page's defaults: no header, no frame, ACES, all, 1, preserve, the three regions at 0.
            d.regions = {{"highlights", 400, 2000, 0}, {"speculars", 2000, 8000, 0}, {"shadows", 0.05, 12, 0}};
            if (st.contains("header")) {
                const auto& h = st["header"];
                if (h.contains("checkpoint")) d.checkpoint = h["checkpoint"].get<std::string>();
                if (h.contains("step")) d.step = h["step"].get<double>();
                if (h.contains("resolution")) d.resolution = h["resolution"].get<std::string>();
            }
            if (st.contains("frames")) d.frame = st["frames"][st["index"].get<std::size_t>()]["name"].get<std::string>();
            if (st.contains("container")) d.aces = st["container"] == "aces";
            if (st.contains("mode")) d.mode = st["mode"];
            if (st.contains("strength")) d.strength = st["strength"];
            if (st.contains("preserve")) d.preserve = st["preserve"];
            if (st.contains("regions")) {
                d.regions.clear();
                for (const auto& r : st["regions"]) d.regions.push_back({r["label"], r["low_nits"], r["high_nits"], r["ev"]});
            }
            if (st.contains("metrics")) d.metrics = metrics(st["metrics"]);
            text = delivery_value(d).stringify();
            EXPECT_EQ(c["log"], "copied delivery metadata to the clipboard");
        }
        EXPECT_EQ(text, c["text"].get<std::string>()) << name;
    }
}

TEST(CopyTexts, StringifyEdgeCases) {
    EXPECT_EQ(JsValue::array().stringify(), "[]");
    EXPECT_EQ(JsValue::object().stringify(), "{}");
    EXPECT_EQ(JsValue(std::numeric_limits<double>::infinity()).stringify(), "null");
    EXPECT_EQ(JsValue(-0.0).stringify(), "0");
    EXPECT_EQ(js_quote(std::string("a\x01" "b\x1f\x7f", 5)), "\"a\\u0001b\\u001f\x7f\"");
    JsValue nested = JsValue::object();
    nested.add("a", JsValue::array({JsValue(1), JsValue::object()})).add("b", JsValue());
    EXPECT_EQ(nested.stringify(), "{\n  \"a\": [\n    1,\n    {}\n  ],\n  \"b\": null\n}");
}
