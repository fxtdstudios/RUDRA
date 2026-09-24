// Phase 3 step 5: the scopes are drawn as the page draws them.
//
// tests/golden/scopes/drawings.json is drawScopes in the page itself, run
// headless on the scope data it records (tools/emit_scopes_golden.py). The
// SVG it wrote is a display list; core/scope_draw.cpp must build the same one
// from the same data: every element, attribute and text, as the same string.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/js_format.hpp"
#include "rudra/core/scope_draw.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/scopes/drawings.json");
        return json::parse(f);
    }();
    return g;
}

ScopeData scopes_of(const json& s) {
    ScopeData d;
    d.lo = s["lo"].get<std::vector<double>>();
    d.q1 = s["q1"].get<std::vector<double>>();
    d.mid = s["mid"].get<std::vector<double>>();
    d.q3 = s["q3"].get<std::vector<double>>();
    d.hi = s["hi"].get<std::vector<double>>();
    d.histogram = s["histogram"].get<std::vector<double>>();
    return d;
}

void same(const SvgElement& ours, const json& page, const std::string& where) {
    ASSERT_EQ(ours.tag, page["tag"].get<std::string>()) << where;
    std::vector<std::pair<std::string, std::string>> want;
    for (const auto& [k, v] : page["attrs"].items()) want.emplace_back(k, v.get<std::string>());
    // The browser lists attributes in the order they were written; so do we.
    ASSERT_EQ(ours.attrs.size(), want.size()) << where;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const std::string* v = ours.attr(want[i].first);
        ASSERT_NE(v, nullptr) << where << " has no " << want[i].first;
        EXPECT_EQ(*v, want[i].second) << where << " " << want[i].first;
    }
    if (page.contains("text")) {
        EXPECT_EQ(ours.text, page["text"].get<std::string>()) << where;
    }
    const std::size_t kids = page.contains("children") ? page["children"].size() : 0;
    ASSERT_EQ(ours.children.size(), kids) << where;
    for (std::size_t i = 0; i < kids; ++i) same(ours.children[i], page["children"][i], where + "/" + std::to_string(i));
}

void same(const SvgDrawing& ours, const json& page, const std::string& where) {
    EXPECT_EQ(ours.view_box, page["viewBox"].get<std::string>()) << where;
    ASSERT_EQ(ours.elements.size(), page["elements"].size()) << where;
    for (std::size_t i = 0; i < ours.elements.size(); ++i)
        same(ours.elements[i], page["elements"][i], where + " #" + std::to_string(i));
}

}  // namespace

TEST(ScopeDraw, WaveformAndHistogramAreThePagesSvg) {
    ASSERT_GE(golden()["cases"].size(), 5u);
    for (const auto& c : golden()["cases"]) {
        const std::string name = c["name"];
        const ScopeData s = scopes_of(c["scopes"]);
        std::optional<double> maxcll;
        if (!c["metrics"].is_null()) {
            maxcll = c["metrics"]["maxcll"].is_null() ? std::numeric_limits<double>::quiet_NaN()
                                                      : c["metrics"]["maxcll"].get<double>();
        }
        same(waveform_svg(s, maxcll), c["drawn"]["wave"], name + " wave");
        same(histogram_svg(s), c["drawn"]["hist"], name + " hist");
    }
}

TEST(ScopeDraw, VectorscopeFrameIsThePages) {
    const auto& v = golden()["vector"];
    const auto& f = vector_frame();
    EXPECT_EQ(v["display"][0].get<int>(), f.display);
    EXPECT_EQ(v["canvas"][0].get<int>(), kVectorSize);
    auto rgb = [](const Rgb8& c) {
        return "rgb(" + std::to_string(c.r) + ", " + std::to_string(c.g) + ", " + std::to_string(c.b) + ")";
    };
    EXPECT_EQ(v["canvas_background"].get<std::string>(), rgb(f.background));
    EXPECT_EQ(v["ring"]["border"].get<std::string>(), rgb(f.ring));
    EXPECT_EQ(v["inner_ring"]["border"].get<std::string>(), rgb(f.inner_ring));
    // inset: 27 % of the ring's inside, in pixels as the browser resolved it
    // (its layout unit is 1/64 px).
    const double inset = std::stod(v["inner_ring"]["inset"].get<std::string>());
    EXPECT_NEAR(inset, f.inner_inset * (f.display - 2), 1.0 / 64.0);
    ASSERT_EQ(v["labels"].size(), f.labels.size());
    for (std::size_t i = 0; i < f.labels.size(); ++i) {
        const auto& l = v["labels"][i];
        EXPECT_EQ(l["text"].get<std::string>(), f.labels[i].text);
        EXPECT_DOUBLE_EQ(std::stod(l["left"].get<std::string>()) / 100.0, f.labels[i].left);
        EXPECT_DOUBLE_EQ(std::stod(l["top"].get<std::string>()) / 100.0, f.labels[i].top);
        EXPECT_EQ(l["color"].get<std::string>(), rgb(f.label));
        EXPECT_DOUBLE_EQ(std::stod(l["size"].get<std::string>()), f.label_px);
    }
}

TEST(JsFormat, ToFixedIsJavaScripts) {
    EXPECT_EQ(js_to_fixed(0.25, 1), "0.3");     // a tie goes up (printf would say 0.2)
    EXPECT_EQ(js_to_fixed(0.35, 1), "0.3");     // 0.35 is 0.34999... as a double
    EXPECT_EQ(js_to_fixed(1.005, 2), "1.00");   // likewise 1.00499...
    EXPECT_EQ(js_to_fixed(2.5, 0), "3");
    EXPECT_EQ(js_to_fixed(-2.5, 0), "-3");
    EXPECT_EQ(js_to_fixed(-0.04, 1), "-0.0");
    EXPECT_EQ(js_to_fixed(-0.0, 1), "0.0");
    EXPECT_EQ(js_to_fixed(9.96, 1), "10.0");
    EXPECT_EQ(js_to_fixed(132.0, 1), "132.0");
    EXPECT_EQ(js_to_fixed(123.456, 0), "123");
    EXPECT_EQ(js_to_fixed(0.000001, 2), "0.00");
    EXPECT_EQ(js_to_fixed(1e21, 2), "1e+21");
    EXPECT_EQ(js_round(-2.5), -2.0);   // Math.round: halves toward +infinity
    EXPECT_EQ(js_round(2.5), 3.0);
}

TEST(ScopeDraw, ParseColour) {
    ASSERT_TRUE(parse_colour("#777"));
    EXPECT_EQ(parse_colour("#777")->r, 0x77);
    EXPECT_EQ(parse_colour("#4d8fd6")->b, 0xd6);
    EXPECT_FALSE(parse_colour("none"));
}
