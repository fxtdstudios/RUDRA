// The viewport maths against the browser Studio's own layout (goldens from
// tools/emit_viewport_golden.py: ui/style.css and ui/app.js fitScale,
// applyViewport, zoomAbout in headless Chromium).

#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/viewport.hpp"

using namespace rudra;
using nlohmann::json;

namespace {
// Chromium lays boxes out in 1/64 px units; transforms are float.
constexpr double kLayoutTol = 1.0 / 64.0 + 1e-3;
}

TEST(Viewport, EqualsTheBrowsersLayout) {
    std::ifstream in(std::filesystem::path(RUDRA_GOLDEN_DIR) / "viewport" / "index.json");
    const json idx = json::parse(in);
    for (const auto& c : idx.at("cases")) {
        const auto vs = c.at("viewer").get<std::vector<double>>(), fs = c.at("frame").get<std::vector<double>>();
        const ViewSize viewer{vs[0], vs[1]}, frame{fs[0], fs[1]};
        const std::string what = c.at("viewer").dump() + " " + c.at("frame").dump();
        ViewportState v;
        const auto& states = c.at("states");
        const auto& ops = c.at("ops");
        for (std::size_t k = 0; k < states.size(); ++k) {
            if (k > 0) {
                const auto& op = ops.at(k - 1);
                const std::string kind = op.at(0).get<std::string>();
                if (kind == "wheel")
                    zoom_about(v, viewer, frame, op.at(1).get<double>(), op.at(2).get<double>(),
                               op.at(3).get<int>() > 0 ? kWheelStep : 1.0 / kWheelStep);
                else if (kind == "pan") pan_by(v, viewer, frame, op.at(1).get<double>(), op.at(2).get<double>());
                else if (kind == "actual") zoom_actual(v);
                else zoom_fit(v);
            }
            const json& s = states.at(k);
            const std::string at = what + " step " + std::to_string(k);
            EXPECT_EQ(fit_scale(viewer, frame), s.at("fit").get<double>()) << at;
            if (s.at("scale").is_null()) {
                EXPECT_FALSE(v.scale.has_value()) << at;
            } else {
                ASSERT_TRUE(v.scale.has_value()) << at;
                EXPECT_EQ(*v.scale, s.at("scale").get<double>()) << at;
                // zoomAbout reads the cursor offset from getBoundingClientRect,
                // which carries the transform in float32.
                EXPECT_NEAR(v.pan_x, s.at("pan_x").get<double>(), 1e-3) << at;
                EXPECT_NEAR(v.pan_y, s.at("pan_y").get<double>(), 1e-3) << at;
            }
            EXPECT_EQ(std::to_string(zoom_percent(v, viewer, frame)) + "%", s.at("label").get<std::string>()) << at;
            const PlacedRect r = place(v, viewer, frame);
            const auto want = s.at("rect").get<std::vector<double>>();
            EXPECT_NEAR(r.left, want[0], kLayoutTol) << at;
            EXPECT_NEAR(r.top, want[1], kLayoutTol) << at;
            EXPECT_NEAR(r.width, want[2], kLayoutTol) << at;
            EXPECT_NEAR(r.height, want[3], kLayoutTol) << at;
        }
    }
}

TEST(Viewport, PixelAndWipeUnderThePointer) {
    const ViewSize viewer{960, 600}, frame{1920, 1080};
    ViewportState v;
    const PlacedRect r = place(v, viewer, frame);   // fit: 932 x 524.25 at (14, 37.875)
    EXPECT_FALSE(pixel_at(r, frame, 13.9, 300).has_value());
    const auto p = pixel_at(r, frame, r.left + r.width / 2, r.top + r.height / 2);
    ASSERT_TRUE(p.has_value());
    EXPECT_EQ(p->x, 960);
    EXPECT_EQ(p->y, 540);
    EXPECT_EQ(wipe_at(r, r.left), 0.0);
    EXPECT_EQ(wipe_at(r, r.left + r.width * 0.25), 0.25);
    EXPECT_EQ(wipe_at(r, 5000), 1.0);
}

#include "rudra/core/guides.hpp"

TEST(Guides, SafeAreasCrossAndMaskLandOnTheirPixels) {
    GuideOptions g;
    g.action_safe = g.title_safe = g.centre = true;
    g.aspect = 2.39;
    auto at = [&](int x, int y) { return guide_at(g, 0, 0, 100, 100, x, y); };
    // Action safe (5 %): columns 5 and 94, dashed 3 on, 3 off from row 5.
    EXPECT_GT(at(5, 5).line, 0.0f);
    EXPECT_GT(at(5, 7).line, 0.0f);
    EXPECT_EQ(at(5, 8).line, 0.0f);
    EXPECT_GT(at(5, 11).line, 0.0f);
    EXPECT_GT(at(94, 5).line, 0.0f);
    EXPECT_EQ(at(95, 5).line, 0.0f);
    // Title safe (10 %): columns 10 and 89.
    EXPECT_GT(at(10, 10).line, 0.0f);
    EXPECT_GT(at(89, 10).line, 0.0f);
    // Centre cross: column 50 and row 50, arms of 8 pixels.
    EXPECT_GT(at(50, 42).line, 0.0f);
    EXPECT_EQ(at(50, 41).line, 0.0f);
    EXPECT_GT(at(58, 50).line, 0.0f);
    EXPECT_EQ(at(59, 50).line, 0.0f);
    EXPECT_FLOAT_EQ(at(5, 5).line, kGuideLineAlpha);
    // 2.39 letterbox in a square: 41.84 high, from 29.08; row 28 is outside, 29 inside.
    EXPECT_FLOAT_EQ(at(40, 28).mask, kGuideMaskAlpha);
    EXPECT_EQ(at(40, 29).mask, 0.0f);
    EXPECT_EQ(at(40, 70).mask, 0.0f);
    EXPECT_FLOAT_EQ(at(40, 71).mask, kGuideMaskAlpha);
    // Blend: the mask darkens, then the line goes toward white.
    const GuideSample both{kGuideLineAlpha, kGuideMaskAlpha};
    EXPECT_FLOAT_EQ(apply_guides(0.5f, 1.0f, both), 0.5f * 0.4f * (1 - kGuideLineAlpha) + kGuideLineAlpha);
    // Off: nothing.
    const GuideSample none = guide_at(GuideOptions{}, 0, 0, 100, 100, 5, 5);
    EXPECT_EQ(none.line, 0.0f);
    EXPECT_EQ(none.mask, 0.0f);
}
