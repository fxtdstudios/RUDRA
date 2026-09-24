// Phase 3 step 3: the session is the page's state, undo and params().
//
// tests/golden/session/scripts.json is the Studio page itself, driven in
// headless Chromium by real DOM events (tools/emit_session_golden.py). Each
// gesture is replayed here through the same handler the page ran, and after
// every one params() must be the same bytes and the undo and redo depths and
// the wipe the same values.

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/engine/session.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/session/scripts.json");
        return json::parse(f);
    }();
    return g;
}

// parseFloat of the slider's value string.
double parse(const json& v) { return std::strtod(v.get<std::string>().c_str(), nullptr); }

void play(Session& s, const json& op) {
    const std::string k = op[0];
    if (k == "start") return;
    if (k == "mode") s.set_mode(op[1].get<std::string>());
    else if (k == "preserve") s.toggle_preserve();
    else if (k == "strength") {
        s.strength_press();
        s.strength_input(parse(op[1]));
    } else if (k == "peak") s.peak_input(parse(op[1]));
    else if (k == "region") {
        double x = 100.0;
        s.region_press(op[1].get<int>(), x);
        for (const auto& dx : op[2]) {
            x += dx.get<double>();
            s.region_move(x, op[3].get<bool>());
        }
        s.region_release();
    } else if (k == "region0") s.region_zero(op[1].get<int>());
    else if (k == "act") EXPECT_TRUE(s.run(op[1].get<std::string>())) << op[1];
    else if (k == "key") {
        const std::string key = op[1];
        s.key_down(key, op[2].get<bool>());
        s.key_up(key);
    } else FAIL() << "unknown gesture " << k;
}

}  // namespace

TEST(Session, EveryScriptStepMatchesThePage) {
    ASSERT_GE(golden()["scripts"].size(), 6u);
    for (const auto& [name, steps] : golden()["scripts"].items()) {
        Session s;
        int i = 0;
        for (const auto& st : steps) {
            play(s, st["op"]);
            const std::string where = name + " step " + std::to_string(i++) + " " + st["op"].dump();
            ASSERT_EQ(s.params_json(), st["params"].get<std::string>()) << where;
            EXPECT_EQ(s.undo_depth(), st["undo"].get<std::size_t>()) << where;
            EXPECT_EQ(s.redo_depth(), st["redo"].get<std::size_t>()) << where;
            if (st["wipe"].is_null()) {
                EXPECT_FALSE(s.wipe.has_value()) << where;
            } else {
                ASSERT_TRUE(s.wipe.has_value()) << where;
                EXPECT_EQ(*s.wipe, st["wipe"].get<double>()) << where;   // bit for bit
            }
        }
    }
}

TEST(Session, JsNumberPrintsAsJavaScriptDoes) {
    EXPECT_EQ(js_number(1.0), "1");
    EXPECT_EQ(js_number(-0.0), "0");
    EXPECT_EQ(js_number(0.05), "0.05");
    EXPECT_EQ(js_number(1.1), "1.1");
    EXPECT_EQ(js_number(0.1 + 0.2), "0.30000000000000004");
    EXPECT_EQ(js_number(100000.0), "100000");
    EXPECT_EQ(js_number(1e21), "1e+21");
    EXPECT_EQ(js_number(123456789012345680000.0), "123456789012345680000");
    EXPECT_EQ(js_number(1e-6), "0.000001");
    EXPECT_EQ(js_number(1e-7), "1e-7");
    EXPECT_EQ(js_number(1.5e-7), "1.5e-7");
    EXPECT_EQ(js_number(-4.0), "-4");
    EXPECT_EQ(js_number(1148.3414126469534), "1148.3414126469534");
}

TEST(Session, ChangesAreAnnounced) {
    Session s;
    std::uint32_t seen = 0;
    int calls = 0;
    s.on_change([&](std::uint32_t w) {
        seen |= w;
        ++calls;
    });
    s.set_mode("all");   // unchanged: nothing
    EXPECT_EQ(calls, 0);
    s.set_mode("off");
    EXPECT_TRUE(seen & Session::Grade);
    s.peak_input(1.0);
    EXPECT_TRUE(seen & Session::Peak);
    s.key_down("w", false);
    EXPECT_TRUE(seen & Session::Wipe);
    s.key_down("b", false);
    EXPECT_TRUE(s.flip_held);
    s.key_up("b");
    EXPECT_FALSE(s.flip_held);
    // A modified key runs nothing; the app's own actions come back to it.
    EXPECT_EQ(s.key_down("z", false, true), "");
    EXPECT_EQ(s.key_down("o", false), "open");
    EXPECT_EQ(s.key_down(",", false), "prev");
    EXPECT_EQ(s.key_down(" ", false), "play");
    const auto c = s.composite_params();
    EXPECT_EQ(c.mode, RecoveryMode::Off);
    EXPECT_EQ(c.regions.size(), 3u);
}
