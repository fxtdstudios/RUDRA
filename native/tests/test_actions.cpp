// Phase 3 step 2: the app's actions are the Studio page's.
//
// tests/golden/actions/actions.json is read from ui/index.html and ui/app.js
// (tools/emit_actions_golden.py). engine/actions.cpp must give the same
// menus (entries, order, separators, labels, key hints, check states), the
// same 33 actions, the same keys, wipe nudge and shortcut sheet, and the same
// enable rules; the native additions must not take a key the page uses.

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/engine/actions.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/actions/actions.json");
        return json::parse(f);
    }();
    return g;
}

std::string s(std::string_view v) { return std::string(v); }

// A native menu as the page would have it: native entries and submenus out,
// then no leading, trailing or doubled separators.
std::vector<std::string> studio_entries(const MenuSpec& m) {
    std::vector<std::string> out;
    int depth = 0;
    for (auto e : m.entries) {
        if (e.substr(0, 1) == ">") { ++depth; continue; }
        if (e == "<") { --depth; continue; }
        if (depth > 0) continue;
        if (e == "-") {
            out.push_back("-");
            continue;
        }
        const ActionSpec* a = find_action(e);
        if (a && a->origin == ActionOrigin::Studio) out.push_back(s(e));
    }
    std::vector<std::string> clean;
    for (const auto& e : out) {
        if (e == "-" && (clean.empty() || clean.back() == "-")) continue;
        clean.push_back(e);
    }
    while (!clean.empty() && clean.back() == "-") clean.pop_back();
    return clean;
}

EnableRule rule_of(const std::string& expr) {
    static const std::map<std::string, EnableRule> rules = {
        {"any", EnableRule::AnyFrames},
        {"many", EnableRule::ManyFrames},
        {"any && state.live && !state.busy", EnableRule::CanMaster},
        {"state.undo.length > 0", EnableRule::CanUndo},
        {"state.redo.length > 0", EnableRule::CanRedo},
        {"!!state.metrics", EnableRule::HasMetrics},
        {"!!state.scopeData", EnableRule::HasScopes},
    };
    const auto it = rules.find(expr);
    if (it == rules.end()) ADD_FAILURE() << "the page has a new enable rule: " << expr;
    return it == rules.end() ? EnableRule::Always : it->second;
}

}  // namespace

TEST(Actions, MenusAreThePages) {
    for (const auto& gm : golden()["menus"]) {
        const std::string title = gm["title"];
        const MenuSpec* ours = nullptr;
        for (const auto& m : menu_specs())
            if (m.title == title) ours = &m;
        ASSERT_NE(ours, nullptr) << "no " << title << " menu";
        std::vector<std::string> want;
        for (const auto& item : gm["items"]) want.push_back(item.contains("separator") ? "-" : std::string(item["act"]));
        EXPECT_EQ(studio_entries(*ours), want) << title;
        for (const auto& item : gm["items"]) {
            if (item.contains("separator")) continue;
            const ActionSpec* a = find_action(std::string(item["act"]));
            ASSERT_NE(a, nullptr) << item["act"];
            EXPECT_EQ(s(a->label), std::string(item["label"])) << a->id;
            EXPECT_EQ(s(a->hint), item["hint"].is_null() ? "" : std::string(item["hint"])) << a->id;
            EXPECT_EQ(s(a->check), item["check"].is_null() ? "" : std::string(item["check"])) << a->id;
        }
    }
    // The page's menus keep their order; native menus go between them.
    std::vector<std::string> page, ours;
    for (const auto& gm : golden()["menus"]) page.push_back(gm["title"]);
    for (const auto& m : menu_specs()) {
        const bool studio = std::find(page.begin(), page.end(), s(m.title)) != page.end();
        if (studio) {
            ours.push_back(s(m.title));
            continue;
        }
        for (auto e : m.entries) {
            const ActionSpec* a = find_action(e);
            if (!a || a->origin != ActionOrigin::Studio) continue;
            EXPECT_EQ(a->id, "wipe") << "a Studio action in the native menu " << m.title << ": " << a->id;
        }
    }
    EXPECT_EQ(ours, page);
}

TEST(Actions, TheStudiosThirtyThree) {
    std::set<std::string> want, have;
    for (const auto& a : golden()["actions"]) want.insert(a);
    for (const auto& a : action_specs())
        if (a.origin == ActionOrigin::Studio) have.insert(s(a.id));
    EXPECT_EQ(have, want);
    EXPECT_EQ(want.size(), 33u);
    // Every entry of every menu is an action, and every action is once.
    std::set<std::string> ids;
    for (const auto& a : action_specs()) EXPECT_TRUE(ids.insert(s(a.id)).second) << "twice: " << a.id;
    for (const auto& m : menu_specs()) {
        for (auto e : m.entries) {
            if (e == "-" || e == "<" || e.substr(0, 1) == ">") continue;
            EXPECT_NE(find_action(e), nullptr) << e;
        }
    }
}

TEST(Actions, KeysAreThePages) {
    std::vector<std::string> want, have;
    for (const auto& [k, v] : golden()["keys"].items()) {
        want.push_back(k);
        EXPECT_EQ(s(action_for_key(k)), std::string(v)) << "key '" << k << "'";
    }
    for (auto k : mapped_keys()) have.push_back(s(k));
    EXPECT_EQ(have, want);
    EXPECT_EQ(action_for_key("q"), "");
    const auto& sp = golden()["special_keys"];
    EXPECT_TRUE(sp["modifiers_pass"].get<bool>());
    EXPECT_TRUE(sp["hold_b_flips"].get<bool>());
    EXPECT_TRUE(sp["escape_leaves_wipe"].get<bool>());
    EXPECT_DOUBLE_EQ(key_nudge_step(false), sp["wipe_nudge"]["step"].get<double>());
    EXPECT_DOUBLE_EQ(key_nudge_step(true), sp["wipe_nudge"]["shift_step"].get<double>());
    // A native shortcut never takes a plain key the page uses (B too: held).
    for (const auto& a : action_specs()) {
        if (a.origin != ActionOrigin::Native) {
            EXPECT_TRUE(a.native_key.empty()) << a.id;
            continue;
        }
        EXPECT_TRUE(a.hint.empty()) << a.id;
        const std::string k = s(a.native_key);
        if (k.empty() || k.find('+') != std::string::npos) continue;
        EXPECT_EQ(action_for_key(k), "") << a.id << " takes the page's key " << k;
        std::string lower = k;
        for (auto& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
        EXPECT_EQ(action_for_key(lower), "") << a.id << " takes the page's key " << lower;
        EXPECT_NE(lower, "b") << a.id;
    }
}

TEST(Actions, ShortcutSheet) {
    const auto& g = golden()["shortcuts"];
    ASSERT_EQ(shortcut_sheet().size(), g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
        EXPECT_EQ(s(shortcut_sheet()[i].keys), std::string(g[i][0])) << i;
        EXPECT_EQ(s(shortcut_sheet()[i].what), std::string(g[i][1])) << i;
    }
}

TEST(Actions, EnableRules) {
    const auto& g = golden()["enable_when"];
    for (const auto& a : action_specs()) {
        if (a.origin != ActionOrigin::Studio) continue;
        const auto it = g.find(s(a.id));
        const EnableRule want = it == g.end() ? EnableRule::Always : rule_of(*it);
        EXPECT_EQ(int(a.enable), int(want)) << a.id;
    }
}
