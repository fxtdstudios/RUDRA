// Phase 3 step 9: the render plan is the Studio's.
//
// tests/golden/render_plan/plans.json is ui/server.py master_targets on 21
// cases (tools/emit_render_plan_golden.py): the same paths, in the same order,
// or the same refusal, word for word, from deliver/master.cpp's port. The
// cases' files are made again under a fresh folder, as the emitter made them.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/deliver/master.hpp"

using nlohmann::json;
using namespace rudra;
namespace fs = std::filesystem;

namespace {

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (std::size_t i = 0; (i = s.find(from, i)) != std::string::npos; i += to.size()) s.replace(i, from.size(), to);
    return s;
}

}  // namespace

TEST(RenderPlan, MasterTargetsAreTheStudios) {
    std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/render_plan/plans.json");
    const json g = json::parse(f);
    ASSERT_EQ(g["cases"].size(), 21u);
    for (const auto& c : g["cases"]) {
        const std::string name = c["name"];
        const fs::path root = fs::weakly_canonical(fs::temp_directory_path()) / ("rudra-plan-" + name);
        fs::remove_all(root);
        fs::create_directories(root);
        for (const auto& file : c["files"]) {
            const fs::path p = root / file.get<std::string>();
            fs::create_directories(p.parent_path());
            std::ofstream(p).put('\n');
        }
        const std::string r = root.generic_string();
        const json& p = c["params"];
        RenderPlan plan;
        plan.render_dir = replace_all(p.value("render_dir", std::string()), "<root>", r);
        plan.render_name = p.value("render_name", std::string("master"));
        plan.render_count = p.value("render_count", 1);
        plan.frame_start = p.value("frame_start", 1);
        plan.sequence = p.value("render_mode", std::string("image")) == "sequence";
        const auto got = master_targets(plan);
        if (c.contains("error")) {
            ASSERT_FALSE(got) << name;
            EXPECT_EQ(replace_all(fs::path(got.error().message).generic_string(), r, "<root>"),
                      c["error"].get<std::string>())
                << name;
        } else {
            ASSERT_TRUE(got) << name << ": " << got.error().message;
            std::vector<std::string> paths;
            for (const auto& t : *got) paths.push_back(replace_all(t.generic_string(), r, "<root>"));
            EXPECT_EQ(paths, c["paths"].get<std::vector<std::string>>()) << name;
        }
        fs::remove_all(root);
    }
}
