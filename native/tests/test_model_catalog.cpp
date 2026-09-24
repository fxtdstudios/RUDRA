// Phase 3 step 10: which model packages there are and which a bare start opens.
//
// tests/golden/catalog/cases.json is ui/server.py find_checkpoint and
// loadable_models on eleven layouts of .pt files (tools/emit_catalog_golden.py).
// Each layout is built here as model packages, one folder per checkpoint whose
// manifest names it as its source, and the catalog must pick and list the same.

#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/engine/model_catalog.hpp"

using nlohmann::json;
using namespace rudra;
namespace fs = std::filesystem;

namespace {

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/catalog/cases.json");
        return json::parse(f);
    }();
    return g;
}

// A manifest read_manifest accepts, naming `source` as the checkpoint.
void write_package(const fs::path& dir, const std::string& name, const std::string& source, const std::string& exported) {
    fs::create_directories(dir);
    const json tol{{"atol", 1e-5}, {"rtol", 1e-5}};
    const json m{{"contract", "1.0"},
                 {"name", name},
                 {"source", {{"file", source}, {"sha256", std::string(64, '0')}}},
                 {"exported", exported},
                 {"network",
                  {{"base_channels", 32}, {"corpus_ev", -1.0}, {"log_scale", 16.0}, {"max_hdr", 4.0},
                   {"heads", {{"residual_gate", false}, {"shadow_gate", true}, {"curve", false}}},
                   {"curve_params", 1}}},
                 {"tiling", {{"tile_size", 512}, {"overlap", 64}}},
                 {"files",
                  {{"torchscript", "model.ts"}, {"onnx_frame", "model.frame.onnx"}, {"onnx_tile", "model.tile.onnx"},
                   {"golden", "golden/golden.json"}, {"torchscript_sha256", std::string(64, '0')},
                   {"onnx_frame_sha256", std::string(64, '0')}, {"onnx_tile_sha256", std::string(64, '0')}}},
                 {"onnx_inputs", {{"frame", {"sdr"}}, {"tile", {"sdr"}}}},
                 {"tolerance", {{"torchscript", tol}, {"onnx", tol}}}};
    std::ofstream(dir / "manifest.json") << m.dump(1);
}

// Hours ago as an ISO 8601 stamp that sorts as time does.
std::string stamp(double age_hours) {
    const long minutes = 100000L - long(age_hours * 60.0);
    char buf[64];
    std::snprintf(buf, sizeof buf, "2026-09-%02ldT%02ld:%02ld:00Z", 1 + minutes / 1440, (minutes / 60) % 24, minutes % 60);
    return buf;
}

fs::path fresh(const std::string& name) {
    const fs::path p = fs::temp_directory_path() / ("rudra-catalog-" + name);
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

}  // namespace

TEST(ModelCatalog, PicksAndListsAsTheServerDoes) {
    ASSERT_GE(golden()["cases"].size(), 10u);
    for (const auto& c : golden()["cases"]) {
        const std::string name = c["name"];
        const fs::path base = fresh(name);
        std::vector<fs::path> roots;
        int ti = 0;
        for (const auto& runs : c["training"]) {
            const fs::path root = base / ("train" + std::to_string(ti++));
            fs::create_directories(root);
            for (const auto& r : runs)
                write_package(root / r["run"].get<std::string>(), r["run"], r["file"], stamp(r["age"].get<double>()));
            roots.push_back(root);
        }
        const fs::path repo = base / "repo";
        fs::create_directories(repo);
        if (!c["registry"].is_null()) std::ofstream(repo / "models.json") << c["registry"].dump();
        for (const auto& f : c["files"]) {
            const std::string file = f;
            write_package(repo / file.substr(0, file.size() - 3), file.substr(0, file.size() - 3), file, stamp(1));
        }
        roots.push_back(repo);

        const Catalog cat = scan_packages(roots);
        if (c["chosen"].is_null()) {
            EXPECT_FALSE(cat.chosen.has_value()) << name << " chose " << cat.entries[*cat.chosen].package;
        } else {
            ASSERT_TRUE(cat.chosen.has_value()) << name;
            const auto& e = cat.entries[*cat.chosen];
            EXPECT_EQ(e.root, c["chosen"]["root"].get<std::size_t>()) << name;
            EXPECT_EQ(e.manifest->source_file, c["chosen"]["file"].get<std::string>()) << name;
            if (!c["chosen"]["run"].is_null()) {
                EXPECT_EQ(e.package.filename().string(), c["chosen"]["run"].get<std::string>()) << name;
            }
        }
        // /api/checkpoints: the registered ones in the repo root, in order, with their words.
        std::vector<CatalogEntry> listed;
        for (const auto& e : cat.entries)
            if (e.registered) listed.push_back(e);
        ASSERT_EQ(listed.size(), c["listed"].size()) << name;
        for (std::size_t i = 0; i < listed.size(); ++i) {
            EXPECT_EQ(listed[i].manifest->source_file, c["listed"][i]["file"].get<std::string>()) << name;
            EXPECT_EQ(listed[i].title, c["listed"][i]["title"].get<std::string>()) << name;
            EXPECT_EQ(listed[i].note, c["listed"][i]["note"].get<std::string>()) << name;
            EXPECT_EQ(listed[i].label(), listed[i].title) << name;
        }
        // The temporal refiner is never offered, even as an unregistered package.
        for (const auto& e : cat.entries)
            EXPECT_NE(e.manifest ? e.manifest->source_file : "", "temporal_v1.pt") << name;
        fs::remove_all(base);
    }
}

TEST(ModelCatalog, PackagesSayWhatTheyAreWhereABarePtCannot) {
    // A folder of exported packages with no models.json: the server's rules
    // pick nothing, and the catalog opens the first readable one instead.
    const fs::path base = fresh("fallback");
    write_package(base / "b_model", "b_model", "b.pt", stamp(1));
    write_package(base / "a_model", "a_model", "a.pt", stamp(2));
    fs::create_directories(base / "_invalid_old");
    write_package(base / "_invalid_old", "old", "old.pt", stamp(0));
    write_package(base / ".hidden", "hidden", "h.pt", stamp(0));
    fs::create_directories(base / "broken");
    std::ofstream(base / "broken" / "manifest.json") << "{ not json";
    const Catalog cat = scan_packages({base});
    ASSERT_EQ(cat.entries.size(), 3u);   // a_model, b_model, broken
    EXPECT_FALSE(cat.chosen);
    ASSERT_TRUE(cat.pick());
    EXPECT_EQ(cat.entries[*cat.pick()].package.filename(), "a_model");
    EXPECT_EQ(cat.entries[2].package.filename(), "broken");
    EXPECT_FALSE(cat.entries[2].manifest);
    EXPECT_FALSE(cat.entries[2].problem.empty());
    EXPECT_EQ(cat.entries[0].label(), "a_model");
    EXPECT_EQ(cat.find(base / "b_model" / "."), std::optional<std::size_t>(1));
    // A root that is itself a package is that package.
    const Catalog one = scan_packages({base / "b_model"});
    ASSERT_EQ(one.entries.size(), 1u);
    EXPECT_EQ(one.entries[0].manifest->name, "b_model");
    fs::remove_all(base);
}

TEST(ModelCatalog, RootsInOrderWithoutRepeats) {
#ifdef _WIN32
    _putenv_s("RUDRA_PACKAGE_ROOTS", " C:/a ;C:/b;;C:/a");
    const auto r = package_roots({"C:/b", "C:/c"}, "C:/app/models", "C:/user/models");
    EXPECT_EQ(r, (std::vector<fs::path>{"C:/a", "C:/b", "C:/c", "C:/app/models", "C:/user/models"}));
    _putenv_s("RUDRA_PACKAGE_ROOTS", "");
#else
    setenv("RUDRA_PACKAGE_ROOTS", " /a :/b::/a", 1);
    const auto r = package_roots({"/b", "/c"}, "/app/models", "/user/models");
    EXPECT_EQ(r, (std::vector<fs::path>{"/a", "/b", "/c", "/app/models", "/user/models"}));
    unsetenv("RUDRA_PACKAGE_ROOTS");
#endif
}

TEST(ModelCatalog, BackendChoicesAreEveryCompiledRuntimeGpuFirst) {
    const auto choices = backend_choices();
    const auto runtimes = compiled_runtimes();
    std::size_t cpus = 0;
    bool seen_cpu = false;
    for (const auto& c : choices) {
        if (c.device == Device::Cpu) {
            ++cpus;
            seen_cpu = true;
        } else {
            EXPECT_FALSE(seen_cpu) << "a GPU after the CPU: " << c.label();
        }
        EXPECT_EQ(BackendChoice::from_key(c.key()), std::optional<BackendChoice>(c));
    }
    EXPECT_EQ(cpus, runtimes.size());
    EXPECT_EQ((BackendChoice{Runtime::LibTorch, Device::Cuda}).label(), "LibTorch on CUDA");
    EXPECT_EQ((BackendChoice{Runtime::OnnxRuntime, Device::Cpu}).key(), "onnxruntime/cpu");
    EXPECT_FALSE(BackendChoice::from_key("nonsense/cpu"));
    EXPECT_EQ(device_pill(BackendInfo{Runtime::LibTorch, Device::Cpu, "", "cpu"}), "CPU");
    EXPECT_EQ(device_pill(BackendInfo{Runtime::OnnxRuntime, Device::DirectML, "", ""}), "DirectML");
}
