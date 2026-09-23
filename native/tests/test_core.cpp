// librudra against the Python it ports (goldens from tools/emit_core_golden.py).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/model_manifest.hpp"
#include "rudra/core/tiling.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kCore = fs::path(RUDRA_GOLDEN_DIR) / "core";

nlohmann::json index_json() {
    std::ifstream in(kCore / "index.json");
    return nlohmann::json::parse(in);
}

NpyArray npy(const std::string& file) {
    auto a = read_npy(kCore / file);
    EXPECT_TRUE(a) << file;
    return a ? std::move(*a) : NpyArray{};
}

SdrImage image_from_nchw(const NpyArray& a) {
    const int h = static_cast<int>(a.shape[2]), w = static_cast<int>(a.shape[3]);
    return SdrImage(PlanarBuffer(3, h, w, a.data));
}

// Elementwise float32 agreement with torch: libm and torch's vectorised pow may
// differ in the last bit, so this is a tight relative bound, not equality.
void expect_close(std::span<const float> got, const std::vector<float>& want, double rtol, double atol,
                  const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = std::abs(double(got[i]) - double(want[i]));
        worst = std::max(worst, d - (atol + rtol * std::abs(double(want[i]))));
    }
    EXPECT_LE(worst, 0.0) << what;
}

}  // namespace

TEST(Baseline, MatchesPythonForEveryExposureAndInput) {
    const auto idx = index_json();
    int checked = 0;
    for (const auto& e : idx.at("baseline")) {
        const float ev = e.at("corpus_ev").get<float>();
        const auto in = npy(idx.at("inputs").at(e.at("input").get<std::string>()).at("file").get<std::string>());
        const auto want = npy(e.at("expected").at("file").get<std::string>());
        const auto got = analytic_baseline(image_from_nchw(in), ev);
        expect_close(got.buffer().span(), want.data, 2e-6, 1e-9,
                     "baseline " + e.at("input").get<std::string>() + " ev " + std::to_string(ev));
        ++checked;
    }
    EXPECT_EQ(checked, 6);
}

TEST(Baseline, ClippedCodeIsFiniteAndBelowCeiling) {
    const float at_clip = baseline_value(1.0f, kLegacyCorpusEv);
    EXPECT_TRUE(std::isfinite(at_clip));
    EXPECT_GT(at_clip, baseline_value(0.99f, kLegacyCorpusEv));
    EXPECT_EQ(baseline_value(0.0f, kLegacyCorpusEv), 0.0f);
}

TEST(Curve, CorrectionMatchesPython) {
    const auto idx = index_json();
    const auto& c = idx.at("curve");
    const auto params = npy(c.at("params").at("file").get<std::string>());
    const auto in = npy(idx.at("inputs").at("image").at("file").get<std::string>());
    const auto want = npy(c.at("expected").at("file").get<std::string>());
    ASSERT_EQ(params.data.size(), 1u + c.at("knots").get<std::size_t>());
    std::vector<float> got(in.data.size());
    for (std::size_t i = 0; i < in.data.size(); ++i) got[i] = curve_correction_log2(in.data[i], params.data);
    expect_close(got, want.data, 0.0, 1e-6, "curve correction");
}

TEST(Tiling, StartsAndWeightsMatchPythonBitForBit) {
    const auto idx = index_json();
    for (const auto& c : idx.at("tiles")) {
        const int H = c.at("full_h"), W = c.at("full_w"), T = c.at("tile"), O = c.at("overlap");
        EXPECT_EQ(tile_starts(H, T, O), c.at("ys").get<std::vector<int>>());
        EXPECT_EQ(tile_starts(W, T, O), c.at("xs").get<std::vector<int>>());
        for (const auto& w : c.at("weights")) {
            const Tile t{w.at("y"), w.at("x"), w.at("h"), w.at("w")};
            const PlanarBuffer got = tile_weight(t, O, H, W);
            const auto row = npy(w.at("row").at("file").get<std::string>());
            const auto col = npy(w.at("col").at("file").get<std::string>());
            for (int x = 0; x < t.w; ++x) ASSERT_EQ(got.at(0, t.h / 2, x), row.data[x]) << "row x=" << x;
            for (int y = 0; y < t.h; ++y) ASSERT_EQ(got.at(0, y, t.w / 2), col.data[y]) << "col y=" << y;
            if (w.contains("full")) {
                const auto full = npy(w.at("full").at("file").get<std::string>());
                ASSERT_EQ(got.span().size(), full.data.size());
                for (std::size_t i = 0; i < full.data.size(); ++i) ASSERT_EQ(got.span()[i], full.data[i]);
            }
        }
    }
}

TEST(Tiling, StitchingAConstantGivesTheConstant) {
    const TileConfig cfg{64, 16};
    const int H = 150, W = 170;
    Stitcher s(1, H, W);
    for (const auto& t : plan_tiles(H, W, cfg)) s.add(t, PlanarBuffer(1, t.h, t.w, 0.25f), tile_weight(t, cfg.overlap, H, W));
    const PlanarBuffer out = std::move(s).finish();
    for (float v : out.span()) ASSERT_NEAR(v, 0.25f, 1e-6f);
}

TEST(Tiling, SmallFrameIsOneTile) {
    const auto tiles = plan_tiles(300, 400, TileConfig{512, 64});
    ASSERT_EQ(tiles.size(), 1u);
    EXPECT_EQ(tiles[0].h, 300);
    EXPECT_EQ(tiles[0].w, 400);
}

// ---- manifest -------------------------------------------------------------------

namespace {

nlohmann::json minimal_manifest() {
    return {
        {"contract", "1.0"},
        {"name", "test"},
        {"source", {{"file", "x.pt"}, {"sha256", "00"}}},
        {"network", {{"corpus_ev", -1.0}, {"log_scale", 16.0}, {"max_hdr", 4.0},
                     {"heads", {{"residual_gate", false}, {"shadow_gate", true}, {"curve", false}}},
                     {"curve_params", 1}}},
        {"tiling", {{"tile_size", 512}, {"overlap", 64}}},
        {"files", {{"torchscript", "model.ts"}, {"onnx_frame", "model.frame.onnx"}, {"onnx_tile", "model.tile.onnx"},
                   {"golden", "golden/golden.json"},
                   {"torchscript_sha256", sha256_hex(std::as_bytes(std::span("ts", 2)))},
                   {"onnx_frame_sha256", sha256_hex(std::as_bytes(std::span("of", 2)))},
                   {"onnx_tile_sha256", sha256_hex(std::as_bytes(std::span("ot", 2)))}}},
        {"onnx_inputs", {{"frame", {"sdr"}}, {"tile", {"sdr"}}}},
        {"tolerance", {{"torchscript", {{"atol", 1e-5}, {"rtol", 0.0}}}, {"onnx", {{"atol", 3e-4}, {"rtol", 1e-4}}}}},
    };
}

fs::path write_package(const std::string& name, const nlohmann::json& manifest) {
    const fs::path dir = fs::temp_directory_path() / ("rudra_manifest_test_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    std::ofstream(dir / "manifest.json") << manifest.dump(2);
    std::ofstream(dir / "model.ts", std::ios::binary) << "ts";
    std::ofstream(dir / "model.frame.onnx", std::ios::binary) << "of";
    std::ofstream(dir / "model.tile.onnx", std::ios::binary) << "ot";
    return dir;
}

}  // namespace

TEST(Manifest, ReadsAndVerifies) {
    const auto dir = write_package("ok", minimal_manifest());
    auto m = read_manifest(dir);
    ASSERT_TRUE(m) << m.error().detail;
    EXPECT_TRUE(m->has_shadow_gate);
    EXPECT_FALSE(m->has_curve);
    EXPECT_DOUBLE_EQ(m->tolerance.at("onnx").atol, 3e-4);
    EXPECT_TRUE(verify_package_files(*m));
}

TEST(Manifest, RefusesAnUnknownContractMajor) {
    auto j = minimal_manifest();
    j["contract"] = "2.0";
    auto m = read_manifest(write_package("contract", j));
    ASSERT_FALSE(m);
    EXPECT_EQ(m.error().code, ErrorCode::ContractMismatch);
}

TEST(Manifest, RefusesAMissingField) {
    auto j = minimal_manifest();
    j.erase("tiling");
    auto m = read_manifest(write_package("missing", j));
    ASSERT_FALSE(m);
    EXPECT_EQ(m.error().code, ErrorCode::ParseError);
}

TEST(Manifest, DetectsATamperedModelFile) {
    const auto dir = write_package("tamper", minimal_manifest());
    std::ofstream(dir / "model.ts", std::ios::binary) << "changed";
    auto m = read_manifest(dir);
    ASSERT_TRUE(m);
    auto v = verify_package_files(*m);
    ASSERT_FALSE(v);
    EXPECT_EQ(v.error().code, ErrorCode::IntegrityError);
}
