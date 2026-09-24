// Phase 4 step 3, the parts around the network: rudra/video.py Predictor's
// canonicalisation to Rec.2020 code, the cut thumbnail (torch's area resize),
// the tile origins and feathers (torch.linspace), and ShadowSmoother, held to
// tools/emit_video_predict_golden.py. The network run is test_video_predict_real.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/core/video_predict.hpp"
#include "rudra/media/video_decode.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "video_predict";

json index() {
    std::ifstream f(kDir / "index.json");
    std::stringstream ss;
    ss << f.rdbuf();
    return json::parse(ss.str());
}

NpyArray npy(const std::string& name) {
    auto a = read_npy(kDir / "frames" / name);
    EXPECT_TRUE(a.ok()) << name;
    return a.ok() ? std::move(*a) : NpyArray{};
}

PlanarBuffer planar(const NpyArray& a) {   // 3 x H x W
    return PlanarBuffer(int(a.shape[0]), int(a.shape[1]), int(a.shape[2]), a.data);
}

RawFrame16 raw(const NpyArray& a) {        // H x W x C, uint16 widened
    RawFrame16 f;
    f.height = int(a.shape[0]), f.width = int(a.shape[1]), f.channels = int(a.shape[2]);
    f.data.assign(a.data.begin(), a.data.end());
    return f;
}

float ulps(float a, float b) {
    return std::fabs(a - b) / std::max(std::numeric_limits<float>::epsilon() * std::max(std::fabs(a), std::fabs(b)),
                                       std::numeric_limits<float>::denorm_min());
}
}  // namespace

TEST(VideoPredict, LinspaceIsTorchs) {
    const json idx = index();
    for (const auto& [n, values] : idx["linspace_up"].items()) {
        const auto v = torch_linspace(0.001f, 1.0f, std::stoi(n));
        ASSERT_EQ(v.size(), values.size());
        for (std::size_t i = 0; i < v.size(); ++i) EXPECT_EQ(v[i], values[i].get<float>()) << n << " " << i;
    }
    for (const auto& [n, values] : idx["linspace_down"].items()) {
        const auto v = torch_linspace(1.0f, 0.001f, std::stoi(n));
        for (std::size_t i = 0; i < v.size(); ++i) EXPECT_EQ(v[i], values[i].get<float>()) << n << " " << i;
    }
}

TEST(VideoPredict, AreaResizeIsTorchs) {
    for (const auto& c : index()["area"]) {
        const PlanarBuffer in = planar(npy(c["input"].get<std::string>()));
        const NpyArray want = npy(c["output"].get<std::string>());
        const PlanarBuffer got = area_resize(in, c["size"][0].get<int>(), c["size"][1].get<int>());
        ASSERT_EQ(got.span().size(), want.data.size());
        float worst = 0;
        for (std::size_t i = 0; i < want.data.size(); ++i) worst = std::max(worst, ulps(got.span()[i], want.data[i]));
        EXPECT_LE(worst, 1.0f) << c["input"];
    }
}

TEST(VideoPredict, TileOriginsAndFeathers) {
    EXPECT_EQ(video_tile_starts(64, 32, 8), (std::vector<int>{0, 24, 32}));
    EXPECT_EQ(video_tile_starts(36, 32, 8), (std::vector<int>{0, 4}));
    EXPECT_EQ(video_tile_starts(20, 32, 8), (std::vector<int>{0}));
    EXPECT_EQ(video_tile_starts(64, 24, 4), (std::vector<int>{0, 20, 40}));
    // A corner tile feathers only toward its neighbours.
    const auto b = video_tile_blend(0, 0, 4, 4, 8, 8, 2);
    EXPECT_EQ(b[0], 1.0f);
    EXPECT_FLOAT_EQ(b[3], 0.001f);        // last column ramps down
    EXPECT_FLOAT_EQ(b[15], 0.001f * 0.001f);
    const auto inner = video_tile_blend(2, 2, 4, 4, 8, 8, 2);
    EXPECT_FLOAT_EQ(inner[0], 0.001f * 0.001f);
}

TEST(VideoPredict, CanonicalFrameThumbAndSmootherAreThePythons) {
    const json idx = index();
    for (const auto& s : idx["sequences"]) {
        SCOPED_TRACE(s["name"].get<std::string>());
        ShadowSmoother smoother(s["retention"].get<double>(), s["cut_threshold"].get<double>());
        for (const auto& f : s["frames"]) {
            const auto contract = f["contract"];
            const SdrImage rgb = rgb_from_frame(raw(npy(f["input"].get<std::string>())));
            auto x = canonicalize_sdr(rgb.buffer(), contract["transfer"].get<std::string>(), "full");
            ASSERT_TRUE(x.ok());
            PlanarBuffer code = std::move(*x);
            if (contract["primaries"] != "rec2020") code = srgb_code_to_rec2020(code, Primaries::Rec709);
            const NpyArray want = npy(f["canonical"].get<std::string>());
            ASSERT_EQ(code.span().size(), want.data.size());
            float worst = 0;
            for (std::size_t i = 0; i < want.data.size(); ++i) worst = std::max(worst, std::fabs(code.span()[i] - want.data[i]));
            EXPECT_LE(worst, 2e-6f) << f["canonical"];   // pow and the 3x3 in float32, a few ulp

            // The thumbnail from the Python's own canonical frame, and the smoother on the Python's gate.
            const PlanarBuffer thumb = area_resize(planar(want), kCutThumbHeight, kCutThumbWidth);
            const NpyArray want_thumb = npy(f["thumb"].get<std::string>());
            for (std::size_t i = 0; i < want_thumb.data.size(); ++i)
                ASSERT_LE(ulps(thumb.span()[i], want_thumb.data[i]), 1.0f);
            const double raw_weight = f["raw_weight"].is_null() ? 1.0 : f["raw_weight"].get<double>();
            const auto [weight, cut] = smoother.update(planar(want_thumb), raw_weight);
            EXPECT_EQ(cut, f["cut"].get<bool>()) << f["hdr"];
            EXPECT_DOUBLE_EQ(weight, f["shadow_weight"].get<double>()) << f["hdr"];
        }
    }
}

TEST(VideoPredict, CanonicalisationRefusesUnknownEncodings) {
    PlanarBuffer b(3, 1, 1, 0.5f);
    EXPECT_FALSE(canonicalize_sdr(b, "pq", "full").ok());
    EXPECT_FALSE(canonicalize_sdr(b, "srgb", "studio").ok());
    auto lim = canonicalize_sdr(PlanarBuffer(3, 1, 1, 16.0f / 255.0f), "srgb", "limited");
    ASSERT_TRUE(lim.ok());
    EXPECT_EQ(lim->span()[0], 0.0f);
}
