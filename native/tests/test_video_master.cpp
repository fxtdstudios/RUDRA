// Phase 4 step 4: mastering and the spool against convert_video's frame loop
// (tools/emit_video_master_golden.py). For HDR10 at 1000 and at 4000 nits with
// a knee, HLG at 1000 and 600, and ProRes 4444 with alpha, every spool PNG's
// code values, each frame's MaxCLL and average, and the encoder's ceilings
// equal the Python's (PQ codes within one: see below); the PNG bytes too when
// the codes are identical and this OpenCV is the recording one.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/deliver/video_master.hpp"
#include "rudra/media/png16.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kGolden = fs::path(RUDRA_GOLDEN_DIR);

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

NetworkLinearImage hdr_image(const NpyArray& a) {   // H x W x 3 -> planar
    const int h = int(a.shape[0]), w = int(a.shape[1]);
    PlanarBuffer b(3, h, w);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) b.at(c, y, x) = a.data[(std::size_t(y) * w + x) * 3 + c];
    return NetworkLinearImage(std::move(b));
}
}  // namespace

TEST(VideoMaster, SpoolFramesAreThePythons) {
    const json idx = json::parse(slurp(kGolden / "video_master" / "index.json"));
    const bool same_opencv = idx["opencv"].get<std::string>() == opencv_version();
    const fs::path out = fs::temp_directory_path() / "rudra-video-master-test";
    fs::create_directories(out);
    // The alpha run's decoded frames: the step 3 goldens' raw inputs of that clip.
    auto alpha_in = read_npy(kGolden / "video_predict" / "frames" / "alpha_odd_tiles_00.in.npy");
    ASSERT_TRUE(alpha_in);
    for (const auto& r : idx["runs"]) {
        SCOPED_TRACE(r["name"].get<std::string>());
        const std::optional<double> knee =
            r["knee_nits"].is_null() ? std::nullopt : std::optional<double>(r["knee_nits"].get<double>());
        std::vector<VideoFrameLight> lights;
        int index = 0;
        for (const auto& f : r["frames"]) {
            auto hdr = read_npy(kGolden / "video_predict" / "frames" / f["hdr"].get<std::string>());
            ASSERT_TRUE(hdr);
            std::vector<std::uint16_t> alpha;
            if (r["alpha"].get<bool>()) {
                // convert_video appends the frame's own decoded alpha; the clip's alpha is constant,
                // so frame 0's plane stands in for every frame (checked against the PNG below).
                alpha.assign(alpha_in->data.begin(), alpha_in->data.end());
            }
            auto got = master_video_frame(hdr_image(*hdr), index, r["format"].get<std::string>(),
                                          r["peak_nits"].get<double>(), knee, alpha);
            ASSERT_TRUE(got) << got.error().message;
            auto want = read_png16(kGolden / "video_master" / "spool" / f["png"].get<std::string>());
            ASSERT_TRUE(want) << want.error().message;
            ASSERT_EQ(want->channels, got->channels);
            ASSERT_EQ(want->data.size(), got->packed.size());
            std::size_t diff = 0;
            int worst = 0;
            for (std::size_t i = 0; i < got->packed.size(); ++i) {
                diff += got->packed[i] != want->data[i];
                worst = std::max(worst, std::abs(int(got->packed[i]) - int(want->data[i])));
            }
            // numpy's float32 power runs SIMD code (SVML on AVX-512) that is not correctly
            // rounded; libm's powf is. PQ codes may differ by one in 65535 where the two
            // powers straddle a rounding boundary; HLG (float64) and the light must not.
            EXPECT_LE(worst, r["format"] == "hlg" ? 0 : 1) << f["png"];
            EXPECT_LE(double(diff), 0.05 * double(got->packed.size())) << f["png"];
            EXPECT_EQ(got->light.max_cll, f["max_cll"].get<double>()) << f["png"];
            EXPECT_EQ(got->light.frame_average, f["frame_average"].get<double>()) << f["png"];
            lights.push_back(got->light);

            const fs::path mine = out / spool_frame_name(index);
            ASSERT_TRUE(write_png16(mine, got->packed, got->width, got->height, got->channels).ok());
            auto back = read_png16(mine);
            ASSERT_TRUE(back);
            EXPECT_EQ(back->data, got->packed);
            if (same_opencv && diff == 0) {
                const std::string bytes = slurp(mine);
                EXPECT_EQ(sha256_hex(std::as_bytes(std::span(bytes.data(), bytes.size()))), f["png_sha256"].get<std::string>());
            }
            ++index;
        }
        const auto ceil = light_ceilings(lights);
        EXPECT_EQ(ceil.max_cll, r["max_cll"].get<int>());
        EXPECT_EQ(ceil.max_fall, r["max_fall"].get<int>());
    }
    EXPECT_EQ(spool_frame_name(7), "00000007.png");
}

TEST(VideoMaster, NonFinitePixelsAreRefused) {
    PlanarBuffer b(3, 2, 2, 0.01f);
    b.at(1, 1, 1) = std::numeric_limits<float>::quiet_NaN();
    auto r = master_video_frame(NetworkLinearImage(b), 12, "hdr10", 1000, std::nullopt);
    ASSERT_FALSE(r);
    EXPECT_EQ(r.error().message, "Invalid HDR pixels at frame 12");
    b.at(1, 1, 1) = -0.001f;
    EXPECT_EQ(master_video_frame(NetworkLinearImage(b), 3, "hlg", 1000, std::nullopt).error().message,
              "Invalid HDR pixels at frame 3");
    EXPECT_FALSE(write_png16(fs::temp_directory_path() / "no-such-dir-rudra" / "x.png",
                             std::vector<std::uint16_t>(12), 2, 2, 3).ok());
    EXPECT_FALSE(spool_space_low(fs::temp_directory_path(), 1024));
}
