// Phase 4 step 3 with the network: engine/video_predictor on the exported
// package against rudra/video.py Predictor.predict in eager PyTorch
// (tools/emit_video_predict_golden.py): every HDR pixel within the package's
// tolerance, the same gate weights and the same cuts, whole frames and tiles,
// on every runtime this build has. Built when RUDRA_TEST_PACKAGE is set.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/core/model_manifest.hpp"
#include "rudra/video/predictor.hpp"
#include "rudra/media/video_decode.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "video_predict";

RawFrame16 raw(const NpyArray& a) {
    RawFrame16 f;
    f.height = int(a.shape[0]), f.width = int(a.shape[1]), f.channels = int(a.shape[2]);
    f.data.assign(a.data.begin(), a.data.end());
    return f;
}
}  // namespace

TEST(VideoPredictReal, FramesWeightsAndCutsAreThePythons) {
    auto m = read_manifest(RUDRA_TEST_PACKAGE_DIR);
    ASSERT_TRUE(m) << m.error().message;
    std::ifstream f(kDir / "index.json");
    std::stringstream ss;
    ss << f.rdbuf();
    const json idx = json::parse(ss.str());
    const ModelConstants model{m->log_scale, m->max_hdr, m->corpus_ev};

    for (const Runtime rt : compiled_runtimes()) {
        auto backend = rt == Runtime::LibTorch ? make_libtorch_backend(*m, Device::Cpu)
                                               : make_onnxruntime_backend(*m, Device::Cpu);
        ASSERT_TRUE(backend) << backend.error().message;
        const Tolerance tol = m->tolerance.at(rt == Runtime::LibTorch ? "torchscript" : "onnx");
        for (const auto& s : idx["sequences"]) {
            const std::string name = s["name"].get<std::string>();
            SCOPED_TRACE(std::string(to_string(rt)) + " " + name);
            VideoPredictor predictor(**backend, model, s["tile_size"].get<int>(), s["overlap"].get<int>());
            ShadowSmoother smoother(s["retention"].get<double>(), s["cut_threshold"].get<double>());
            double worst_excess = 0, worst_abs = 0;
            for (const auto& fr : s["frames"]) {
                auto in = read_npy(kDir / "frames" / fr["input"].get<std::string>());
                auto want = read_npy(kDir / "frames" / fr["hdr"].get<std::string>());   // H x W x 3
                ASSERT_TRUE(in && want);
                const auto& c = fr["contract"];
                auto got = predictor.predict(rgb_from_frame(raw(*in)), c["transfer"].get<std::string>(),
                                             c["primaries"].get<std::string>(), smoother);
                ASSERT_TRUE(got) << got.error().message;
                EXPECT_EQ(got->cut, fr["cut"].get<bool>()) << fr["hdr"];
                if (!fr["raw_weight"].is_null()) {
                    EXPECT_NEAR(got->raw_weight, fr["raw_weight"].get<double>(), tol.atol + tol.rtol) << fr["hdr"];
                }
                EXPECT_NEAR(got->shadow_weight, fr["shadow_weight"].get<double>(), tol.atol + tol.rtol) << fr["hdr"];
                const int h = got->hdr.height(), w = got->hdr.width();
                ASSERT_EQ(want->data.size(), std::size_t(3) * h * w);
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x)
                        for (int ch = 0; ch < 3; ++ch) {
                            const double e = want->data[(std::size_t(y) * w + x) * 3 + ch];
                            const double g = got->hdr.buffer().at(ch, y, x);
                            worst_abs = std::max(worst_abs, std::fabs(g - e));
                            worst_excess = std::max(worst_excess, std::fabs(g - e) - (tol.atol + tol.rtol * std::fabs(e)));
                        }
            }
            std::cout << "[ video ] " << to_string(rt) << " " << name << ": max |native - eager| " << worst_abs << "\n";
            EXPECT_LE(worst_excess, 0.0) << "beyond atol " << tol.atol << " + rtol " << tol.rtol;
        }
    }
}
