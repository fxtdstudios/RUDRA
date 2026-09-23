// The frame path with a real model (Phase 2 step 10): the FrameEngine driving
// media (still decode) and infer (the exported package, CPU) through a fast
// scrub of a folder of real images, the decode goldens. Every frame delivered
// must be the frame asked for, pixel for pixel and field for field; a file
// the decoder refuses is reported, not replaced; latency is recorded.
// Built when RUDRA_TEST_PACKAGE points at a package and OpenCV is on.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "rudra/core/model_manifest.hpp"
#include "rudra/engine/frame_engine.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/media/sequence.hpp"
#include "rudra/media/still.hpp"

using namespace rudra;

TEST(FrameEngineReal, ScrubsAFolderOfRealImagesWithoutAStaleFrame) {
    auto m = read_manifest(RUDRA_TEST_PACKAGE_DIR);
    ASSERT_TRUE(m) << m.error().message;
    auto backend = make_onnxruntime_backend(*m, Device::Cpu);
    if (!backend) backend = make_libtorch_backend(*m, Device::Cpu);
    ASSERT_TRUE(backend) << backend.error().message;
    auto seq = open_sequence((std::filesystem::path(RUDRA_GOLDEN_DIR) / "decode").string());
    ASSERT_TRUE(seq) << seq.error().message;
    const auto files = seq->frames;
    const int n = int(files.size());
    ASSERT_GE(n, 10);

    InferenceBackend* be = backend->get();
    FrameEngine engine(
        [&files](int i) -> Result<SdrImage> {
            auto d = decode_sdr_file(files[std::size_t(i)]);
            if (!d) return d.error();
            return std::move(d->rgb);
        },
        [be](const SdrImage& s) { return infer_frame(*be, s, TileConfig{0, 0}); }, {32, 4});

    std::mutex mu;
    std::vector<ReadyFrame> seen;
    std::vector<double> latency;
    std::chrono::steady_clock::time_point asked;
    int wanted = -1, wrong = 0;
    engine.on_ready([&](const ReadyFrame& f) {
        std::lock_guard lk(mu);
        if (f.index != wanted) ++wrong;
        seen.push_back(f);
        latency.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - asked).count());
    });
    engine.set_sequence(n);
    auto ask = [&](int i) {
        {
            std::lock_guard lk(mu);
            wanted = i;
            asked = std::chrono::steady_clock::now();
        }
        engine.show(i);
    };
    auto settled = [&](int i) {
        for (int t = 0; t < 20000; t += 5) {
            {
                std::lock_guard lk(mu);
                if (!seen.empty() && seen.back().index == i) return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return false;
    };
    // A fast scrub forward, then every frame one at a time, then back again (cached).
    for (int i = 0; i < n; ++i) ask(i);
    ASSERT_TRUE(settled(n - 1));
    for (int i = 0; i < n; ++i) {
        ask(i);
        ASSERT_TRUE(settled(i)) << "frame " << i << " never arrived";
    }
    for (int i = n - 1; i >= 0; --i) {
        ask(i);
        ASSERT_TRUE(settled(i));
    }

    std::lock_guard lk(mu);
    EXPECT_EQ(wrong, 0);
    int refused = 0;
    for (const auto& f : seen) {
        const auto direct = decode_sdr_file(files[std::size_t(f.index)]);
        if (!direct) {
            ASSERT_TRUE(f.error.has_value()) << files[std::size_t(f.index)];
            EXPECT_EQ(f.error->message, direct.error().message);
            ++refused;
            continue;
        }
        ASSERT_FALSE(f.error.has_value()) << files[std::size_t(f.index)] << ": " << f.error->message;
        ASSERT_TRUE(std::equal(f.sdr->buffer().span().begin(), f.sdr->buffer().span().end(),
                               direct->rgb.buffer().span().begin()))
            << "frame " << f.index << " carried another frame's pixels";
    }
    EXPECT_GT(refused, 0);   // the float TIFF the decoder refuses
    // The two cached passes are free; record what a miss costs.
    std::vector<double> sorted = latency;
    std::sort(sorted.begin(), sorted.end());
    const auto st = engine.stats();
    ::testing::Test::RecordProperty("frames", n);
    ::testing::Test::RecordProperty("median_ms", std::to_string(sorted[sorted.size() / 2]));
    ::testing::Test::RecordProperty("max_ms", std::to_string(sorted.back()));
    ::testing::Test::RecordProperty("cache_hits", std::to_string(st.cache_hits));
    ::testing::Test::RecordProperty("cancelled", std::to_string(st.cancelled));
    EXPECT_GE(st.cache_hits, std::uint64_t(n));   // the way back came from the cache
}
