#include <gtest/gtest.h>

#include "rudra/engine/scheduling.hpp"
#include "rudra/render/viewer_backend.hpp"

using namespace rudra;

TEST(Generation, ASeekMakesOlderResultsStale) {
    Generation g;
    const auto requested = g.current();
    EXPECT_TRUE(g.is_current(requested));
    g.bump();   // the user seeks while the request is in flight
    EXPECT_FALSE(g.is_current(requested));
    EXPECT_TRUE(g.is_current(g.current()));
}

TEST(OutputScale, EachSwapchainGetsItsOwnUnit) {
    const Nits thousand{1000.0f}, sdr_white{203.0f};
    EXPECT_FLOAT_EQ(output_scale(OutputPath::ScRgb, thousand, sdr_white), 12.5f);        // 1.0 = 80 nits
    EXPECT_FLOAT_EQ(output_scale(OutputPath::Edr, thousand, sdr_white), 1000.0f / 203.0f);
    EXPECT_FLOAT_EQ(output_scale(OutputPath::Hdr10, thousand, sdr_white), 0.1f);          // then PQ
}

// ---- the frame path (Phase 2 step 10) ----------------------------------------

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "rudra/engine/frame_engine.hpp"

namespace {

// A frame whose every pixel says which frame it is, and fields that say it too:
// a result delivered for the wrong frame is caught by value, not by bookkeeping.
struct Fakes {
    std::chrono::milliseconds decode_time{1}, infer_time{8};
    std::atomic<int> decoded{0}, inferred{0};
    FrameLoader loader() {
        return [this](int i) -> Result<SdrImage> {
            ++decoded;
            std::this_thread::sleep_for(decode_time);
            return SdrImage(PlanarBuffer(3, 2, 2, float(i)));
        };
    }
    FrameInfer infer() {
        return [this](const SdrImage& s) -> Result<FrameResult> {
            ++inferred;
            std::this_thread::sleep_for(infer_time);
            FrameResult r;
            const float tag = s.buffer().span()[0];
            r.fields = Fields{PlanarBuffer(3, 2, 2, tag), PlanarBuffer(1, 2, 2, tag), PlanarBuffer(1, 2, 2, tag)};
            return r;
        };
    }
};

struct Seen {
    std::mutex mu;
    std::vector<ReadyFrame> frames;
    void add(const ReadyFrame& f) {
        std::lock_guard lk(mu);
        frames.push_back(f);
    }
    std::vector<ReadyFrame> all() {
        std::lock_guard lk(mu);
        return frames;
    }
};

bool wait_for(const std::function<bool()>& done, int ms = 3000) {
    for (int t = 0; t < ms && !done(); t += 2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    return done();
}

}  // namespace

TEST(FrameEngine, DeliversOnlyTheFrameItWasAskedFor) {
    Fakes fk;
    FrameEngine engine(fk.loader(), fk.infer(), {32, 4});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(240);
    // A fast scrub: 0, 50, 120, 180 without waiting.
    for (int i : {0, 50, 120, 180}) engine.show(i);
    ASSERT_TRUE(wait_for([&] { auto v = seen.all(); return !v.empty() && v.back().index == 180; }));
    for (const auto& f : seen.all()) {
        ASSERT_FALSE(f.error.has_value());
        EXPECT_EQ(f.sdr->buffer().span()[0], float(f.index)) << "a frame showed another frame's pixels";
        EXPECT_EQ(f.fields->fields.residual.span()[0], float(f.index)) << "a frame showed another frame's fields";
    }
    // Only the last request may be delivered once the scrub settles.
    EXPECT_EQ(seen.all().back().index, 180);
    const auto st = engine.stats();
    EXPECT_GE(st.cancelled, 1u);   // queued work of older generations never ran
}

TEST(FrameEngine, ScrubbingAFolderNeverShowsAStaleFrame) {
    Fakes fk;
    fk.infer_time = std::chrono::milliseconds(3);
    FrameEngine engine(fk.loader(), fk.infer(), {32, 12});
    Seen seen;
    std::atomic<int> asked{-1};
    std::atomic<int> wrong{0};
    engine.on_ready([&](const ReadyFrame& f) {
        if (f.index != asked.load()) ++wrong;   // delivered for a request that is no longer current
        seen.add(f);
    });
    engine.set_sequence(240);
    for (int i = 0; i < 240; ++i) {
        asked = i;
        engine.show(i);
        std::this_thread::sleep_for(std::chrono::microseconds(700));   // faster than inference
    }
    ASSERT_TRUE(wait_for([&] { auto v = seen.all(); return !v.empty() && v.back().index == 239; }));
    EXPECT_EQ(wrong.load(), 0);
    for (const auto& f : seen.all()) EXPECT_EQ(f.sdr->buffer().span()[0], float(f.index));
    const auto st = engine.stats();
    EXPECT_EQ(st.shown, 240u);
    EXPECT_LE(st.cached, 32);
    EXPECT_LT(fk.inferred.load(), 240 + 12 + 1);   // cancelled work was not run
}

TEST(FrameEngine, ReadAheadMakesTheNextFramesFreeAndTheCacheIsBounded) {
    Fakes fk;
    FrameEngine engine(fk.loader(), fk.infer(), {8, 5});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(20);
    engine.show(0);
    ASSERT_TRUE(wait_for([&] { return engine.is_cached(5); }));   // 1..5 read ahead
    const auto before = fk.inferred.load();
    engine.show(3);   // already read ahead: delivered at once, from the cache
    auto v = seen.all();
    ASSERT_FALSE(v.empty());
    EXPECT_EQ(v.back().index, 3);
    EXPECT_TRUE(v.back().from_cache);
    EXPECT_EQ(fk.inferred.load(), before);   // no inference for it
    for (int i = 0; i < 20; ++i) engine.show(i);
    ASSERT_TRUE(wait_for([&] { auto w = seen.all(); return w.back().index == 19; }));
    EXPECT_LE(engine.stats().cached, 8);
    EXPECT_TRUE(engine.is_cached(19));   // the frame on screen is never evicted
}

TEST(FrameEngine, AFailedFrameIsReportedNotShownAsAnother) {
    Fakes fk;
    FrameLoader bad = [&](int i) -> Result<SdrImage> {
        if (i == 7) return make_error(ErrorCode::InvalidArgument, "frame 7 is not an image");
        return SdrImage(PlanarBuffer(3, 2, 2, float(i)));
    };
    FrameEngine engine(bad, fk.infer(), {8, 0});
    Seen seen;
    engine.on_ready([&](const ReadyFrame& f) { seen.add(f); });
    engine.set_sequence(10);
    engine.show(7);
    ASSERT_TRUE(wait_for([&] { return !seen.all().empty(); }));
    const auto f = seen.all().back();
    EXPECT_EQ(f.index, 7);
    ASSERT_TRUE(f.error.has_value());
    EXPECT_EQ(f.error->message, "frame 7 is not an image");
    EXPECT_EQ(engine.stats().errors, 1u);
}
