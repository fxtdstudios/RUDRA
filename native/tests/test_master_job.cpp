// Phase 3 step 9: the master as a background job.
//
// The render plan's targets written in order, each EXR published with its
// sidecar and nothing replaced, progress reported frame by frame, a cancel
// stopping it after the frame in hand, and a failure stopping it with what
// was done. The frames are synthetic (the network's part is the caller's);
// the bytes of a real master are held to the Studio's by master-check.

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

#include "rudra/engine/master_job.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

MasterFrame frame(int w = 24, int h = 16, float level = 0.5f) {
    MasterFrame f;
    PlanarBuffer sdr(3, h, w, level), residual(3, h, w, 0.2f), hi(1, h, w, 0.5f), sh(1, h, w, 0.0f);
    f.sdr = SdrImage(std::move(sdr));
    f.fields = Fields{std::move(residual), std::move(hi), std::move(sh)};
    return f;
}

fs::path fresh(const std::string& name) {
    const fs::path root = fs::temp_directory_path() / ("rudra-master-" + name);
    fs::remove_all(root);
    fs::create_directories(root);
    return root;
}

std::vector<fs::path> plan(const fs::path& root, int count) {
    RenderPlan p;
    p.render_dir = (root / "out").string();
    p.render_name = "shot";
    p.render_count = count;
    p.sequence = true;
    auto t = master_targets(p);
    EXPECT_TRUE(t) << (t ? "" : t.error().message);
    return t ? *t : std::vector<fs::path>{};
}

const ModelConstants kModel{16.0f, 4.0f, -1.0f};

}  // namespace

TEST(MasterJob, WritesEveryTargetInOrderWithItsSidecar) {
    const fs::path root = fresh("order");
    const auto targets = plan(root, 3);
    ASSERT_EQ(targets.size(), 3u);
    MasterJob job(kModel, MasterRequest{}, targets, [](std::size_t i) -> Result<MasterFrame> {
        return frame(24, 16, 0.3f + 0.2f * float(i));
    });
    std::vector<std::size_t> begun, done;
    std::mutex mu;
    std::promise<MasterOutcome> finished;
    job.start([&](std::size_t i) { std::lock_guard l(mu); begun.push_back(i); },
              [&](const MasterProgress& p) {
                  std::lock_guard l(mu);
                  done.push_back(p.done);
                  EXPECT_EQ(p.total, 3u);
                  EXPECT_EQ(p.last.exr, targets[p.done - 1]);
                  EXPECT_EQ(p.last.width, 24);
              },
              [&](const MasterOutcome& o) { finished.set_value(o); });
    const MasterOutcome o = finished.get_future().get();
    EXPECT_EQ(o.completed, 3u);
    EXPECT_FALSE(o.cancelled);
    EXPECT_FALSE(o.error.has_value());
    EXPECT_EQ(begun, (std::vector<std::size_t>{0, 1, 2}));
    EXPECT_EQ(done, (std::vector<std::size_t>{1, 2, 3}));
    for (const auto& t : targets) {
        EXPECT_TRUE(fs::exists(t)) << t;
        fs::path side = t;
        side.replace_extension(".json");
        ASSERT_TRUE(fs::exists(side)) << side;
        std::ifstream in(side);
        const auto j = nlohmann::json::parse(in);
        EXPECT_EQ(j.at("resolution"), (nlohmann::json{24, 16}));
        EXPECT_EQ(j.at("container"), "ACES 2065-1 (AP0)");
    }
    // Staging leaves nothing behind.
    for (const auto& e : fs::directory_iterator(root / "out"))
        EXPECT_EQ(e.path().filename().string().rfind(".rudra-render-", 0), std::string::npos) << e.path();
    // And a second plan for the same files is refused: nothing is replaced.
    RenderPlan again;
    again.render_dir = (root / "out").string();
    again.render_name = "shot";
    again.render_count = 3;
    again.sequence = true;
    EXPECT_FALSE(master_targets(again));
    fs::remove_all(root);
}

TEST(MasterJob, CancelStopsAfterTheFrameInHand) {
    const fs::path root = fresh("cancel");
    const auto targets = plan(root, 5);
    std::promise<void> first_begun;
    std::atomic<bool> release{false};
    MasterJob job(kModel, MasterRequest{}, targets, [&](std::size_t i) -> Result<MasterFrame> {
        if (i == 0) {
            first_begun.set_value();
            while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return frame();
    });
    std::promise<MasterOutcome> finished;
    job.start({}, {}, [&](const MasterOutcome& o) { finished.set_value(o); });
    first_begun.get_future().wait();
    job.cancel();
    release = true;
    const MasterOutcome o = finished.get_future().get();
    EXPECT_TRUE(o.cancelled);
    EXPECT_EQ(o.completed, 1u);   // the frame in hand is finished, no more
    EXPECT_TRUE(fs::exists(targets[0]));
    EXPECT_FALSE(fs::exists(targets[1]));
    fs::remove_all(root);
}

TEST(MasterJob, AFailureStopsItAndSaysWhatWasDone) {
    const fs::path root = fresh("fail");
    const auto targets = plan(root, 3);
    MasterJob job(kModel, MasterRequest{}, targets, [](std::size_t i) -> Result<MasterFrame> {
        if (i == 1) return make_error(ErrorCode::IoError, "Cannot read frame 2.");
        return frame();
    });
    std::promise<MasterOutcome> finished;
    job.start({}, {}, [&](const MasterOutcome& o) { finished.set_value(o); });
    const MasterOutcome o = finished.get_future().get();
    EXPECT_EQ(o.completed, 1u);
    ASSERT_TRUE(o.error.has_value());
    EXPECT_EQ(o.error->message, "Cannot read frame 2.");
    EXPECT_TRUE(fs::exists(targets[0]));
    EXPECT_FALSE(fs::exists(targets[1]));
    // A bad request fails the first frame, before anything is written.
    MasterRequest bad;
    bad.container = "tiff";
    const fs::path root2 = fresh("bad");
    const auto t2 = plan(root2, 1);
    MasterJob job2(kModel, bad, t2, [](std::size_t) -> Result<MasterFrame> { return frame(); });
    std::promise<MasterOutcome> f2;
    job2.start({}, {}, [&](const MasterOutcome& x) { f2.set_value(x); });
    const MasterOutcome o2 = f2.get_future().get();
    EXPECT_EQ(o2.completed, 0u);
    ASSERT_TRUE(o2.error.has_value());
    EXPECT_FALSE(fs::exists(t2[0]));
    fs::remove_all(root);
    fs::remove_all(root2);
}
