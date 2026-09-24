// Phase 4 step 9: video jobs in the queue against rudra/batch.py run_queue
// (tools/emit_video_queue_golden.py). A queue the Python stopped at its second
// clip resumes here to the Python's final state, and a native run stopped the
// same way leaves the Python's interrupted state, so each side resumes the
// other's queue. Built with RUDRA_TEST_PACKAGE; needs ffmpeg on PATH.
#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/platform/hash.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/video/queue_runner.hpp"

using namespace rudra;
using ojson = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {
const fs::path kGolden = fs::path(RUDRA_GOLDEN_DIR);
const fs::path kQueue = kGolden / "video_queue";

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

fs::path make_project(const fs::path& where) {
    fs::remove_all(where);
    fs::create_directories(where);
    const ojson q = ojson::parse(slurp(kQueue / "queue.json"));
    for (const auto& j : q["jobs"]) fs::copy_file(kGolden / "video" / "clips" / j["input"].get<std::string>(), where / j["input"].get<std::string>());
    fs::copy_file(kGolden.parent_path().parent_path().parent_path() / "checkpoints" / "sdr2hdr_shadow_v1.pt", where / "sdr2hdr_shadow_v1.pt");
    fs::copy_file(kQueue / "queue.json", where / "queue.json");
    return where / "queue.json";
}

// The state as the Python writes it, artifacts held to the files instead of the golden's digests.
void expect_state(const ojson& got, const ojson& want, const fs::path& project) {
    EXPECT_EQ(got["version"], want["version"]);
    EXPECT_EQ(got["queue_sha256"], want["queue_sha256"]);
    const ojson q = ojson::parse(slurp(kQueue / "queue.json"));
    ASSERT_EQ(got["jobs"].size(), want["jobs"].size());
    for (std::size_t i = 0; i < want["jobs"].size(); ++i) {
        SCOPED_TRACE("job " + std::to_string(i + 1));
        ojson g = got["jobs"][i], w = want["jobs"][i];
        std::vector<std::string> gk, wk;
        for (auto it = g.begin(); it != g.end(); ++it) gk.push_back(it.key());
        for (auto it = w.begin(); it != w.end(); ++it) wk.push_back(it.key());
        EXPECT_EQ(gk, wk);
        if (w.contains("artifacts")) {
            const fs::path out = project.parent_path() / q["jobs"][i]["output"].get<std::string>();
            fs::path side = out;
            side += ".json";
            EXPECT_EQ(g["artifacts"], ojson::array({*sha256_file(out), *sha256_file(side)}));
            g.erase("artifacts");
            w.erase("artifacts");
        }
        EXPECT_EQ(g, w);
    }
}
}  // namespace

TEST(VideoQueueReal, ResumesThePythonsStoppedQueueToItsFinalState) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const ojson idx = ojson::parse(slurp(kQueue / "index.json"));
    const fs::path queue = make_project(fs::temp_directory_path() / "rudra-video-queue-resume");
    fs::copy(kQueue / "interrupted" / "out", queue.parent_path() / "out", fs::copy_options::recursive);
    fs::copy_file(kQueue / "interrupted" / "queue.json.state.json", queue.parent_path() / "queue.json.state.json");
    const ojson before = ojson::parse(slurp(queue.parent_path() / "queue.json.state.json"));

    VideoQueueOptions o;
    o.package = fs::path(RUDRA_TEST_PACKAGE_DIR);
    std::vector<std::string> lines;
    o.print = [&](const std::string& l) { lines.push_back(l); };
    auto r = run_video_queue(queue, false, o);
    ASSERT_TRUE(r) << r.error().message;
    EXPECT_EQ(*r, idx["full_exit"].get<int>());
    ASSERT_FALSE(lines.empty());
    EXPECT_EQ(lines.front(), "Job 1/3: verified, skipped");
    const ojson after = ojson::parse(slurp(queue.parent_path() / "queue.json.state.json"));
    expect_state(after, idx["full"], queue);
    EXPECT_EQ(after["jobs"][0]["artifacts"], before["jobs"][0]["artifacts"]);   // the Python's master, untouched
    fs::remove_all(queue.parent_path());
}

TEST(VideoQueueReal, StopsAsThePythonStopsAndResumes) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const ojson idx = ojson::parse(slurp(kQueue / "index.json"));
    const fs::path queue = make_project(fs::temp_directory_path() / "rudra-video-queue-stop");
    std::atomic<bool> stop{false};
    VideoQueueOptions o;
    o.package_roots = {fs::path(RUDRA_TEST_PACKAGE_DIR).parent_path()};   // found by the checkpoint's digest
    o.cancel = &stop;
    o.print = [&](const std::string& l) {
        if (l.rfind("Job 2/3:", 0) == 0) stop = true;   // Ctrl+C as the second clip starts
    };
    auto r = run_video_queue(queue, false, o);
    ASSERT_FALSE(r) << slurp(queue.parent_path() / "queue.json.state.json");
    EXPECT_EQ(r.error().code, ErrorCode::Cancelled);
    expect_state(ojson::parse(slurp(queue.parent_path() / "queue.json.state.json")), idx["interrupted"], queue);
    EXPECT_FALSE(fs::exists(queue.parent_path() / "out" / "b_hlg.mp4"));

    stop = false;
    o.print = {};
    auto again = run_video_queue(queue, false, o);
    ASSERT_TRUE(again) << again.error().message;
    EXPECT_EQ(*again, 0);
    expect_state(ojson::parse(slurp(queue.parent_path() / "queue.json.state.json")), idx["full"], queue);
    fs::remove_all(queue.parent_path());
}

TEST(VideoQueueReal, AJobWhosePackageIsMissingFailsAndTheQueueMovesOn) {
    const fs::path queue = make_project(fs::temp_directory_path() / "rudra-video-queue-nopkg");
    VideoQueueOptions o;   // no package, no roots
    auto r = run_video_queue(queue, false, o);
    ASSERT_TRUE(r);
    EXPECT_EQ(*r, 1);
    const ojson st = ojson::parse(slurp(queue.parent_path() / "queue.json.state.json"));
    for (const auto& j : st["jobs"]) {
        EXPECT_EQ(j["status"], "failed");
        EXPECT_EQ(j["error"], "No model package exported from this job's checkpoint was found");
    }
    fs::remove_all(queue.parent_path());
}
