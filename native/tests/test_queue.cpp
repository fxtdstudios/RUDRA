// The queue against rudra/batch.py (goldens from tools/emit_queue_golden.py):
// the same state file, byte for byte, at every step, and a Python state
// resumed here to the state the Python itself resumes it to.

#include <algorithm>
#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <set>

#include <nlohmann/json.hpp>

#include "rudra/deliver/queue.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "queue";

std::string text_of(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}
void write_text(const fs::path& p, const std::string& s) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << s;
}

fs::path fresh_project(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / "rudra_queue_test" / name;
    fs::remove_all(d);
    fs::create_directories(d);
    for (const auto& e : fs::directory_iterator(kDir / "project")) fs::copy_file(e.path(), d / e.path().filename());
    return fs::weakly_canonical(d);
}

// The stand-in export tools/emit_queue_golden.py gives the Python.
QueueRunner stand_in(std::set<std::string> fail) {
    return [fail](const QueueJob& job, const QueueProgress& progress) -> Result<void> {
        if (fail.count(job.input.filename().string())) return make_error(ErrorCode::BackendError, "synthetic failure");
        progress(pyjson::Dict{{"phase", "encoding"}, {"frame", 1}, {"of", 2}});
        write_text(job.output, "out:" + job.input.filename().string());
        write_text(job.sidecar(), "{\"qc\": {\"passed\": true}}");
        return {};
    };
}

}  // namespace

TEST(Queue, FreshRunWritesThePythonsState) {
    const fs::path d = fresh_project("fresh");
    auto r = run_queue(d / "queue.json", false, stand_in({"b.mov"}));
    ASSERT_TRUE(r) << r.error().message;
    EXPECT_EQ(*r, 1);
    EXPECT_EQ(text_of(d / "queue.json.state.json"), text_of(kDir / "state_first.json"));

    // ...and resuming our own state lands where the Python's resume does.
    auto again = run_queue(d / "queue.json", true, stand_in({}));
    ASSERT_TRUE(again);
    EXPECT_EQ(*again, 0);
    EXPECT_EQ(text_of(d / "queue.json.state.json"), text_of(kDir / "state_final.json"));
}

TEST(Queue, ResumesAPythonStartedQueue) {
    const fs::path d = fresh_project("resume");
    fs::copy_file(kDir / "state_first.json", d / "queue.json.state.json");
    for (const char* n : {"a", "c"}) {   // what the Python's first run published
        const std::string out = std::string("out/") + n + (std::string(n) == "a" ? ".mp4" : ".mov");
        write_text(d / out, std::string("out:") + n + ".mov");
        write_text(d / (out + ".json"), "{\"qc\": {\"passed\": true}}");
    }
    std::vector<std::string> log;
    auto r = run_queue(d / "queue.json", true, stand_in({}), [&](const std::string& m) { log.push_back(m); });
    ASSERT_TRUE(r) << r.error().message;
    EXPECT_EQ(*r, 0);
    EXPECT_EQ(text_of(d / "queue.json.state.json"), text_of(kDir / "state_final.json"));
    ASSERT_EQ(log.size(), 3u);
    EXPECT_EQ(log[0], "Job 1/3: verified, skipped");
    EXPECT_EQ(log[1], "Job 2/3: b.mov");
    EXPECT_EQ(log[2], "Job 3/3: verified, skipped");
}

TEST(Queue, RefusesWhatThePythonRefuses) {
    std::ifstream in(kDir / "index.json");
    const auto idx = nlohmann::json::parse(in);
    const fs::path d = fresh_project("refusals");
    for (const auto& [name, c] : idx.at("refusals").items()) {
        SCOPED_TRACE(name);
        write_text(d / (name + ".json"), c.at("spec").dump());
        auto r = load_queue(d / (name + ".json"));
        if (c.at("message").is_null()) {
            EXPECT_TRUE(r);
            continue;
        }
        ASSERT_FALSE(r);
        // The golden is written with forward slashes on every OS (see
        // tools/emit_queue_golden.py), so the message is compared the same way.
        std::string want = c.at("message");
        if (auto p = want.find("<root>"); p != std::string::npos) want.replace(p, 6, d.generic_string());
        std::string got = r.error().message;
        std::replace(got.begin(), got.end(), '\\', '/');
        EXPECT_EQ(got, want);
    }
}

TEST(Queue, OneRunnerAtATime) {
    const fs::path d = fresh_project("lock");
    Result<int> inner = 0;
    auto r = run_queue(d / "queue.json", false, [&](const QueueJob& j, const QueueProgress& p) -> Result<void> {
        inner = run_queue(d / "queue.json", false, stand_in({}));
        return stand_in({})(j, p);
    });
    ASSERT_TRUE(r);
    ASSERT_FALSE(inner);
    EXPECT_EQ(inner.error().message, "This queue is already running");
    EXPECT_EQ(queue_status(d / "queue.json"), text_of(d / "queue.json.state.json"));
    EXPECT_EQ(queue_status(d / "nothing.json"), "Queue has not run yet.");
}
