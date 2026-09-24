// Phase 4 step 5: the encoder's command, argument for argument, against the
// command convert_video itself built (tools/emit_video_master_golden.py) for
// all five profiles, audio copy, AAC and none, CRF and preset, MKV and an
// upper-case .MOV; the spool of the Python run encodes with the native command
// when ffmpeg is on PATH; publishing never replaces a file.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/deliver/video_encode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/process.hpp"

using namespace rudra;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kGolden = fs::path(RUDRA_GOLDEN_DIR);

json master_index() {
    std::ifstream f(kGolden / "video_master" / "index.json");
    std::stringstream ss;
    ss << f.rdbuf();
    return json::parse(ss.str());
}

VideoEncodeRequest request_of(const json& r) {
    VideoEncodeRequest q;
    q.format = r["format"].get<std::string>();
    q.preset = r["preset"].get<std::string>();
    q.crf = r["crf"].get<int>();
    q.audio = r["audio"].get<std::string>();
    q.peak_nits = r["peak_nits"].get<double>();
    q.min_nits = r["min_nits"].get<double>();
    q.alpha = r["alpha"].get<bool>();
    return q;
}

VideoClock clock_of(const json& c) {
    return VideoClock{c["fps"].get<std::string>(), c["frames"].get<int>(), c["start"].get<double>(),
                      c["duration"].get<double>(), c["tolerance"].get<double>()};
}
}  // namespace

TEST(VideoEncode, CommandIsThePythons) {
    const json idx = master_index();
    ASSERT_GE(idx["runs"].size(), 10u);
    for (const auto& r : idx["runs"]) {
        SCOPED_TRACE(r["name"].get<std::string>());
        const std::string suffix = r["suffix"].get<std::string>();
        auto cmd = encode_command("ffmpeg", request_of(r["request"]), clock_of(r["clock"]), "SPOOL",
                                  r["clip"].get<std::string>(), "OUTPUT" + suffix, r["max_cll"].get<int>(),
                                  r["max_fall"].get<int>());
        ASSERT_TRUE(cmd) << cmd.error().message;
        for (auto& a : *cmd)
            if (a == (fs::path("SPOOL") / "%08d.png").string()) a = "SPOOL/%08d.png";
        EXPECT_EQ(json(*cmd), r["command"]);
    }
    EXPECT_EQ(mastering_display(1000, 0.005), "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,50)");
    EXPECT_EQ(mastering_display(1000, 0.00005), "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(10000000,0)");   // 0.5 rounds to even
    VideoEncodeRequest bad;
    bad.format = "av1";
    EXPECT_FALSE(encode_command("ffmpeg", bad, {}, "s", "i", "o.mp4", 1, 1).ok());
}

TEST(VideoEncode, ThePythonsSpoolEncodesWithTheNativeCommand) {
    auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const json idx = master_index();
    for (const char* name : {"hdr10", "hlg", "prores4444_alpha"}) {
        SCOPED_TRACE(name);
        const json* run = nullptr;
        for (const auto& r : idx["runs"])
            if (r["name"] == name) run = &r;
        ASSERT_NE(run, nullptr);
        const fs::path work = fs::temp_directory_path() / (std::string("rudra-encode-test-") + name);
        fs::remove_all(work);
        fs::create_directories(work / "spool");
        for (int i = 0; i < (*run)["clock"]["frames"].get<int>(); ++i) {
            char n[32];
            std::snprintf(n, sizeof n, "%08d.png", i);
            fs::copy_file(kGolden / "video_master" / "spool" / (std::string(name) + "_" + n), work / "spool" / n);
        }
        auto staging = StagingDir::create(work, "rudra-master-");
        ASSERT_TRUE(staging);
        const std::string suffix = (*run)["suffix"].get<std::string>();
        const fs::path staged = staging->path() / ("master" + suffix);
        auto cmd = encode_command(ffmpeg->string(), request_of((*run)["request"]), clock_of((*run)["clock"]),
                                  work / "spool", kGolden / "video" / "clips" / (*run)["clip"].get<std::string>(),
                                  staged, (*run)["max_cll"].get<int>(), (*run)["max_fall"].get<int>());
        ASSERT_TRUE(cmd);
        auto ran = run_tool(*cmd);
        ASSERT_TRUE(ran) << ran.error().message;
        auto probe = probe_video(staged);
        ASSERT_TRUE(probe) << probe.error().message;
        ASSERT_EQ(probe->video_streams.size(), 1u);
        EXPECT_EQ(probe->frames.size(), std::size_t((*run)["clock"]["frames"].get<int>()));
        EXPECT_EQ(probe->video_streams[0].codec_name, std::string(name) == std::string("prores4444_alpha") ? "prores" : "hevc");
        EXPECT_EQ(probe->video_streams[0].color_transfer.value_or(""), std::string(name) == std::string("hlg") ? "arib-std-b67" : "smpte2084");
        { std::ofstream(staging->path() / "master.json") << "{}"; }
        EXPECT_TRUE(publish_video(staged, staging->path() / "master.json", work / ("out" + suffix), work / "out.json").ok());
        EXPECT_TRUE(fs::exists(work / ("out" + suffix)));
        const fs::path kept = staging->path();
        { auto moved = std::move(*staging); }
        EXPECT_FALSE(fs::exists(kept));   // the staging folder goes with its owner
        fs::remove_all(work);
    }
}

TEST(VideoEncode, PublishingNeverReplaces) {
    const fs::path work = fs::temp_directory_path() / "rudra-publish-test";
    fs::remove_all(work);
    fs::create_directories(work);
    { std::ofstream(work / "staged.mp4") << "new"; }
    { std::ofstream(work / "staged.json") << "{}"; }
    { std::ofstream(work / "out.mp4") << "old"; }
    auto r = publish_video(work / "staged.mp4", work / "staged.json", work / "out.mp4", work / "out.mp4.json");
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().message, "Output appeared during processing; refusing overwrite");
    std::ifstream in(work / "out.mp4");
    std::string s;
    in >> s;
    EXPECT_EQ(s, "old");
    fs::remove(work / "out.mp4");
    { std::ofstream(work / "out.mp4.json") << "{}"; }
    EXPECT_FALSE(publish_video(work / "staged.mp4", work / "staged.json", work / "out.mp4", work / "out.mp4.json").ok());
    EXPECT_FALSE(fs::exists(work / "out.mp4"));
    fs::remove_all(work);
}
