// Phase 4 step 1: the video probe and input contract against rudra/video.py
// (tools/emit_video_golden.py). The recorded ffprobe JSON checks the logic on
// any machine; with ffprobe on PATH the committed clips are probed live too.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/media/video_probe.hpp"
#include "rudra/platform/process.hpp"

using namespace rudra;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "video";

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

json index() { return json::parse(slurp(kDir / "index.json")); }

VideoArgs args_for(const json& spec) {
    VideoArgs a;
    a.format = spec.value("format", a.format);
    if (spec.contains("alpha_mode")) a.alpha_mode = spec["alpha_mode"].get<std::string>();
    a.input_transfer = spec.value("input_transfer", a.input_transfer);
    a.input_primaries = spec.value("input_primaries", a.input_primaries);
    a.input_matrix = spec.value("input_matrix", a.input_matrix);
    a.input_range = spec.value("input_range", a.input_range);
    return a;
}

std::string stem(const std::string& clip) {
    std::string s = clip;
    for (char& c : s)
        if (c == '.') c = '_';
    return s;
}

void expect_matches(const json& c, const Result<VideoSource>& got, const std::string& where) {
    SCOPED_TRACE(where + " " + c["clip"].get<std::string>() + " / " + c["args"].get<std::string>());
    if (c.contains("error")) {
        ASSERT_FALSE(got.ok());
        EXPECT_EQ(got.error().message, c["error"].get<std::string>());
        return;
    }
    ASSERT_TRUE(got.ok()) << got.error().message;
    const json& ok = c["ok"];
    EXPECT_EQ(got->contract.transfer, ok["contract"]["transfer"].get<std::string>());
    EXPECT_EQ(got->contract.primaries, ok["contract"]["primaries"].get<std::string>());
    EXPECT_EQ(got->contract.matrix, ok["contract"]["matrix"].get<std::string>());
    EXPECT_EQ(got->contract.range, ok["contract"]["range"].get<std::string>());
    EXPECT_EQ(got->clock.fps, ok["clock"]["fps"].get<std::string>());
    EXPECT_EQ(got->clock.frames, ok["clock"]["frames"].get<int>());
    EXPECT_EQ(got->clock.start, ok["clock"]["start"].get<double>());          // exact: same doubles
    EXPECT_EQ(got->clock.duration, ok["clock"]["duration"].get<double>());
    EXPECT_EQ(got->clock.tolerance, ok["clock"]["tolerance"].get<double>());
    EXPECT_EQ(got->alpha, ok["alpha"].get<bool>());
    EXPECT_EQ(got->stream.width, ok["width"].get<int>());
    EXPECT_EQ(got->stream.height, ok["height"].get<int>());
    EXPECT_EQ(decoder_filter(got->contract, got->alpha), ok["decoder_filter"].get<std::string>());
}
}  // namespace

TEST(VideoProbe, ContractAndClockMatchThePython) {
    const json idx = index();
    ASSERT_GE(idx["cases"].size(), 20u);
    for (const auto& c : idx["cases"]) {
        const std::string s = stem(c["clip"].get<std::string>());
        auto probe = parse_video_probe(slurp(kDir / "probes" / (s + ".streams.json")),
                                       slurp(kDir / "probes" / (s + ".frames.json")));
        ASSERT_TRUE(probe.ok()) << probe.error().message;
        expect_matches(c, open_video_source(*probe, args_for(idx["args"][c["args"].get<std::string>()])), "recorded");
    }
}

TEST(VideoProbe, LiveProbeOfTheClips) {
    if (!find_executable("ffprobe")) GTEST_SKIP() << "ffprobe is not on PATH";
    const json idx = index();
    for (const auto& c : idx["cases"])
        expect_matches(c, open_video_source(kDir / "clips" / c["clip"].get<std::string>(),
                                            args_for(idx["args"][c["args"].get<std::string>()])), "live");
}

TEST(VideoProbe, Fractions) {
    EXPECT_EQ(parse_fraction("24000/1001")->str(), "24000/1001");
    EXPECT_EQ(parse_fraction("50/2")->str(), "25");
    EXPECT_EQ(parse_fraction("30")->str(), "30");
    EXPECT_EQ(parse_fraction("0.5")->str(), "1/2");
    EXPECT_EQ(parse_fraction("1/90000")->value(), 1.0 / 90000.0);
    EXPECT_FALSE(parse_fraction("0/0").ok());
    EXPECT_FALSE(parse_fraction("abc").ok());
}

TEST(VideoProbe, RefusalsTheClipsDoNotReach) {
    VideoStreamInfo s;
    s.width = 64, s.height = 36, s.pix_fmt = "yuv420p", s.avg_frame_rate = "24/1", s.time_base = "1/24";
    s.color_transfer = "bt709", s.color_primaries = "bt709", s.color_space = "bt709", s.color_range = "tv";
    VideoProbe p;
    p.video_streams = {s};
    p.frames = {{0.0, 64, 36}, {1.0 / 24, 64, 36}, {2.0 / 24, 64, 32}};
    EXPECT_EQ(open_video_source(p, {}).error().message, "Changing frame dimensions are unsupported");
    p.frames[1].pts.reset();
    EXPECT_EQ(open_video_source(p, {}).error().message, "Invalid video timestamps");
    p.frames.clear();
    EXPECT_EQ(open_video_source(p, {}).error().message, "No valid video frame rate/timestamps");
    p.frames = {{0.0, 64, 36}};
    p.video_streams[0].avg_frame_rate = "0/0";
    EXPECT_EQ(open_video_source(p, {}).error().message, "No valid video frame rate/timestamps");
    p.video_streams[0].avg_frame_rate = "24/1";
    p.video_streams[0].rotation = -360;
    EXPECT_TRUE(open_video_source(p, {}).ok());   // a full turn is no rotation
    p.video_streams[0].rotation = 180;
    EXPECT_EQ(open_video_source(p, {}).error().message, "Bake the input display rotation before conversion");
    p.video_streams.clear();
    EXPECT_EQ(open_video_source(p, {}).error().message, "Select a source containing exactly one video stream");
    EXPECT_EQ(open_video_source(kDir / "clips" / "no-such.mp4", {}).ok(), false);
}
