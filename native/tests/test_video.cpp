// Phase 4 step 1: the video probe and input contract against rudra/video.py
// (tools/emit_video_golden.py). The recorded ffprobe JSON checks the logic on
// any machine; with ffprobe on PATH the committed clips are probed live too.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/platform/hash.hpp"
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

// Step 2: the decoder. Its command is held to the Python's argument for
// argument; with ffmpeg on PATH every frame is decoded, and when that ffmpeg is
// the build that recorded the goldens, each frame's bytes equal the Python's.
namespace {
std::string first_line(const std::string& s) { return s.substr(0, s.find_first_of("\r\n")); }

std::string sha(const RawFrame16& f) {
    return sha256_hex(std::as_bytes(std::span(f.data.data(), f.data.size())));
}

const json& decode_case_args(const json& idx, const json& c) { return idx["args"][c["args"].get<std::string>()]; }
}  // namespace

TEST(VideoDecode, CommandIsThePythons) {
    const json idx = index();
    const json dec = json::parse(slurp(kDir / "decode.json"));
    ASSERT_GE(dec["cases"].size(), 6u);
    for (const auto& c : dec["cases"]) {
        const std::string clip = c["clip"].get<std::string>();
        const std::string s = stem(clip);
        auto probe = parse_video_probe(slurp(kDir / "probes" / (s + ".streams.json")),
                                       slurp(kDir / "probes" / (s + ".frames.json")));
        ASSERT_TRUE(probe.ok());
        auto src = open_video_source(*probe, args_for(decode_case_args(idx, c)));
        ASSERT_TRUE(src.ok()) << clip << ": " << src.error().message;
        EXPECT_EQ(json(decoder_command("ffmpeg", clip, src->contract, src->alpha)), c["command"]) << clip;
    }
}

TEST(VideoDecode, FramesAreThePythons) {
    auto ffmpeg = find_executable("ffmpeg");
    if (!ffmpeg || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const json idx = index();
    const json dec = json::parse(slurp(kDir / "decode.json"));
    auto version = run_process({ffmpeg->string(), "-version"});
    ASSERT_TRUE(version.ok());
    const bool same_build = first_line(version->out) == dec["ffmpeg"].get<std::string>();
    const fs::path log = fs::temp_directory_path() / "rudra-video-decode-test.log";
    for (const auto& c : dec["cases"]) {
        const std::string clip = c["clip"].get<std::string>();
        SCOPED_TRACE(clip);
        auto src = open_video_source(kDir / "clips" / clip, args_for(decode_case_args(idx, c)));
        ASSERT_TRUE(src.ok()) << src.error().message;
        auto d = VideoDecoder::open(kDir / "clips" / clip, *src, log);
        ASSERT_TRUE(d.ok()) << d.error().message;
        ASSERT_EQ((*d)->frames(), static_cast<int>(c["frames"].size()));
        for (int i = 0; i < (*d)->frames(); ++i) {
            auto f = (*d)->next();
            ASSERT_TRUE(f.ok()) << f.error().message;
            EXPECT_EQ(f->channels, src->alpha ? 4 : 3);
            if (same_build) {
                EXPECT_EQ(sha(*f), c["frames"][static_cast<std::size_t>(i)].get<std::string>()) << i;
            }
            const SdrImage rgb = rgb_from_frame(*f);
            EXPECT_EQ(rgb.buffer().at(1, 0, 0), static_cast<float>(f->data[1]) / 65535.0f);
        }
        const auto end = (*d)->finish();
        EXPECT_TRUE(end.ok()) << (end.ok() ? "" : end.error().message);
    }
    if (!same_build) std::cout << "[ info ] ffmpeg differs from the golden build; frame bytes not compared\n";
}

TEST(VideoDecode, ShortLongAndFailedStreamsAreRefused) {
    const fs::path dir = fs::temp_directory_path() / "rudra-video-decode-refusals";
    fs::create_directories(dir);
    const int w = 4, h = 2, frame = w * h * 3 * 2;
    auto write = [&](const std::string& name, int bytes) {
        std::string data(static_cast<std::size_t>(bytes), '\0');
        for (int i = 0; i < bytes; ++i) data[static_cast<std::size_t>(i)] = static_cast<char>(i * 7);
        std::ofstream(dir / name, std::ios::binary).write(data.data(), bytes);
        return (dir / name).string();
    };
    auto cat = [&](const std::string& file, int frames) {
        auto d = VideoDecoder::open_command({RUDRA_CMAKE_COMMAND, "-E", "cat", file}, w, h, 3, frames, dir / "log.txt");
        EXPECT_TRUE(d.ok());
        return std::move(*d);
    };
    {   // exactly three frames
        auto d = cat(write("three.raw", 3 * frame), 3);
        for (int i = 0; i < 3; ++i) {
            auto f = d->next();
            ASSERT_TRUE(f.ok());
            EXPECT_EQ(f->data[0], static_cast<std::uint16_t>(((i * frame * 7) & 0xff) | ((((i * frame + 1) * 7) & 0xff) << 8)));
        }
        EXPECT_TRUE(d->finish().ok());
    }
    {
        auto d = cat(write("short.raw", 2 * frame), 3);
        EXPECT_TRUE(d->next().ok());
        EXPECT_TRUE(d->next().ok());
        EXPECT_EQ(d->next().error().message, "Decoder ended before expected frame count");
    }
    {
        auto d = cat(write("partial.raw", 2 * frame + 5), 3);
        EXPECT_TRUE(d->next().ok());
        EXPECT_TRUE(d->next().ok());
        EXPECT_EQ(d->next().error().message, "Truncated decoded frame");
    }
    {
        auto d = cat(write("extra.raw", 3 * frame), 2);
        EXPECT_TRUE(d->next().ok());
        EXPECT_TRUE(d->next().ok());
        EXPECT_EQ(d->finish().error().message, "Decoder produced unexpected extra frames");
    }
    {
        auto d = cat((dir / "missing.raw").string(), 0);
        const auto r = d->finish();
        ASSERT_FALSE(r.ok());
        EXPECT_TRUE(r.error().message.starts_with("Decoder failed; ")) << r.error().message;
        EXPECT_GT(r.error().message.size(), std::string("Decoder failed; ").size());   // the log's tail
    }
}
