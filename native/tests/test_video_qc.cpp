// Phase 4 step 6: export QC against rudra/video.py quality_check and
// check_alpha (tools/emit_video_qc_golden.py). Five masters convert_video made
// and five broken from them, fourteen checks: the same verdict, the same
// words, and for a pass the same record byte for byte as json.dumps writes it.
// From ffprobe's recorded JSON on any machine; live with ffmpeg on PATH.
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/deliver/video_qc.hpp"
#include "rudra/engine/video_qc.hpp"
#include "rudra/platform/process.hpp"

using namespace rudra;
using nlohmann::json;
namespace fs = std::filesystem;

namespace {
const fs::path kGolden = fs::path(RUDRA_GOLDEN_DIR);
const fs::path kQc = kGolden / "video_qc";

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string stem(std::string s) {
    for (char& c : s)
        if (c == '.') c = '_';
    return s;
}

VideoQcRequest request_of(const json& c) {
    VideoQcRequest r;
    r.audio_mode = c["audio"].get<std::string>();
    r.format = c["format"].get<std::string>();
    r.alpha = c["alpha"].get<bool>();
    const auto& e = c["expected_hdr"];
    r.expected = ExpectedLight{e["max_cll"].get<double>(), e["max_fall"].get<double>(), e["peak"].get<double>(),
                               e["minimum"].get<double>()};
    return r;
}

VideoClock clock_of(const json& c) {
    return VideoClock{c["fps"].get<std::string>(), c["frames"].get<int>(), c["start"].get<double>(),
                      c["duration"].get<double>(), c["tolerance"].get<double>()};
}

void expect_verdict(const json& c, const Result<pyjson::Value>& got, bool compare_record) {
    if (c.contains("error")) {
        ASSERT_FALSE(got.ok()) << c["name"];
        EXPECT_EQ(got.error().message, c["error"].get<std::string>());
        return;
    }
    ASSERT_TRUE(got.ok()) << got.error().message;
    if (compare_record) {
        EXPECT_EQ(pyjson::dumps(*got, 2), c["ok_json"].get<std::string>());
    }
}
}  // namespace

TEST(VideoQc, VerdictsAreThePythonsFromRecordedProbes) {
    const json idx = json::parse(slurp(kQc / "index.json"));
    ASSERT_EQ(idx["checks"].size(), 14u);
    for (const auto& c : idx["checks"]) {
        SCOPED_TRACE(c["name"].get<std::string>());
        const std::string src = slurp(kGolden / "video" / "probes" / (stem(c["clip"].get<std::string>()) + ".streams.json"));
        const std::string m = stem(c["master"].get<std::string>());
        auto findings = evaluate_video_qc(src, clock_of(c["clock"]), slurp(kQc / "probes" / (m + ".streams.json")),
                                          slurp(kQc / "probes" / (m + ".frames.json")), request_of(c));
        ASSERT_TRUE(findings) << findings.error().message;
        expect_verdict(c, qc_verdict(*findings), true);
    }
}

TEST(VideoQc, LiveQualityCheck) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    const json idx = json::parse(slurp(kQc / "index.json"));
    auto version = run_process({find_executable("ffmpeg")->string(), "-version"});
    ASSERT_TRUE(version);
    const bool same_build = version->out.substr(0, version->out.find_first_of("\r\n")) == idx["ffmpeg"].get<std::string>();
    for (const auto& c : idx["checks"]) {
        SCOPED_TRACE(c["name"].get<std::string>());
        const std::string src = slurp(kGolden / "video" / "probes" / (stem(c["clip"].get<std::string>()) + ".streams.json"));
        expect_verdict(c, run_quality_check(src, clock_of(c["clock"]), kQc / "masters" / c["master"].get<std::string>(),
                                            request_of(c)),
                       same_build);
    }
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(VideoQc, AlphaCheckIsThePythons) {
    if (!find_executable("ffmpeg")) GTEST_SKIP() << "ffmpeg is not on PATH";
    const json idx = json::parse(slurp(kQc / "index.json"));
    const json* conv = nullptr;
    for (const auto& c : idx["converted"])
        if (c["name"] == "prores4444_alpha") conv = &c;
    ASSERT_NE(conv, nullptr);
    // The spool convert_video encoded from: the same frames the step 4 golden kept.
    const fs::path spool = fs::temp_directory_path() / "rudra-alpha-check-spool";
    fs::remove_all(spool);
    fs::create_directories(spool);
    for (int i = 0; i < 6; ++i) {
        char n[32];
        std::snprintf(n, sizeof n, "%08d.png", i);
        fs::copy_file(kGolden / "video_master" / "spool" / (std::string("prores4444_alpha_") + n), spool / n);
    }
    auto got = run_alpha_check(kQc / "masters" / "prores4444_alpha.mov", spool, 6, 64, 36);
    ASSERT_TRUE(got) << got.error().message;
    EXPECT_EQ(pyjson::dumps(*got, 2), (*conv)["report_alpha_json"].get<std::string>());
    EXPECT_TRUE(fs::exists(spool / "alpha_qc.log"));
    fs::remove_all(spool);
}
#endif

TEST(VideoQc, AlphaVerdictAndDecodeCommand) {
    EXPECT_TRUE(alpha_verdict(128).ok());
    EXPECT_EQ(alpha_verdict(129).error().message, "Alpha QC failed: maximum 16-bit code error 129 exceeds 128");
    EXPECT_EQ(json(qc_decode_command("ffmpeg", "m.mp4")),
              json::parse(R"(["ffmpeg","-v","error","-xerror","-nostdin","-i","m.mp4","-map","0:v:0","-map","0:a?","-f","null","-"])"));
}
