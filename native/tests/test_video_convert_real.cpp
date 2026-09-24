// Phase 4 step 7: convert_video end to end on the exported package against
// rudra/video.py convert_video on the checkpoint (tools/emit_video_convert_golden.py):
// HDR10 with audio, HLG smoothed and tiled, ProRes 4444 with alpha. The report
// key for key in the same order with the same values (paths, the spool and
// staging folders and the elapsed time aside; the per-frame weights and light
// within the package's tolerance), QC passed, and the master byte for byte the
// Python's when the ffmpeg is the recording one. Built with RUDRA_TEST_PACKAGE.
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include "rudra/platform/hash.hpp"
#include "rudra/platform/process.hpp"
#include "rudra/video/convert.hpp"

using namespace rudra;
using ojson = nlohmann::ordered_json;
namespace fs = std::filesystem;

namespace {
const fs::path kGolden = fs::path(RUDRA_GOLDEN_DIR);

std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Same keys in the same order, same values; `tol` for floats under "frames".
void compare(const ojson& got, const ojson& want, const std::string& path, double tol, int& floats_off) {
    if (want.is_object()) {
        ASSERT_TRUE(got.is_object()) << path;
        std::vector<std::string> gk, wk;
        for (auto it = got.begin(); it != got.end(); ++it) gk.push_back(it.key());
        for (auto it = want.begin(); it != want.end(); ++it) wk.push_back(it.key());
        EXPECT_EQ(gk, wk) << path;
        for (auto it = want.begin(); it != want.end(); ++it)
            if (got.contains(it.key())) compare(got[it.key()], it.value(), path + "." + it.key(), tol, floats_off);
        return;
    }
    if (want.is_array()) {
        ASSERT_TRUE(got.is_array()) << path;
        ASSERT_EQ(got.size(), want.size()) << path;
        for (std::size_t i = 0; i < want.size(); ++i) compare(got[i], want[i], path + "[" + std::to_string(i) + "]", tol, floats_off);
        return;
    }
    if (want.is_number_float() && path.rfind(".frames[", 0) == 0) {
        const double w = want.get<double>(), g = got.get<double>();
        EXPECT_LE(std::fabs(g - w), tol * std::max(1.0, std::fabs(w))) << path;
        floats_off += g != w;
        return;
    }
    EXPECT_EQ(got, want) << path;
}
}  // namespace

TEST(VideoConvertReal, ReportAndMasterAreThePythons) {
    if (!find_executable("ffmpeg") || !find_executable("ffprobe")) GTEST_SKIP() << "ffmpeg and ffprobe are not on PATH";
    auto m = read_manifest(RUDRA_TEST_PACKAGE_DIR);
    ASSERT_TRUE(m) << m.error().message;
    const ojson idx = ojson::parse(slurp(kGolden / "video_convert" / "index.json"));
    auto version = run_process({find_executable("ffmpeg")->string(), "-version"});
    ASSERT_TRUE(version);
    const bool same_ffmpeg = version->out.substr(0, version->out.find_first_of("\r\n")) == idx["ffmpeg"].get<std::string>();
    const fs::path work = fs::temp_directory_path() / "rudra-video-convert-test";

    for (const Runtime rt : compiled_runtimes()) {
        auto backend = rt == Runtime::LibTorch ? make_libtorch_backend(*m, Device::Cpu) : make_onnxruntime_backend(*m, Device::Cpu);
        ASSERT_TRUE(backend) << backend.error().message;
        const auto& tol = m->tolerance.at(rt == Runtime::LibTorch ? "torchscript" : "onnx");
        for (const auto& c : idx["cases"]) {
            SCOPED_TRACE(std::string(to_string(rt)) + " " + c["name"].get<std::string>());
            fs::remove_all(work);
            fs::create_directories(work);
            VideoConvertArgs a;
            a.package = RUDRA_TEST_PACKAGE_DIR;
            a.input = kGolden / "video" / "clips" / c["clip"].get<std::string>();
            a.output = work / c["output"].get<std::string>();
            const auto& flags = c["flags"];
            for (std::size_t i = 0; i + 1 < flags.size(); i += 2) {
                const std::string k = flags[i].get<std::string>(), v = flags[i + 1].get<std::string>();
                if (k == "--format") a.format = v;
                else if (k == "--alpha-mode") a.alpha_mode = v;
                else if (k == "--input-range") a.input_range = v;
                else if (k == "--audio") a.audio = v;
                else if (k == "--shadow-smoothing") a.shadow_smoothing = std::stod(v);
                else if (k == "--peak-nits") a.peak_nits = std::stod(v);
                else if (k == "--tile-size") a.tile_size = std::stoi(v);
                else if (k == "--tile-overlap") a.tile_overlap = std::stoi(v);
                else FAIL() << "flag not mapped: " << k;
            }
            std::vector<VideoProgress> progress;
            VideoConvertHooks hooks;
            hooks.progress = [&](const VideoProgress& p) { progress.push_back(p); };
            auto r = convert_video(a, *m, **backend, hooks);
            ASSERT_TRUE(r) << r.error().message;
            ASSERT_TRUE(fs::exists(r->output) && fs::exists(r->sidecar));
            ASSERT_GE(progress.size(), 3u);
            EXPECT_EQ(progress.back().phase, "quality_check");

            ojson got = ojson::parse(slurp(r->sidecar));
            ojson want = ojson::parse(slurp(kGolden / "video_convert" / c["report"].get<std::string>()));
            EXPECT_EQ(slurp(r->sidecar).back(), '}');   // json.dumps: no trailing newline
            EXPECT_EQ(fs::path(got["output"].get<std::string>()).filename(), fs::path(want["output"].get<std::string>()).filename());
            EXPECT_EQ(fs::path(got["source"].get<std::string>()).filename(), fs::path(want["source"].get<std::string>()).filename());
            EXPECT_GT(got["elapsed_seconds"].get<double>(), 0.0);
            for (auto* j : {&got, &want}) {
                (*j)["source"] = (*j)["output"] = (*j)["checkpoint"] = "";
                (*j)["elapsed_seconds"] = 0.0;
                auto& cmd = (*j)["encode_command"];
                cmd[0] = "ffmpeg";
                cmd[10] = fs::path(cmd[10].get<std::string>()).filename().string();          // the spool
                cmd[14] = fs::path(cmd[14].get<std::string>()).filename().string();          // the source
                cmd.back() = fs::path(cmd.back().get<std::string>()).filename().string();    // the staged master
            }
            const auto digest = sha256_file(r->output);
            const bool same_master = digest.ok() && *digest == c["master_sha256"].get<std::string>();
            if (!same_master) {   // another encode of near-identical codes: its stream facts may differ
                for (const char* k : {"bit_rate", "nb_frames", "extradata_size"}) {
                    got["qc"]["video"].erase(k);
                    want["qc"]["video"].erase(k);
                }
            }
            got["encode_command"].back() = want["encode_command"].back();   // the output's own name
            int floats_off = 0;
            compare(got, want, "", tol.atol + tol.rtol, floats_off);
            std::cout << "[ video ] " << to_string(rt) << " " << c["name"].get<std::string>() << ": master "
                      << (same_master ? "byte-identical" : "differs") << ", " << floats_off << " per-frame floats not bit-equal\n";
            // HLG's codes are float64 and exact, so its master is the Python's byte for byte.
            // PQ goes through numpy's float32 power (not correctly rounded, see step 4): the
            // codes may move by one in 65535, and the master is compared decoded instead.
            if (same_ffmpeg && rt == Runtime::LibTorch && a.format == "hlg") {
                EXPECT_TRUE(same_master);
            }
            if (!same_master) {
                auto decode = [](const fs::path& f) {
                    return run_process({find_executable("ffmpeg")->string(), "-v", "error", "-i", f.string(), "-map",
                                        "0:v:0", "-f", "rawvideo", "-pix_fmt", "yuva444p16le", "-"});
                };
                auto g = decode(r->output), w = decode(kGolden / "video_convert" / c["output"].get<std::string>());
                ASSERT_TRUE(g && w && g->exit_code == 0 && w->exit_code == 0);
                ASSERT_EQ(g->out.size(), w->out.size());
                int worst = 0;
                double sum = 0;
                const auto* gp = reinterpret_cast<const std::uint16_t*>(g->out.data());
                const auto* wp = reinterpret_cast<const std::uint16_t*>(w->out.data());
                const std::size_t n = g->out.size() / 2;
                for (std::size_t i = 0; i < n; ++i) {
                    const int d = std::abs(int(gp[i]) - int(wp[i]));
                    worst = std::max(worst, d);
                    sum += d;
                }
                std::cout << "[ video ]   decoded: max difference " << worst << " of 65535, mean " << sum / double(n) << "\n";
                // A lossy codec turns a one-in-65535 input change into a few output codes: at most
                // four 10-bit steps anywhere, well under one code on average.
                EXPECT_LE(worst, 256);
                EXPECT_LE(sum / double(n), 1.0);
            }
        }
    }
    fs::remove_all(work);
}

TEST(VideoConvertReal, RefusalsComeBeforeAnyWork) {
    auto m = read_manifest(RUDRA_TEST_PACKAGE_DIR);
    ASSERT_TRUE(m);
    auto backend = make_onnxruntime_backend(*m, Device::Cpu);
    if (!backend) backend = make_libtorch_backend(*m, Device::Cpu);
    ASSERT_TRUE(backend);
    const fs::path work = fs::temp_directory_path() / "rudra-video-refusals";
    fs::remove_all(work);
    fs::create_directories(work);
    auto run = [&](VideoConvertArgs a) {
        a.package = RUDRA_TEST_PACKAGE_DIR;
        if (a.input.empty()) a.input = kGolden / "video" / "clips" / "h264_709.mp4";
        auto r = convert_video(a, *m, **backend);
        return r ? std::string("ok") : r.error().message;
    };
    VideoConvertArgs a;
    a.output = work / "x.avi";
    EXPECT_EQ(run(a), "Output must be MP4, MOV or MKV");
    a.output = work / "x.mp4";
    a.format = "prores422";
    EXPECT_EQ(run(a), "ProRes delivery requires a .mov output");
    a.format = "hdr10";
    { std::ofstream(work / "x.mp4.json") << "{}"; }
    EXPECT_EQ(run(a), "Refusing to overwrite source/output/sidecar");
    fs::remove(work / "x.mp4.json");
    if (find_executable("ffmpeg") && find_executable("ffprobe")) {
        a.shadow_smoothing = 1.0;
        EXPECT_EQ(run(a), "Invalid smoothing/cut threshold");
        a.shadow_smoothing = 0;
        a.tile_overlap = 512;
        EXPECT_EQ(run(a), "Invalid tile size/overlap");
        a.tile_overlap = 64;
        a.crf = 52;
        EXPECT_EQ(run(a), "Invalid mastering/encoding settings");
        a.crf = 12;
        a.min_nits = 2000;
        EXPECT_EQ(run(a), "Invalid mastering/encoding settings");   // min_nits must be below the peak
        a.min_nits = 0.005;
        a.peak_nits = 50;
        EXPECT_EQ(run(a), "peak_nits must be between 100 and 10000");
        a.peak_nits = 1000;
        a.knee_nits = 1000;
        EXPECT_EQ(run(a), "knee_nits must be >= 0 and below peak_nits");
        a.knee_nits.reset();
        a.input = kGolden / "video" / "clips" / "pq.mkv";
        EXPECT_EQ(run(a), "Input is already HDR; this command accepts SDR only");
        EXPECT_FALSE(fs::exists(work / "x.mp4"));
    }
    fs::remove_all(work);
}
