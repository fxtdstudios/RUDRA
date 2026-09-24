#include "rudra/video/ffmpeg_check.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>

#include <nlohmann/json.hpp>

#include "rudra/deliver/video_encode.hpp"
#include "rudra/deliver/video_master.hpp"
#include "rudra/deliver/video_qc.hpp"
#include "rudra/platform/hash.hpp"
#include "rudra/platform/process.hpp"
#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/png16.hpp"
#include "rudra/media/video_decode.hpp"
#include "rudra/media/video_probe.hpp"
#include "rudra/video/qc.hpp"
#endif

namespace rudra {
namespace fs = std::filesystem;

std::vector<std::string> FfmpegCapabilities::missing() const {
    std::vector<std::string> m;
    if (!zscale) m.push_back("the zscale filter (zimg), which every video delivery converts colour with");
    if (!libx265) m.push_back("the libx265 encoder, for HDR10 and HLG");
    if (!prores_ks) m.push_back("the prores_ks encoder, for ProRes");
    if (!alphaextract) m.push_back("the alphaextract filter, which checks ProRes 4444 alpha");
    return m;
}

pyjson::Value FfmpegCapabilities::to_json() const {
    pyjson::List miss;
    for (const auto& s : missing()) miss.push_back(s);
    return pyjson::Dict{{"ffmpeg", ffmpeg.string()},
                        {"ffprobe", ffprobe.string()},
                        {"version", version},
                        {"ffmpeg_sha256", ffmpeg_sha256},
                        {"ffprobe_sha256", ffprobe_sha256},
                        {"libx265", libx265},
                        {"prores_ks", prores_ks},
                        {"zscale", zscale},
                        {"alphaextract", alphaextract},
                        {"write_colr", write_colr},
                        {"prores_metadata", prores_metadata},
                        {"missing", miss}};
}

namespace {
// A whole word in ffmpeg's -encoders / -filters table (" libx265 ", not "libx265rgb").
bool listed(const std::string& table, const std::string& name) {
    std::size_t at = 0;
    while ((at = table.find(name, at)) != std::string::npos) {
        const bool before = at == 0 || table[at - 1] == ' ';
        const std::size_t end = at + name.size();
        const bool after = end >= table.size() || table[end] == ' ' || table[end] == '\n' || table[end] == '\r';
        if (before && after) return true;
        at = end;
    }
    return false;
}
}  // namespace

Result<FfmpegCapabilities> probe_ffmpeg(const std::optional<fs::path>& ffmpeg, const std::optional<fs::path>& ffprobe) {
    FfmpegCapabilities c;
    if (ffmpeg) c.ffmpeg = *ffmpeg;
    else if (auto f = require_executable("ffmpeg")) c.ffmpeg = *f;
    else return f.error();
    if (ffprobe) c.ffprobe = *ffprobe;
    else if (auto f = require_executable("ffprobe")) c.ffprobe = *f;
    else return f.error();
    const std::string exe = c.ffmpeg.string();
    auto version = run_tool({exe, "-version"});
    if (!version) return version.error();
    c.version = version->substr(0, version->find_first_of("\r\n"));
    auto encoders = run_tool({exe, "-hide_banner", "-encoders"});
    if (!encoders) return encoders.error();
    auto filters = run_tool({exe, "-hide_banner", "-filters"});
    if (!filters) return filters.error();
    c.libx265 = listed(*encoders, "libx265");
    c.prores_ks = listed(*encoders, "prores_ks");
    c.zscale = listed(*filters, "zscale");
    c.alphaextract = listed(*filters, "alphaextract");
    auto mov = run_process({exe, "-hide_banner", "-h", "muxer=mov"});
    c.write_colr = mov && (mov->out + mov->err).find("write_colr") != std::string::npos;
    auto bsf = run_process({exe, "-hide_banner", "-h", "bsf=prores_metadata"});
    c.prores_metadata = bsf && (bsf->out + bsf->err).find("color_primaries") != std::string::npos;
    auto a = sha256_file(c.ffmpeg);
    if (!a) return a.error();
    auto b = sha256_file(c.ffprobe);
    if (!b) return b.error();
    c.ffmpeg_sha256 = *a, c.ffprobe_sha256 = *b;
    return c;
}

fs::path default_cache_dir() {
    if (const char* d = std::getenv("RUDRA_CACHE_DIR"); d && *d) return fs::path(d);
#ifdef _WIN32
    if (const char* d = std::getenv("LOCALAPPDATA"); d && *d) return fs::path(d) / "RUDRA";
#elif defined(__APPLE__)
    if (const char* h = std::getenv("HOME"); h && *h) return fs::path(h) / "Library" / "Caches" / "RUDRA";
#else
    if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) return fs::path(x) / "rudra";
    if (const char* h = std::getenv("HOME"); h && *h) return fs::path(h) / ".cache" / "rudra";
#endif
    return fs::temp_directory_path() / "rudra-cache";
}

#ifdef RUDRA_HAVE_STILL_DECODE

namespace {
fs::path verdict_path(const FfmpegCapabilities& c, const fs::path& dir) {
    return dir / ("ffmpeg-selftest-" + c.ffmpeg_sha256.substr(0, 16) + "-" + c.ffprobe_sha256.substr(0, 16) + ".json");
}

Result<void> run_self_test(const FfmpegCapabilities& caps, const fs::path& work) {
    const std::string ffmpeg = caps.ffmpeg.string();
    const fs::path source = work / "source.mkv";
    // A tagged SDR clip every build can make: lavfi into FFV1.
    auto made = run_tool({ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                          "testsrc2=s=128x72:r=24", "-frames:v", "16", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
                          "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range",
                          "tv", source.string()});
    if (!made) return made.error();
    auto probe = probe_video(source);
    if (!probe) return probe.error();
    auto video = open_video_source(*probe, VideoArgs{});
    if (!video) return video.error();
    fs::create_directories(work / "spool");
    auto dec = VideoDecoder::open_command(decoder_command(ffmpeg, source, video->contract, false), video->stream.width,
                                          video->stream.height, 3, video->clock.frames, work / "decode.log");
    if (!dec) return dec.error();
    std::vector<VideoFrameLight> lights;
    for (int i = 0; i < video->clock.frames; ++i) {
        auto f = (*dec)->next();
        if (!f) return f.error();
        SdrImage sdr = rgb_from_frame(*f);
        PlanarBuffer hdr(3, sdr.height(), sdr.width());
        const auto in = sdr.buffer().span();
        auto out = hdr.span();
        for (std::size_t k = 0; k < in.size(); ++k) out[k] = std::pow(in[k], 2.2f) * 0.08f;   // up to 800 nits
        auto sf = master_video_frame(NetworkLinearImage(std::move(hdr)), i, "hdr10", 1000.0, std::nullopt);
        if (!sf) return sf.error();
        if (auto w = write_png16(work / "spool" / spool_frame_name(i), sf->packed, sf->width, sf->height, 3); !w)
            return w.error();
        lights.push_back(sf->light);
    }
    if (auto d = (*dec)->finish(); !d) return d.error();
    const auto ceil = light_ceilings(lights);
    VideoEncodeRequest req;
    req.preset = "ultrafast";
    const fs::path master = work / "selftest.mp4";
    auto cmd = encode_command(ffmpeg, req, video->clock, work / "spool", source, master, ceil.max_cll, ceil.max_fall);
    if (!cmd) return cmd.error();
    if (auto r = run_tool(*cmd); !r) return r.error();
    VideoQcRequest qc;
    qc.expected = ExpectedLight{double(ceil.max_cll), double(ceil.max_fall), 1000.0, 0.005};
    auto verdict = run_quality_check(video->streams_json, video->clock, master, qc);
    if (!verdict) return verdict.error();
    return {};
}
}  // namespace

Result<FfmpegSelfTest> ffmpeg_self_test(const FfmpegCapabilities& caps, const fs::path& cache_dir, bool force) {
    FfmpegSelfTest t;
    const fs::path vpath = verdict_path(caps, cache_dir);
    std::error_code ec;
    if (!force && fs::exists(vpath, ec)) {
        std::ifstream in(vpath, std::ios::binary);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        try {
            const auto j = nlohmann::json::parse(text);
            if (j.value("passed", false) && j.value("ffmpeg_sha256", "") == caps.ffmpeg_sha256 &&
                j.value("ffprobe_sha256", "") == caps.ffprobe_sha256 && j.value("version", "") == caps.version) {
                // The version line too: a packaged ffmpeg is a small binary over shared
                // libraries that can change under it, and the package version moves with them.
                t.passed = t.cached = true;
                t.seconds = j.value("seconds", 0.0);
                return t;
            }
        } catch (const nlohmann::json::exception&) {
        }
    }
    if (!caps.libx265 || !caps.zscale) {
        t.detail = "This FFmpeg build needs libx265 and zscale support";
        return t;
    }
    const auto started = std::chrono::steady_clock::now();
    auto work = StagingDir::create(fs::temp_directory_path(), "rudra-ffmpeg-selftest-");
    if (!work) return work.error();
    auto r = run_self_test(caps, work->path());
    t.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    t.passed = r.ok();
    if (!r) t.detail = r.error().message;
    if (t.passed) {
        fs::create_directories(cache_dir, ec);
        std::ofstream out(vpath, std::ios::binary);
        out << pyjson::dumps(pyjson::Dict{{"ffmpeg_sha256", caps.ffmpeg_sha256},
                                          {"ffprobe_sha256", caps.ffprobe_sha256},
                                          {"version", caps.version},
                                          {"passed", true},
                                          {"seconds", t.seconds}},
                             2);
    }
    return t;
}

#endif

}  // namespace rudra
