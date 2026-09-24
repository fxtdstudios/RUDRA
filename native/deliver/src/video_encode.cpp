#include "rudra/deliver/video_encode.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <random>

#include "rudra/core/hdr10.hpp"
#include "rudra/platform/pyjson.hpp"

namespace rudra {
namespace fs = std::filesystem;

namespace {
const DeliveryProfile* find_profile(const std::string& name) {
    for (const auto& p : delivery_profiles())
        if (p.name == name) return &p;
    return nullptr;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    return s;
}

long long py_round(double v) { return static_cast<long long>(std::nearbyint(v)); }   // round half to even
}  // namespace

std::string mastering_display(double peak_nits, double min_nits) {
    return "G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L(" + std::to_string(py_round(peak_nits * 10000)) +
           "," + std::to_string(py_round(min_nits * 10000)) + ")";
}

Result<std::vector<std::string>> encode_command(const std::string& ffmpeg, const VideoEncodeRequest& r,
                                                const VideoClock& clock, const fs::path& spool, const fs::path& source,
                                                const fs::path& output, int max_cll, int max_fall) {
    const DeliveryProfile* profile = find_profile(r.format);
    if (!profile) return make_error(ErrorCode::InvalidArgument, "Unknown delivery format: " + r.format);
    const std::string& transfer = profile->transfer;
    std::string params = "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
                         "master-display=" + mastering_display(r.peak_nits, r.min_nits) +
                         ":max-cll=" + std::to_string(max_cll) + "," + std::to_string(max_fall);
    std::vector<std::string> cmd = {ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n", "-copyts",
                                    "-framerate", clock.fps, "-i", (spool / "%08d.png").string(),
                                    "-itsoffset", pyjson::repr(-clock.start), "-i", source.string(),
                                    "-map", "0:v:0"};
    if (r.audio != "none") cmd.insert(cmd.end(), {"-map", "1:a?", "-c:a", r.audio});
    if (r.audio == "aac") cmd.insert(cmd.end(), {"-b:a", "320k"});
    const std::string pix = r.alpha ? "yuva444p10le" : profile->pixel_format;
    cmd.insert(cmd.end(), {"-map_metadata", "1", "-map_chapters", "-1", "-vf",
                           "zscale=matrixin=gbr:transferin=" + transfer + ":primariesin=2020:rangein=full:"
                           "matrix=2020_ncl:transfer=" + transfer + ":primaries=2020:range=limited,format=" + pix,
                           "-c:v", profile->encoder});
    if (profile->codec == "hevc") {
        if (r.format == "hlg") params = "repeat-headers=1:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc";
        cmd.insert(cmd.end(), {"-preset", r.preset, "-crf", std::to_string(r.crf), "-x265-params", params,
                               "-tag:v", "hvc1"});
    } else {
        cmd.insert(cmd.end(), {"-profile:v", std::to_string(profile->prores_profile), "-tag:v", profile->tag,
                               "-alpha_bits", r.alpha ? "16" : "0"});
    }
    cmd.insert(cmd.end(), {"-color_primaries", "bt2020", "-color_trc", transfer, "-colorspace", "bt2020nc",
                           "-color_range", "tv", "-fps_mode", "passthrough", "-frames:v", std::to_string(clock.frames),
                           "-t", pyjson::repr(clock.duration), "-avoid_negative_ts", "disabled"});
    const std::string ext = lower(output.extension().string());
    if (ext == ".mp4" || ext == ".mov") cmd.insert(cmd.end(), {"-movflags", "+faststart+write_colr"});
    cmd.push_back(output.string());
    return cmd;
}

Result<StagingDir> StagingDir::create(const fs::path& parent, const std::string& prefix) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    const char* alphabet = "abcdefghijklmnopqrstuvwxyz0123456789_";
    for (int attempt = 0; attempt < 100; ++attempt) {
        std::string name = prefix;
        for (int i = 0; i < 8; ++i) name += alphabet[gen() % 37];
        std::error_code ec;
        if (fs::create_directory(parent / name, ec)) {
            StagingDir d;
            d.path_ = parent / name;
            return d;
        }
        if (ec) return make_error(ErrorCode::IoError, "Could not create a staging folder", (parent / name).string());
    }
    return make_error(ErrorCode::IoError, "Could not create a staging folder", parent.string());
}

StagingDir::StagingDir(StagingDir&& o) noexcept : path_(std::move(o.path_)) { o.path_.clear(); }

StagingDir& StagingDir::operator=(StagingDir&& o) noexcept {
    if (this != &o) {
        std::error_code ec;
        if (!path_.empty()) fs::remove_all(path_, ec);
        path_ = std::move(o.path_);
        o.path_.clear();
    }
    return *this;
}

StagingDir::~StagingDir() {
    std::error_code ec;
    if (!path_.empty()) fs::remove_all(path_, ec);
}

Result<void> publish_video(const fs::path& staged, const fs::path& staged_sidecar, const fs::path& output,
                           const fs::path& sidecar) {
    std::error_code ec;
    if (fs::exists(output, ec) || fs::exists(sidecar, ec))
        return make_error(ErrorCode::InvalidArgument, "Output appeared during processing; refusing overwrite");
    fs::rename(staged, output, ec);
    if (ec) return make_error(ErrorCode::IoError, "Could not publish the master", ec.message());
    fs::rename(staged_sidecar, sidecar, ec);
    if (ec) return make_error(ErrorCode::IoError, "Could not publish the sidecar", ec.message());
    return {};
}

}  // namespace rudra
