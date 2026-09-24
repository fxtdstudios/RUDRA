#pragma once
// ffmpeg on this machine (Phase 4, step 10): what the build can do, asked
// before any work, and a 16-frame HDR10 self-test through the real pipeline
// (a tagged SDR clip decoded through zscale, mastered, spooled, encoded with
// libx265 and its HDR10 SEI, and QC'd), cached by the SHA-256 of the ffmpeg
// and ffprobe binaries so it runs once per build.

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct FfmpegCapabilities {
    std::filesystem::path ffmpeg, ffprobe;
    std::string version;                       // ffmpeg -version, first line
    std::string ffmpeg_sha256, ffprobe_sha256;
    bool libx265 = false, prores_ks = false, zscale = false, alphaextract = false;
    bool write_colr = false, prores_metadata = false;
    // What a delivery needs and this build lacks, in words for the user.
    std::vector<std::string> missing() const;
    pyjson::Value to_json() const;
};

// Finds ffmpeg and ffprobe (on PATH unless given) and asks them.
Result<FfmpegCapabilities> probe_ffmpeg(const std::optional<std::filesystem::path>& ffmpeg = std::nullopt,
                                        const std::optional<std::filesystem::path>& ffprobe = std::nullopt);

struct FfmpegSelfTest {
    bool passed = false;
    bool cached = false;        // a pass recorded earlier for these exact binaries
    std::string detail;         // the failure, in the pipeline's own words
    double seconds = 0;
};

// Where the self-test's verdicts are kept: $RUDRA_CACHE_DIR, else
// %LOCALAPPDATA%\RUDRA (Windows), ~/Library/Caches/RUDRA (macOS) or
// $XDG_CACHE_HOME/rudra (~/.cache/rudra).
std::filesystem::path default_cache_dir();

#ifdef RUDRA_HAVE_STILL_DECODE
// The self-test. A cached pass is returned without running when `force` is false.
Result<FfmpegSelfTest> ffmpeg_self_test(const FfmpegCapabilities& caps, const std::filesystem::path& cache_dir,
                                        bool force = false);
#endif

}  // namespace rudra
