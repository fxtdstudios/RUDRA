#pragma once
// The encoder's command and publishing (Phase 4, step 5): rudra/video.py
// encode_command, argument for argument, for the five delivery profiles, and
// convert_video's publish: the master and its sidecar staged beside the
// output and renamed into place, never replacing anything.

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/video_clock.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// The video.py arguments the encoder reads (argparse defaults).
struct VideoEncodeRequest {
    std::string format = "hdr10";   // hdr10, hlg, prores422, prores422hq, prores4444
    std::string preset = "medium";  // x265 preset
    int crf = 12;
    std::string audio = "copy";     // copy, aac, none
    double peak_nits = 1000.0;
    double min_nits = 0.005;
    bool alpha = false;             // args.preserve_alpha
};

// x265's master-display string, luminances in 0.0001-nit units (round half even).
std::string mastering_display(double peak_nits, double min_nits);

// encode_command: ffmpeg reading the spool (spool/%08d.png) at the clip's rate,
// the source for audio and metadata, into `output`.
Result<std::vector<std::string>> encode_command(const std::string& ffmpeg, const VideoEncodeRequest& request,
                                                const VideoClock& clock, const std::filesystem::path& spool,
                                                const std::filesystem::path& source,
                                                const std::filesystem::path& output, int max_cll, int max_fall);

// A staging folder on the output's volume (tempfile.TemporaryDirectory(prefix,
// dir=output.parent)), removed when this goes out of scope.
class StagingDir {
public:
    static Result<StagingDir> create(const std::filesystem::path& parent, const std::string& prefix);
    StagingDir(StagingDir&&) noexcept;
    StagingDir& operator=(StagingDir&&) noexcept;
    StagingDir(const StagingDir&) = delete;
    StagingDir& operator=(const StagingDir&) = delete;
    ~StagingDir();
    const std::filesystem::path& path() const noexcept { return path_; }

private:
    StagingDir() = default;
    std::filesystem::path path_;
};

// convert_video's last lines: "Output appeared during processing; refusing
// overwrite" when either target exists, else the two renames.
Result<void> publish_video(const std::filesystem::path& staged, const std::filesystem::path& staged_sidecar,
                           const std::filesystem::path& output, const std::filesystem::path& sidecar);

}  // namespace rudra
