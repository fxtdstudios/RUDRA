#pragma once
// rudra/video.py convert_video in C++ (Phase 4, step 7): an SDR clip to an
// HDR10, HLG or ProRes master with its audio, checked before it is published
// and never replacing anything. The same checks in the same order with the
// same words, the same decode, predictor, mastering, spool, encoder command,
// QC and sidecar report, key for key.

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// video.py's arguments, with argparse's defaults. `package` stands where the
// Python takes --checkpoint: the model package exported from that checkpoint.
struct VideoConvertArgs {
    std::filesystem::path input, output, package;
    std::string format = "hdr10";
    std::optional<std::string> alpha_mode;
    std::string input_transfer = "auto", input_primaries = "auto", input_matrix = "auto", input_range = "auto";
    std::string audio = "copy";
    double peak_nits = 1000.0, min_nits = 0.005;
    std::optional<double> knee_nits;
    int crf = 12;
    std::string preset = "medium";
    int tile_size = 512, tile_overlap = 64;
    double shadow_smoothing = 0.0, cut_threshold = 0.15;
    std::optional<std::filesystem::path> work_dir;
};

// progress(dict(phase=..., frames_done=..., frames_total=...)).
struct VideoProgress {
    std::string phase;   // inference, encoding, quality_check
    int frames_done = 0, frames_total = 0;
};

struct VideoConvertHooks {
    std::function<void(const VideoProgress&)> progress;
    std::function<void(const std::string&)> print;   // the lines the Python prints
    const std::atomic<bool>* cancel = nullptr;        // checked once per frame
};

struct VideoConvertResult {
    std::filesystem::path output, sidecar;
    pyjson::Value report;
};

// sidecar = output + ".json" (output.with_suffix(output.suffix + '.json')).
std::filesystem::path video_sidecar_path(const std::filesystem::path& output);

#ifdef RUDRA_HAVE_STILL_DECODE
Result<VideoConvertResult> convert_video(const VideoConvertArgs& args, const ModelManifest& package,
                                         InferenceBackend& backend, const VideoConvertHooks& hooks = {});
#endif

}  // namespace rudra
