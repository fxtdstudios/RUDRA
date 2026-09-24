#pragma once
// The queue's video jobs (Phase 4, step 9): rudra/batch.py run_queue with
// convert_video as the runner. A job names its checkpoint as the Python does
// (the .pt file, whose digest is the job's identity), so a queue started by
// `rudra batch run` resumes here and the other way round; the model package
// that runs it is the one exported from that checkpoint, found by the
// manifest's source SHA-256.

#include <atomic>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "rudra/deliver/queue.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/video/convert.hpp"

namespace rudra {

// The package exported from `checkpoint`: `explicit_package` when given (and
// it must match), else the first under `roots` (each root and its immediate
// folders) whose manifest names the checkpoint's SHA-256.
Result<ModelManifest> package_for_checkpoint(const std::filesystem::path& checkpoint,
                                             const std::optional<std::filesystem::path>& explicit_package,
                                             const std::vector<std::filesystem::path>& roots);

// A queue job's options (the argparse names and strings load_queue kept) as
// convert_video's arguments.
Result<VideoConvertArgs> convert_args_for(const QueueJob& job);

struct VideoQueueOptions {
    std::optional<std::filesystem::path> package;       // --package
    std::vector<std::filesystem::path> package_roots;   // where else to look
    std::string runtime = "auto";                       // libtorch, onnxruntime, auto
    std::optional<Device> device;                       // overrides the job's --device
    const std::atomic<bool>* cancel = nullptr;
    std::function<void(const std::string&)> print;
};

QueueRunner make_video_queue_runner(const VideoQueueOptions& options);

#ifdef RUDRA_HAVE_STILL_DECODE
// run_queue with convert_video.
Result<int> run_video_queue(const std::filesystem::path& queue, bool retry_failed, const VideoQueueOptions& options);
#endif

}  // namespace rudra
