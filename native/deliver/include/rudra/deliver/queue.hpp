#pragma once
// Durable sequential queues. Port of rudra/batch.py: the same queue.json
// (version 1, defaults, jobs), the same <queue>.state.json written the same
// way (atomic replace, key order and all), the same <queue>.lock, the same
// refusals. A queue started by `rudra batch run` resumes here and the other
// way round (ADR-008). What a job DOES is the runner's business: video
// export arrives with libav in Phase 4.

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct QueueJob {
    std::filesystem::path input, output, checkpoint;
    std::optional<std::filesystem::path> work_dir;
    // Every video option after defaults and the job's own, validated against
    // rudra/video.py's argument table, as the strings argparse would see.
    std::vector<std::pair<std::string, std::string>> options;
    std::filesystem::path sidecar() const;   // <output>.json, e.g. clip.mov.json
};

// load_jobs: parse, merge, validate, resolve paths against the queue's folder,
// refuse collisions between outputs, sources and the queue's own files.
Result<std::vector<QueueJob>> load_queue(const std::filesystem::path& queue);

using QueueProgress = std::function<void(pyjson::Value)>;
using QueueRunner = std::function<Result<void>(const QueueJob&, const QueueProgress&)>;
using QueueLog = std::function<void(const std::string&)>;

// run_queue: 0 when every job is complete, 1 otherwise.
Result<int> run_queue(const std::filesystem::path& queue, bool retry_failed, const QueueRunner& runner,
                      const QueueLog& log = {});

// show_status's text.
std::string queue_status(const std::filesystem::path& queue);

}  // namespace rudra
