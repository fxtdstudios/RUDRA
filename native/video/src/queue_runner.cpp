#include "rudra/video/queue_runner.hpp"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>

#include "rudra/platform/hash.hpp"

namespace rudra {
namespace fs = std::filesystem;

Result<ModelManifest> package_for_checkpoint(const fs::path& checkpoint, const std::optional<fs::path>& explicit_package,
                                             const std::vector<fs::path>& roots) {
    // The app's own queues name the package itself (its manifest.json): the
    // app has the package, not always the .pt it was exported from.
    if (checkpoint.filename() == "manifest.json" && !explicit_package) return read_manifest(checkpoint.parent_path());
    auto sha = sha256_file(checkpoint);
    if (!sha) return make_error(ErrorCode::NotFound, "[Errno 2] No such file or directory: '" + checkpoint.string() + "'");
    if (explicit_package) {
        auto m = read_manifest(*explicit_package);
        if (!m) return m.error();
        if (m->source_sha256 != *sha)
            return make_error(ErrorCode::IntegrityError, "The model package was not exported from this job's checkpoint",
                              explicit_package->string() + " holds " + m->source_sha256 + ", " + checkpoint.string() +
                                  " is " + *sha);
        return m;
    }
    std::error_code ec;
    for (const auto& root : roots) {
        std::vector<fs::path> candidates{root};
        if (fs::is_directory(root, ec))
            for (const auto& e : fs::directory_iterator(root, ec))
                if (e.is_directory(ec)) candidates.push_back(e.path());
        std::sort(candidates.begin() + 1, candidates.end());
        for (const auto& c : candidates) {
            if (!fs::exists(c / "manifest.json", ec)) continue;
            auto m = read_manifest(c);
            if (m && m->source_sha256 == *sha) return m;
        }
    }
    return make_error(ErrorCode::NotFound, "No model package exported from this job's checkpoint was found",
                      checkpoint.filename().string() + " (sha256 " + *sha + "); export one with tools/export_model.py "
                      "or pass --package");
}

Result<VideoConvertArgs> convert_args_for(const QueueJob& job) {
    VideoConvertArgs a;
    a.input = job.input;
    a.output = job.output;
    a.work_dir = job.work_dir;
    for (const auto& [k, v] : job.options) {
        char* end = nullptr;
        if (k == "output" || k == "checkpoint" || k == "work-dir" || k == "device") continue;
        if (k == "format") a.format = v;
        else if (k == "alpha-mode") a.alpha_mode = v;
        else if (k == "input-transfer") a.input_transfer = v;
        else if (k == "input-primaries") a.input_primaries = v;
        else if (k == "input-matrix") a.input_matrix = v;
        else if (k == "input-range") a.input_range = v;
        else if (k == "audio") a.audio = v;
        else if (k == "peak-nits") a.peak_nits = std::strtod(v.c_str(), &end);
        else if (k == "min-nits") a.min_nits = std::strtod(v.c_str(), &end);
        else if (k == "knee-nits") a.knee_nits = std::strtod(v.c_str(), &end);
        else if (k == "crf") a.crf = std::atoi(v.c_str());
        else if (k == "preset") a.preset = v;
        else if (k == "tile-size") a.tile_size = std::atoi(v.c_str());
        else if (k == "tile-overlap") a.tile_overlap = std::atoi(v.c_str());
        else if (k == "shadow-smoothing") a.shadow_smoothing = std::strtod(v.c_str(), &end);
        else if (k == "cut-threshold") a.cut_threshold = std::strtod(v.c_str(), &end);
        else return make_error(ErrorCode::InvalidArgument, "Invalid video options in queue", k);
    }
    return a;
}

#ifdef RUDRA_HAVE_STILL_DECODE
namespace {
Device device_of(const std::string& torch_device) {
    if (torch_device.rfind("cuda", 0) == 0) return Device::Cuda;
    if (torch_device == "mps") return Device::Mps;
    return Device::Cpu;
}
}  // namespace
#endif

QueueRunner make_video_queue_runner(const VideoQueueOptions& o) {
#ifdef RUDRA_HAVE_STILL_DECODE
    struct Loaded {
        ModelManifest manifest;
        std::unique_ptr<InferenceBackend> backend;
    };
    auto cache = std::make_shared<std::map<std::string, std::shared_ptr<Loaded>>>();
    return [o, cache](const QueueJob& job, const QueueProgress& progress) -> Result<void> {
        auto args = convert_args_for(job);
        if (!args) return args.error();
        std::string job_device = "cpu";
        for (const auto& [k, v] : job.options)
            if (k == "device") job_device = v;
        const Device device = o.device ? *o.device : device_of(job_device);
        const std::string key = job.checkpoint.string() + "|" + to_string(device);
        std::shared_ptr<Loaded> loaded;
        if (auto it = cache->find(key); it != cache->end()) {
            loaded = it->second;
        } else {
            auto m = package_for_checkpoint(job.checkpoint, o.package, o.package_roots);
            if (!m) return m.error();
            Result<std::unique_ptr<InferenceBackend>> b = make_error(ErrorCode::Unsupported, "No inference runtime in this build");
            if (o.runtime == "libtorch") b = make_libtorch_backend(*m, device);
            else if (o.runtime == "onnxruntime") b = make_onnxruntime_backend(*m, device);
            else {
                for (auto r : compiled_runtimes())
                    if (r == Runtime::LibTorch) b = make_libtorch_backend(*m, device);
                if (!b) b = make_onnxruntime_backend(*m, device);
            }
            if (!b) return b.error();
            loaded = std::make_shared<Loaded>(Loaded{std::move(*m), std::move(*b)});
            (*cache)[key] = loaded;
        }
        args->package = loaded->manifest.root;
        VideoConvertHooks hooks;
        hooks.cancel = o.cancel;
        hooks.print = o.print;
        hooks.progress = [&](const VideoProgress& p) {
            progress(pyjson::Dict{{"phase", p.phase}, {"frames_done", p.frames_done}, {"frames_total", p.frames_total}});
        };
        auto r = convert_video(*args, loaded->manifest, *loaded->backend, hooks);
        if (!r) return r.error();
        return {};
    };
#else
    (void)o;
    return [](const QueueJob&, const QueueProgress&) -> Result<void> {
        return make_error(ErrorCode::Unsupported, "This build has no video pipeline (OpenCV is off)");
    };
#endif
}

#ifdef RUDRA_HAVE_STILL_DECODE
Result<int> run_video_queue(const fs::path& queue, bool retry_failed, const VideoQueueOptions& o) {
    return run_queue(queue, retry_failed, make_video_queue_runner(o), o.print);
}
#endif

}  // namespace rudra
