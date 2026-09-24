#include "video_queues.hpp"

#include <fstream>
#include <iterator>

#include <nlohmann/json.hpp>

#include "rudra/platform/pyjson.hpp"

namespace rudra::app {
namespace fs = std::filesystem;

QueueJobState read_queue_state(const fs::path& queue) {
    QueueJobState s;
    fs::path state = queue;
    state += ".state.json";
    std::ifstream in(state, std::ios::binary);
    if (!in) return s;
    try {
        const auto j = nlohmann::json::parse(std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()));
        if (!j.contains("jobs") || j["jobs"].empty()) return s;
        const auto& job = j["jobs"][0];
        s.status = job.value("status", "pending");
        s.error = job.value("error", "");
        if (job.contains("progress") && job["progress"].is_object()) {
            s.phase = job["progress"].value("phase", "");
            s.frames_done = job["progress"].value("frames_done", 0);
            s.frames_total = job["progress"].value("frames_total", 0);
        }
    } catch (const nlohmann::json::exception&) {
        // a state file being replaced: read it on the next tick
    }
    return s;
}

VideoQueues::~VideoQueues() {
    {
        std::lock_guard lk(mu_);
        pending_.clear();
    }
    cancel_ = true;
    if (worker_.joinable()) worker_.join();
}

void VideoQueues::run(const QueueEntry& entry, const VideoQueueOptions& options, bool retry_failed) {
    {
        std::lock_guard lk(mu_);
        bool known = false;
        for (auto& e : entries_)
            if (e.queue == entry.queue) known = true;
        if (!known) entries_.push_back(entry);
        for (const auto& p : pending_)
            if (p.queue == entry.queue) return;
        if (current_ == entry.queue) return;
        pending_.push_back({entry.queue, options, retry_failed});
    }
    pump();
}

void VideoQueues::pump() {
    if (running_.exchange(true)) return;   // the worker takes what was added
    if (worker_.joinable()) worker_.join();
    cancel_ = false;
    worker_ = std::thread([this] {
        for (;;) {
            Pending next;
            {
                std::lock_guard lk(mu_);
                if (pending_.empty() || cancel_) {
                    current_.reset();
                    running_ = false;
                    return;
                }
                next = pending_.front();
                pending_.pop_front();
                current_ = next.queue;
            }
            next.options.cancel = &cancel_;
            Result<int> r = make_error(ErrorCode::Unsupported, "This build has no video pipeline (OpenCV is off)");
#ifdef RUDRA_HAVE_STILL_DECODE
            r = run_video_queue(next.queue, next.retry, next.options);
#endif
            if (finished) finished(next.queue, r);
        }
    });
}

void VideoQueues::stop() { cancel_ = true; }

std::optional<fs::path> VideoQueues::current() const {
    std::lock_guard lk(mu_);
    return current_;
}

bool VideoQueues::waiting(const fs::path& queue) const {
    std::lock_guard lk(mu_);
    for (const auto& p : pending_)
        if (p.queue == queue) return true;
    return false;
}

std::vector<QueueEntry> VideoQueues::entries() const {
    std::lock_guard lk(mu_);
    return entries_;
}

void VideoQueues::forget(const fs::path& queue) {
    std::lock_guard lk(mu_);
    std::erase_if(entries_, [&](const QueueEntry& e) { return e.queue == queue && current_ != queue; });
    std::erase_if(pending_, [&](const Pending& p) { return p.queue == queue; });
}

void VideoQueues::set_entries(std::vector<QueueEntry> e) {
    std::lock_guard lk(mu_);
    entries_ = std::move(e);
}

Result<fs::path> write_export_queue(const fs::path& movie, const fs::path& output, const fs::path& manifest,
                                    const std::string& format) {
    const fs::path dir = output.parent_path();
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string base = output.stem().string() + "." + format;
    fs::path queue = dir / (base + ".queue.json");
    for (int n = 2; fs::exists(queue, ec); ++n) queue = dir / (base + " " + std::to_string(n) + ".queue.json");
    const pyjson::Value spec = pyjson::Dict{
        {"version", 1},
        {"defaults", pyjson::Dict{{"checkpoint", manifest.string()}}},
        {"jobs", pyjson::List{pyjson::Dict{{"input", movie.string()},
                                           {"output", output.string()},
                                           {"options", pyjson::Dict{{"format", format}}}}}}};
    std::ofstream out(queue, std::ios::binary);
    out << pyjson::dumps(spec, 2) << "\n";
    if (!out) return make_error(ErrorCode::IoError, "Could not write the queue", queue.string());
    return queue;
}

}  // namespace rudra::app
