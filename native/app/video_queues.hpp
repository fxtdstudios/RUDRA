#pragma once
// The app's video exports (Phase 4, step 11): each export from the sheet is a
// queue file beside its master, in rudra/batch.py's format, run by the video
// queue runner on a worker thread, one queue at a time. The state the runner
// writes (<queue>.state.json) is what the queue window shows, so a queue
// stopped here resumes here or from `rudra-native batch run`.

#include <atomic>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "rudra/video/queue_runner.hpp"

namespace rudra::app {

struct QueueEntry {
    std::filesystem::path queue;
    std::string title;     // the shot and the format, "clip.mp4 → HDR10"
};

// What a queue's state file says about its (single) job.
struct QueueJobState {
    std::string status = "pending";   // pending, running, complete, failed, interrupted, waiting
    std::string phase;                // starting, inference, encoding, quality_check, complete
    int frames_done = 0, frames_total = 0;
    std::string error;
};
QueueJobState read_queue_state(const std::filesystem::path& queue);

class VideoQueues {
public:
    ~VideoQueues();
    // Adds (or re-adds) a queue and runs it after any already waiting.
    void run(const QueueEntry& entry, const VideoQueueOptions& options, bool retry_failed = false);
    void stop();                                   // the running one, as Ctrl+C stops it
    bool running() const { return running_; }
    std::optional<std::filesystem::path> current() const;
    bool waiting(const std::filesystem::path& queue) const;
    std::vector<QueueEntry> entries() const;
    void forget(const std::filesystem::path& queue);   // off the list; the files stay
    void set_entries(std::vector<QueueEntry> e);        // restored from settings
    // Called on the worker thread when a queue ends (its result).
    std::function<void(const std::filesystem::path&, const Result<int>&)> finished;

private:
    void pump();
    struct Pending {
        std::filesystem::path queue;
        VideoQueueOptions options;
        bool retry = false;
    };
    mutable std::mutex mu_;
    std::vector<QueueEntry> entries_;
    std::deque<Pending> pending_;
    std::optional<std::filesystem::path> current_;
    std::atomic<bool> running_{false}, cancel_{false};
    std::thread worker_;
};

// The queue file for one export (batch.py's format), written beside the master.
Result<std::filesystem::path> write_export_queue(const std::filesystem::path& movie, const std::filesystem::path& output,
                                                 const std::filesystem::path& package_manifest, const std::string& format);

}  // namespace rudra::app
