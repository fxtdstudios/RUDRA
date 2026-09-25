#pragma once
// engine: the frame path (NATIVE_ARCHITECTURE.md 5.2, 5.3; Phase 2 step 10).
//
// One worker thread owns inference: it is the InferActor, and the backend
// session is only ever used from it. The UI asks for a frame with show(); that
// bumps the generation, cancels every queued request of older generations,
// and queues the frame (interactive) and the ones after it (read-ahead). A
// result is delivered only if its generation is still current when it is
// ready, so a late result can never land on the wrong frame; it is cached
// either way, so going back is free. Frames already in the cache are
// delivered at once, from the caller's thread.
//
// Qt-free: the loader and the inference are functions, so the same engine
// drives the app (media + infer), the tests (fakes) and, later, the CLI.

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include "rudra/core/image.hpp"
#include "rudra/engine/scheduling.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

using FrameLoader = std::function<Result<SdrImage>(int index)>;
using FrameInfer = std::function<Result<FrameResult>(const SdrImage& frame)>;

struct EngineOptions {
    int cache_frames = 32;   // frames held (decoded SDR and fields), as the Studio's CACHE_FRAMES
    int read_ahead = 12;     // frames queued after the one shown (PREFETCH_AHEAD)
};

// A frame ready to show: immutable, shared with the cache.
struct ReadyFrame {
    int index = -1;
    Generation::value_type generation = 0;
    std::shared_ptr<const SdrImage> sdr;
    std::shared_ptr<const FrameResult> fields;
    double decode_ms = 0.0, infer_ms = 0.0;
    bool from_cache = false;
    std::optional<Error> error;   // decode or inference failed for this frame
};

struct EngineStats {
    std::uint64_t shown = 0, cache_hits = 0, inferred = 0, stale_dropped = 0, cancelled = 0, errors = 0;
    double last_request_to_ready_ms = 0.0;   // show() to delivery of the frame it asked for
    int cached = 0;
};

class FrameEngine {
public:
    FrameEngine(FrameLoader loader, FrameInfer infer, EngineOptions options = {});
    ~FrameEngine();
    FrameEngine(const FrameEngine&) = delete;
    FrameEngine& operator=(const FrameEngine&) = delete;

    // A new source of `count` frames: the cache and every queued request go.
    void set_sequence(int count);
    int count() const;

    // Ask for frame `index` (clamped). Returns the new generation.
    Generation::value_type show(int index);

    // Called with the frame show() asked for, once, if it is still current:
    // from the worker thread, or from show()'s caller on a cache hit.
    void on_ready(std::function<void(const ReadyFrame&)> cb);

    bool is_cached(int index) const;
    EngineStats stats() const;

private:
    struct Job {
        int index;
        Priority priority;
        Generation::value_type generation;
    };
    void run();
    std::shared_ptr<ReadyFrame> lookup(int index);   // under mu_: moves it to the front of the LRU
    void insert(std::shared_ptr<ReadyFrame> f);       // under mu_
    void deliver(const ReadyFrame& f);

    FrameLoader loader_;
    FrameInfer infer_;
    EngineOptions opt_;
    Generation generation_;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;   // under mu_: the destructor asks the worker to return
    std::deque<Job> queue_;
    int count_ = 0;
    int wanted_ = -1;   // the frame the current generation asked for
    std::chrono::steady_clock::time_point asked_at_;
    std::list<int> lru_;   // most recent first
    std::unordered_map<int, std::pair<std::shared_ptr<ReadyFrame>, std::list<int>::iterator>> cache_;
    std::function<void(const ReadyFrame&)> ready_cb_;
    EngineStats stats_;
    // A std::thread and stop_, not std::jthread: Apple's libc++ in Xcode 15 has
    // no jthread or stop_token. Last, so it starts after everything it reads.
    std::thread worker_;
};

}  // namespace rudra
