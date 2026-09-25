#include "rudra/engine/frame_engine.hpp"

#include <algorithm>

namespace rudra {
namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

FrameEngine::FrameEngine(FrameLoader loader, FrameInfer infer, EngineOptions options)
    : loader_(std::move(loader)), infer_(std::move(infer)), opt_(options),
      worker_([this] { run(); }) {}

FrameEngine::~FrameEngine() {
    {
        std::lock_guard lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
}

void FrameEngine::set_sequence(int count) {
    std::lock_guard lk(mu_);
    generation_.bump();
    stats_.cancelled += queue_.size();
    queue_.clear();
    cache_.clear();
    lru_.clear();
    count_ = std::max(0, count);
    wanted_ = -1;
}

int FrameEngine::count() const {
    std::lock_guard lk(mu_);
    return count_;
}

void FrameEngine::on_ready(std::function<void(const ReadyFrame&)> cb) {
    std::lock_guard lk(mu_);
    ready_cb_ = std::move(cb);
}

bool FrameEngine::is_cached(int index) const {
    std::lock_guard lk(mu_);
    return cache_.count(index) != 0;
}

EngineStats FrameEngine::stats() const {
    std::lock_guard lk(mu_);
    EngineStats s = stats_;
    s.cached = int(cache_.size());
    return s;
}

std::shared_ptr<ReadyFrame> FrameEngine::lookup(int index) {
    auto it = cache_.find(index);
    if (it == cache_.end()) return nullptr;
    lru_.erase(it->second.second);
    lru_.push_front(index);
    it->second.second = lru_.begin();
    return it->second.first;
}

void FrameEngine::insert(std::shared_ptr<ReadyFrame> f) {
    const int index = f->index;
    if (cache_.count(index)) return;
    lru_.push_front(index);
    cache_.emplace(index, std::make_pair(std::move(f), lru_.begin()));
    // Evict the least recent, never the frame on screen.
    while (int(cache_.size()) > std::max(1, opt_.cache_frames)) {
        auto victim = std::prev(lru_.end());
        if (*victim == wanted_ && lru_.size() > 1) victim = std::prev(victim);
        cache_.erase(*victim);
        lru_.erase(victim);
    }
}

Generation::value_type FrameEngine::show(int index) {
    std::shared_ptr<ReadyFrame> hit;
    Generation::value_type g;
    std::function<void(const ReadyFrame&)> cb;
    {
        std::lock_guard lk(mu_);
        if (count_ <= 0) return generation_.current();
        index = std::clamp(index, 0, count_ - 1);
        g = generation_.bump();
        // Cancellation: nothing queued for an older generation will run.
        stats_.cancelled += queue_.size();
        queue_.clear();
        wanted_ = index;
        asked_at_ = std::chrono::steady_clock::now();
        ++stats_.shown;
        hit = lookup(index);
        if (hit) {
            ++stats_.cache_hits;
            stats_.last_request_to_ready_ms = 0.0;
            cb = ready_cb_;
        } else {
            queue_.push_back({index, Priority::Interactive, g});
        }
        // Read ahead, as the Studio's prefetch: the frames after this one, wrapping.
        for (int k = 1; k <= opt_.read_ahead && k < count_; ++k) {
            const int i = (index + k) % count_;
            if (!cache_.count(i)) queue_.push_back({i, Priority::ReadAhead, g});
        }
    }
    cv_.notify_all();
    if (hit && cb) {
        ReadyFrame f = *hit;
        f.generation = g;
        f.from_cache = true;
        cb(f);
    }
    return g;
}

void FrameEngine::deliver(const ReadyFrame& f) {
    std::function<void(const ReadyFrame&)> cb;
    {
        std::lock_guard lk(mu_);
        if (!generation_.is_current(f.generation) || f.index != wanted_) {
            ++stats_.stale_dropped;
            return;
        }
        stats_.last_request_to_ready_ms = ms_since(asked_at_);
        cb = ready_cb_;
    }
    if (cb) cb(f);
}

void FrameEngine::run() {
    for (;;) {
        Job job;
        {
            std::unique_lock lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            // Interactive first; otherwise in the order read-ahead was queued.
            auto it = std::find_if(queue_.begin(), queue_.end(), [](const Job& j) { return j.priority == Priority::Interactive; });
            if (it == queue_.end()) it = queue_.begin();
            job = *it;
            queue_.erase(it);
            if (!generation_.is_current(job.generation)) {
                ++stats_.cancelled;
                continue;
            }
            if (auto cached = lookup(job.index)) {
                if (job.priority == Priority::Interactive) {
                    ReadyFrame f = *cached;
                    f.generation = job.generation;
                    f.from_cache = true;
                    lk.unlock();
                    deliver(f);
                }
                continue;
            }
        }

        auto f = std::make_shared<ReadyFrame>();
        f->index = job.index;
        f->generation = job.generation;
        const auto t0 = std::chrono::steady_clock::now();
        auto sdr = loader_(job.index);
        f->decode_ms = ms_since(t0);
        if (!sdr) {
            f->error = sdr.error();
        } else {
            // A seek while decoding makes a read-ahead frame not worth inferring now.
            if (job.priority != Priority::Interactive && !generation_.is_current(job.generation)) {
                std::lock_guard lk(mu_);
                ++stats_.cancelled;
                continue;
            }
            const auto t1 = std::chrono::steady_clock::now();
            auto fr = infer_(*sdr);
            f->infer_ms = ms_since(t1);
            if (!fr) f->error = fr.error();
            else {
                f->sdr = std::make_shared<const SdrImage>(std::move(*sdr));
                f->fields = std::make_shared<const FrameResult>(std::move(*fr));
            }
        }
        {
            std::lock_guard lk(mu_);
            if (f->error) ++stats_.errors;
            else {
                ++stats_.inferred;
                insert(f);
            }
        }
        // The frame asked for is delivered even when it failed, so the UI can say why.
        if (job.priority == Priority::Interactive || f->index == wanted_) deliver(*f);
    }
}

}  // namespace rudra
