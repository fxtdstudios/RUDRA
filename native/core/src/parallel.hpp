#pragma once
// Internal to core: split [0, n) into contiguous chunks over a few threads.
// Every caller merges integer counts or fills disjoint output, so the result
// is the same number however many threads ran; sums that depend on order are
// kept out of here.

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace rudra::detail {

inline int worker_count(std::size_t n, std::size_t min_per_worker = 16384) {
    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    return int(std::max<std::size_t>(1, std::min<std::size_t>({std::size_t(hw), 8, n / min_per_worker + 1})));
}

// fn(worker, begin, end) for each chunk; chunk w covers [w n / k, (w + 1) n / k).
template <class F>
void parallel_chunks(std::size_t n, int workers, F&& fn) {
    if (workers <= 1) {
        fn(0, std::size_t(0), n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(std::size_t(workers - 1));
    for (int w = 1; w < workers; ++w)
        pool.emplace_back([&, w] { fn(w, n * std::size_t(w) / std::size_t(workers), n * std::size_t(w + 1) / std::size_t(workers)); });
    fn(0, std::size_t(0), n / std::size_t(workers));
    for (auto& t : pool) t.join();
}

}  // namespace rudra::detail
