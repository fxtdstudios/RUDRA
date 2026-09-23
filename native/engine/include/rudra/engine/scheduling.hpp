#pragma once
// engine: priorities and generations (NATIVE_ARCHITECTURE.md 5.3).
//
// Every request carries the generation current when it was made. A seek bumps
// the generation, so a result that arrives late is recognised as stale and
// dropped instead of being shown on the wrong frame. No lock guards this.

#include <atomic>
#include <cstdint>

namespace rudra {

enum class Priority : std::uint8_t {
    Interactive = 0,   // the frame under the playhead
    ReadAhead = 1,     // frames in front of it
    Background = 2,    // masters, deliveries, thumbnails, the clip lane
};

class Generation {
public:
    using value_type = std::uint64_t;
    value_type current() const noexcept { return value_.load(std::memory_order_acquire); }
    value_type bump() noexcept { return value_.fetch_add(1, std::memory_order_acq_rel) + 1; }
    bool is_current(value_type g) const noexcept { return g == current(); }

private:
    std::atomic<value_type> value_{0};
};

}  // namespace rudra
