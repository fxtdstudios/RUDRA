#pragma once
// Image<Space>: planar float32, owns its pixels, tagged with what they mean.
// Conversions between spaces are named functions, never implicit, so every
// colour transform in the program can be found by name (principle P1).

#include <cassert>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "rudra/core/color.hpp"

namespace rudra {

// Untagged planar float32 buffer: channels x height x width, C order, which is
// NCHW with N = 1, the layout both inference runtimes take.
class PlanarBuffer {
public:
    PlanarBuffer() = default;
    PlanarBuffer(int channels, int height, int width, float fill = 0.0f)
        : c_(channels), h_(height), w_(width),
          data_(static_cast<std::size_t>(channels) * height * width, fill) {
        assert(channels > 0 && height > 0 && width > 0);
    }
    PlanarBuffer(int channels, int height, int width, std::vector<float> data)
        : c_(channels), h_(height), w_(width), data_(std::move(data)) {
        assert(data_.size() == static_cast<std::size_t>(channels) * height * width);
    }

    int channels() const noexcept { return c_; }
    int height() const noexcept { return h_; }
    int width() const noexcept { return w_; }
    std::size_t plane_size() const noexcept { return static_cast<std::size_t>(h_) * w_; }
    bool empty() const noexcept { return data_.empty(); }

    float* plane(int c) noexcept { return data_.data() + c * plane_size(); }
    const float* plane(int c) const noexcept { return data_.data() + c * plane_size(); }
    float& at(int c, int y, int x) noexcept { return data_[(c * static_cast<std::size_t>(h_) + y) * w_ + x]; }
    float at(int c, int y, int x) const noexcept { return data_[(c * static_cast<std::size_t>(h_) + y) * w_ + x]; }

    std::span<float> span() noexcept { return data_; }
    std::span<const float> span() const noexcept { return data_; }
    std::vector<float>& vector() noexcept { return data_; }

    // Copy of the rectangle [y, y+h) x [x, x+w), all channels.
    PlanarBuffer crop(int y, int x, int h, int w) const {
        assert(y >= 0 && x >= 0 && y + h <= h_ && x + w <= w_);
        PlanarBuffer out(c_, h, w);
        for (int c = 0; c < c_; ++c)
            for (int r = 0; r < h; ++r) {
                const float* src = plane(c) + static_cast<std::size_t>(y + r) * w_ + x;
                std::copy(src, src + w, out.plane(c) + static_cast<std::size_t>(r) * w);
            }
        return out;
    }

private:
    int c_ = 0, h_ = 0, w_ = 0;
    std::vector<float> data_;
};

template <class Space>
class Image {
public:
    using space_type = Space;
    static constexpr ColorEncoding encoding = Space::encoding;

    Image() = default;
    Image(int height, int width, float fill = 0.0f) : buf_(3, height, width, fill) {}
    explicit Image(PlanarBuffer rgb) : buf_(std::move(rgb)) { assert(buf_.channels() == 3); }

    int height() const noexcept { return buf_.height(); }
    int width() const noexcept { return buf_.width(); }
    const PlanarBuffer& buffer() const noexcept { return buf_; }
    PlanarBuffer& buffer() noexcept { return buf_; }
    Image crop(int y, int x, int h, int w) const { return Image(buf_.crop(y, x, h, w)); }

private:
    PlanarBuffer buf_;
};

using SdrImage = Image<space::SdrDisplay>;
using NetworkLinearImage = Image<space::NetworkLinear>;

}  // namespace rudra
