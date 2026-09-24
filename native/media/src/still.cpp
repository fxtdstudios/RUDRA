#include "rudra/media/still.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <cstdio>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace rudra {
namespace {

template <class T>
std::size_t distinct_green(const cv::Mat& m) {
    std::vector<T> g;
    g.reserve(m.total());
    const int ch = m.channels();
    const int gi = ch >= 3 ? 1 : 0;   // BGR: green is channel 1 either way
    for (int y = 0; y < m.rows; ++y) {
        const T* row = m.ptr<T>(y);
        for (int x = 0; x < m.cols; ++x) g.push_back(row[x * ch + gi]);
    }
    std::sort(g.begin(), g.end());
    return static_cast<std::size_t>(std::unique(g.begin(), g.end()) - g.begin());
}

// Every stored value on the 8-bit lattice of a 16-bit container, either
// scaled (n * 257) or shifted (n * 256): an 8-bit frame padded up.
bool padded_8bit(const cv::Mat& m) {
    const int ch = std::min(m.channels(), 3);
    for (const unsigned step : {65535u / 255u, 65536u / 256u}) {
        bool all = true;
        for (int y = 0; y < m.rows && all; ++y) {
            const std::uint16_t* row = m.ptr<std::uint16_t>(y);
            for (int x = 0; x < m.cols && all; ++x)
                for (int c = 0; c < ch; ++c)
                    if (row[x * m.channels() + c] % step) { all = false; break; }
        }
        if (all) return true;
    }
    return false;
}

}  // namespace

Result<DecodedStill> decode_sdr(std::span<const std::uint8_t> bytes) {
    if (bytes.empty()) return make_error(ErrorCode::ParseError, "The image file is empty.");
    const cv::Mat buf(1, static_cast<int>(bytes.size()), CV_8UC1, const_cast<std::uint8_t*>(bytes.data()));
    cv::Mat m;
    try {
        m = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& e) {
        return make_error(ErrorCode::ParseError, "The image could not be decoded.", e.what());
    }
    if (m.empty()) return make_error(ErrorCode::Unsupported, "This image format is not readable by this build.");
    const int depth = m.depth(), ch = m.channels();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F)
        return make_error(ErrorCode::Unsupported, "Only 8-bit, 16-bit and float images are read.");
    if (ch != 1 && ch != 3 && ch != 4)
        return make_error(ErrorCode::Unsupported, "Only grey, RGB and RGBA images are read.");

    const int h = m.rows, w = m.cols;
    DecodedStill out;
    out.source = "opencv";
    PlanarBuffer rgb(3, h, w);
    // BGR(A) -> RGB planes; grey replicated.
    auto channel_of = [&](int c) { return ch == 1 ? 0 : 2 - c; };

    if (depth == CV_32F) {
        float top = -std::numeric_limits<float>::infinity();
        for (int y = 0; y < h; ++y) {
            const float* row = m.ptr<float>(y);
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < std::min(ch, 3); ++c)
                    if (!std::isnan(row[x * ch + c])) top = std::max(top, row[x * ch + c]);
        }
        if (top > 1.0f + 1e-4f) {
            char msg[256];
            std::snprintf(msg, sizeof msg,
                          "float input peaks at %.3f: this is scene-linear HDR, not an SDR frame. RUDRA "
                          "reconstructs display-encoded SDR; an HDR file does not need reconstructing.",
                          double(top));
            return make_error(ErrorCode::Unsupported, msg);
        }
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y) {
                const float* row = m.ptr<float>(y);
                for (int x = 0; x < w; ++x) rgb.at(c, y, x) = std::clamp(row[x * ch + channel_of(c)], 0.0f, 1.0f);
            }
        out.bits = 32;
        out.distinct_codes = static_cast<int>(distinct_green<float>(m));
    } else if (depth == CV_16U) {
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y) {
                const std::uint16_t* row = m.ptr<std::uint16_t>(y);
                for (int x = 0; x < w; ++x) rgb.at(c, y, x) = float(row[x * ch + channel_of(c)]) / 65535.0f;
            }
        out.bits = padded_8bit(m) ? 8 : 16;
        out.distinct_codes = static_cast<int>(distinct_green<std::uint16_t>(m));
    } else {
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y) {
                const std::uint8_t* row = m.ptr<std::uint8_t>(y);
                for (int x = 0; x < w; ++x) rgb.at(c, y, x) = float(row[x * ch + channel_of(c)]) / 255.0f;
            }
        out.bits = 8;
        out.distinct_codes = static_cast<int>(distinct_green<std::uint8_t>(m));
    }
    out.rgb = SdrImage(std::move(rgb));
    return out;
}

Result<DecodedStill> decode_sdr_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return make_error(ErrorCode::NotFound, "The image file could not be opened.", path.string());
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    auto r = decode_sdr(bytes);
    if (!r) return make_error(r.error().code, r.error().message, path.string() + (r.error().detail.empty() ? "" : ": " + r.error().detail));
    return r;
}

}  // namespace rudra

namespace rudra {

SdrImage fit_max_side(const SdrImage& rgb, int max_side) {
    const auto& b = rgb.buffer();
    const int h = b.height(), w = b.width();
    if (max_side <= 0 || std::max(h, w) <= max_side) return rgb;
    // int(width * ratio), as the Python truncates.
    const double ratio = double(max_side) / double(std::max(h, w));
    const int nw = std::max(1, int(double(w) * ratio)), nh = std::max(1, int(double(h) * ratio));
    cv::Mat src(h, w, CV_32FC3);
    for (int y = 0; y < h; ++y) {
        float* row = src.ptr<float>(y);
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) row[x * 3 + c] = b.at(c, y, x);
    }
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    PlanarBuffer out(3, nh, nw);
    for (int y = 0; y < nh; ++y) {
        const float* row = dst.ptr<float>(y);
        for (int x = 0; x < nw; ++x)
            for (int c = 0; c < 3; ++c) out.at(c, y, x) = row[x * 3 + c];
    }
    return SdrImage(std::move(out));
}

}  // namespace rudra
