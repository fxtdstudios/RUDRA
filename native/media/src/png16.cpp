#include "rudra/media/png16.hpp"

#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>

namespace rudra {

Result<void> write_png16(const std::filesystem::path& path, std::span<const std::uint16_t> hwc, int width,
                         int height, int channels) {
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4) ||
        hwc.size() != static_cast<std::size_t>(width) * height * channels)
        return make_error(ErrorCode::InvalidArgument, "Failed writing signal spool");
    const cv::Mat m(height, width, CV_16UC(channels), const_cast<std::uint16_t*>(hwc.data()));
    bool ok = false;
    try {
        ok = cv::imwrite(path.string(), m);
    } catch (const cv::Exception&) {
        ok = false;
    }
    if (!ok) return make_error(ErrorCode::IoError, "Failed writing signal spool", path.string());
    return {};
}

Result<Png16> read_png16(const std::filesystem::path& path) {
    cv::Mat m;
    try {
        m = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception&) {
    }
    if (m.empty() || m.depth() != CV_16U)
        return make_error(ErrorCode::ParseError, "Not a 16-bit PNG", path.string());
    Png16 p;
    p.width = m.cols, p.height = m.rows, p.channels = m.channels();
    p.data.resize(static_cast<std::size_t>(m.total()) * p.channels);
    const std::size_t row = static_cast<std::size_t>(m.cols) * p.channels;
    for (int y = 0; y < m.rows; ++y) std::memcpy(p.data.data() + y * row, m.ptr<std::uint16_t>(y), row * 2);
    return p;
}

}  // namespace rudra

namespace rudra {
std::string opencv_version() { return CV_VERSION; }
}  // namespace rudra
