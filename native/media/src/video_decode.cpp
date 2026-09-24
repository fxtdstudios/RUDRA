#include "rudra/media/video_decode.hpp"

#include <bit>
#include <cstring>
#include <fstream>
#include <iterator>

#include "rudra/platform/process.hpp"

namespace rudra {
namespace fs = std::filesystem;

static_assert(std::endian::native == std::endian::little, "raw frames are read as little-endian 16-bit");

namespace {
Error fail(std::string message) { return make_error(ErrorCode::IoError, std::move(message)); }
}  // namespace

SdrImage rgb_from_frame(const RawFrame16& frame) {
    PlanarBuffer out(3, frame.height, frame.width);
    const std::size_t n = static_cast<std::size_t>(frame.width) * frame.height;
    const auto c = static_cast<std::size_t>(frame.channels);
    for (int ch = 0; ch < 3; ++ch) {
        float* dst = out.plane(ch);
        for (std::size_t i = 0; i < n; ++i)
            dst[i] = static_cast<float>(frame.data[i * c + static_cast<std::size_t>(ch)]) / 65535.0f;
    }
    return SdrImage(std::move(out));
}

std::vector<std::string> decoder_command(const std::string& ffmpeg, const fs::path& source,
                                         const InputContract& contract, bool alpha) {
    return {ffmpeg,          "-hide_banner", "-v",        "error",
            "-xerror",       "-nostdin",     "-noautorotate", "-i",
            source.string(), "-map",         "0:v:0",     "-vf",
            decoder_filter(contract, alpha), "-fps_mode", "passthrough", "-pix_fmt",
            alpha ? "rgba64le" : "rgb48le",  "-f",        "rawvideo", "pipe:1"};
}

VideoDecoder::~VideoDecoder() { close(); }

Result<std::unique_ptr<VideoDecoder>> VideoDecoder::open(const fs::path& source, const VideoSource& video,
                                                         const fs::path& log) {
    auto ffmpeg = require_executable("ffmpeg");
    if (!ffmpeg) return ffmpeg.error();
    return open_command(decoder_command(ffmpeg->string(), source, video.contract, video.alpha), video.stream.width,
                        video.stream.height, video.alpha ? 4 : 3, video.clock.frames, log);
}

Result<std::unique_ptr<VideoDecoder>> VideoDecoder::open_command(const std::vector<std::string>& argv, int width,
                                                                 int height, int channels, int frames,
                                                                 const fs::path& log) {
    if (width <= 0 || height <= 0 || (channels != 3 && channels != 4) || frames < 0)
        return make_error(ErrorCode::InvalidArgument, "Invalid decoded frame shape");
    auto process = Process::start(argv, log);
    if (!process) return process.error();
    std::unique_ptr<VideoDecoder> d(new VideoDecoder());
    d->process_ = std::move(*process);
    d->log_ = log;
    d->width_ = width, d->height_ = height, d->channels_ = channels, d->frames_ = frames;
    return d;
}

Result<RawFrame16> VideoDecoder::next() {
    if (!process_) return fail("Decoder is closed");
    RawFrame16 f;
    f.width = width_, f.height = height_, f.channels = channels_;
    f.data.resize(static_cast<std::size_t>(width_) * height_ * channels_);
    const auto bytes = std::span<std::uint8_t>(reinterpret_cast<std::uint8_t*>(f.data.data()), f.bytes());
    const std::size_t got = process_->read(bytes);
    if (got == 0) return fail("Decoder ended before expected frame count");
    if (got != bytes.size()) return fail("Truncated decoded frame");
    ++index_;
    return f;
}

Result<void> VideoDecoder::finish() {
    if (!process_) return fail("Decoder is closed");
    std::uint8_t one[1];
    if (process_->read(one) != 0) {
        close();
        return fail("Decoder produced unexpected extra frames");
    }
    const int code = process_->wait();
    if (code != 0) {
        std::string text;
        if (!log_.empty()) {
            std::ifstream in(log_, std::ios::binary);
            text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
            if (text.size() > 2000) text = text.substr(text.size() - 2000);
        }
        process_.reset();
        return fail("Decoder failed; " + text);
    }
    process_.reset();
    return {};
}

void VideoDecoder::close() {
    if (!process_) return;
    if (process_->running()) process_->kill();
    process_->wait();
    process_.reset();
}

}  // namespace rudra
