#include "rudra/deliver/video_master.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "rudra/core/hdr10.hpp"
#include "rudra/core/numeric.hpp"

namespace rudra {

Result<SpoolFrame> master_video_frame(const NetworkLinearImage& hdr, int index, const std::string& format,
                                      double peak_nits, std::optional<double> knee_nits,
                                      std::span<const std::uint16_t> alpha) {
    const auto px = hdr.buffer().span();
    if (std::any_of(px.begin(), px.end(), [](float v) { return !std::isfinite(v) || v < 0.0f; }))
        return make_error(ErrorCode::InvalidArgument, "Invalid HDR pixels at frame " + std::to_string(index));
    auto enc = encode_master(hdr.buffer(), format, peak_nits, knee_nits);
    if (!enc) return enc.error();
    const PlanarBuffer& code = enc->first;
    const PlanarBuffer& mastered = enc->second;
    const auto cs = code.span();
    if (std::any_of(cs.begin(), cs.end(), [](float v) { return !std::isfinite(v); }))
        return make_error(ErrorCode::InvalidArgument, "Invalid PQ pixels at frame " + std::to_string(index));

    const int h = hdr.height(), w = hdr.width();
    const std::size_t n = static_cast<std::size_t>(h) * w;
    SpoolFrame out;
    out.width = w, out.height = h, out.channels = alpha.empty() ? 3 : 4;
    if (!alpha.empty() && alpha.size() != n * 4)
        return make_error(ErrorCode::InvalidArgument, "Alpha frame does not match the HDR frame");

    std::vector<float> maxrgb(n);
    for (std::size_t i = 0; i < n; ++i)
        maxrgb[i] = std::max({mastered.plane(0)[i], mastered.plane(1)[i], mastered.plane(2)[i]});
    out.light.max_cll = static_cast<double>(*std::max_element(maxrgb.begin(), maxrgb.end()));
    out.light.frame_average = static_cast<double>(np_mean(std::span<const float>(maxrgb)));

    out.packed.resize(n * static_cast<std::size_t>(out.channels));
    const std::size_t c = static_cast<std::size_t>(out.channels);
    for (std::size_t i = 0; i < n; ++i) {
        for (int ch = 0; ch < 3; ++ch) {   // [..., ::-1]: B, G, R
            const float v = std::clamp(code.plane(2 - ch)[i], 0.0f, 1.0f) * 65535.0f + 0.5f;
            out.packed[i * c + std::size_t(ch)] = static_cast<std::uint16_t>(v);
        }
        if (c == 4) out.packed[i * c + 3] = alpha[i * 4 + 3];
    }
    return out;
}

LightCeilings light_ceilings(std::span<const VideoFrameLight> frames) {
    double cll = 0, fall = 0;
    bool first = true;
    for (const auto& f : frames) {
        cll = first ? f.max_cll : std::max(cll, f.max_cll);
        fall = first ? f.frame_average : std::max(fall, f.frame_average);
        first = false;
    }
    return {static_cast<int>(std::ceil(cll)), static_cast<int>(std::ceil(fall))};
}

std::string spool_frame_name(int index) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%08d.png", index);
    return buf;
}

bool spool_space_low(const std::filesystem::path& dir, std::uint64_t frame_bytes) {
    std::error_code ec;
    const auto s = std::filesystem::space(dir, ec);
    if (ec) return true;
    return s.available < frame_bytes * 2 + 64ull * 1024 * 1024;
}

}  // namespace rudra
