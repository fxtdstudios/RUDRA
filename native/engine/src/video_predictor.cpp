#include "rudra/engine/video_predictor.hpp"

#include <algorithm>
#include <tuple>

namespace rudra {

Result<NetworkLinearImage> VideoPredictor::forward(const SdrImage& tile, const FrameScalars& scalars) {
    auto fields = backend_.tile_pass(tile, scalars);
    if (!fields) return fields.error();
    CompositeParams p;   // recovery "all", strength 1, preserve_outside, no grade
    return composite(tile, *fields, scalars, model_, p);
}

Result<VideoPrediction> VideoPredictor::predict(const SdrImage& rgb, const std::string& transfer,
                                                const std::string& primaries, ShadowSmoother& smoother) {
    auto canon = canonicalize_sdr(rgb.buffer(), transfer, "full");   // the decoder already made it full range
    if (!canon) return canon.error();
    PlanarBuffer x = std::move(*canon);
    if (primaries != "rec2020") {
        if (primaries != "rec709") return make_error(ErrorCode::InvalidArgument, "Unsupported primaries: " + primaries);
        x = srgb_code_to_rec2020(x, Primaries::Rec709);
    }
    const PlanarBuffer thumb = area_resize(x, kCutThumbHeight, kCutThumbWidth);
    const SdrImage frame(std::move(x));
    auto scalars = backend_.frame_pass(frame);
    if (!scalars) return scalars.error();
    VideoPrediction out;
    out.raw_weight = static_cast<double>(scalars->shadow_weight);   // 1 when the model has no shadow gate
    std::tie(out.shadow_weight, out.cut) = smoother.update(thumb, out.raw_weight);
    FrameScalars use = *scalars;
    use.shadow_weight = static_cast<float>(out.shadow_weight);

    const int h = frame.height(), w = frame.width();
    if (tile_size_ <= 0 || std::max(h, w) <= tile_size_) {
        auto whole = forward(frame, use);
        if (!whole) return whole.error();
        out.hdr = std::move(*whole);
        return out;
    }
    PlanarBuffer result(3, h, w, 0.0f), weights(1, h, w, 0.0f);
    for (const int y : video_tile_starts(h, tile_size_, overlap_)) {
        for (const int left : video_tile_starts(w, tile_size_, overlap_)) {
            const int th = std::min(tile_size_, h - y), tw = std::min(tile_size_, w - left);
            auto hdr = forward(frame.crop(y, left, th, tw), use);
            if (!hdr) return hdr.error();
            const auto blend = video_tile_blend(y, left, th, tw, h, w, overlap_);
            for (int r = 0; r < th; ++r) {
                float* wrow = weights.plane(0) + static_cast<std::size_t>(y + r) * w + left;
                const float* brow = blend.data() + static_cast<std::size_t>(r) * tw;
                for (int c = 0; c < 3; ++c) {
                    float* dst = result.plane(c) + static_cast<std::size_t>(y + r) * w + left;
                    const float* src = hdr->buffer().plane(c) + static_cast<std::size_t>(r) * tw;
                    for (int k = 0; k < tw; ++k) dst[k] += src[k] * brow[k];
                }
                for (int k = 0; k < tw; ++k) wrow[k] += brow[k];
            }
        }
    }
    const float* wp = weights.plane(0);
    for (int c = 0; c < 3; ++c) {
        float* dst = result.plane(c);
        for (std::size_t i = 0; i < result.plane_size(); ++i) dst[i] /= std::max(wp[i], 1e-6f);
    }
    out.hdr = NetworkLinearImage(std::move(result));
    return out;
}

}  // namespace rudra
