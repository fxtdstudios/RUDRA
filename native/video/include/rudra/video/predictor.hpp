#pragma once
// The video predictor (Phase 4, step 3): rudra/video.py Predictor.predict on
// a model package. The frame is canonicalised and brought to Rec.2020 code
// (core/video_predict), the frame pass gives the shadow weight and residual
// scale once, the smoother holds the weight across a shot, and the frame runs
// whole or in tiles blended with the video path's own feathers; each tile is
// the network's fields through the composite with preserve_outside and
// recovery "all", which is forward(...).hdr.

#include <string>

#include "rudra/core/composite.hpp"
#include "rudra/core/video_predict.hpp"
#include "rudra/infer/backend.hpp"

namespace rudra {

struct VideoPrediction {
    NetworkLinearImage hdr;      // 1.0 = 10 000 nits
    double raw_weight = 1.0;     // the frame pass's shadow weight, before smoothing
    double shadow_weight = 1.0;  // what the frame was reconstructed with
    bool cut = false;
};

class VideoPredictor {
public:
    VideoPredictor(InferenceBackend& backend, const ModelConstants& model, int tile_size, int overlap)
        : backend_(backend), model_(model), tile_size_(tile_size), overlap_(overlap) {}

    // `rgb` is the decoded frame as full-range code values in the clip's
    // transfer; `transfer` and `primaries` are the input contract's.
    Result<VideoPrediction> predict(const SdrImage& rgb, const std::string& transfer, const std::string& primaries,
                                    ShadowSmoother& smoother);

private:
    Result<NetworkLinearImage> forward(const SdrImage& tile, const FrameScalars& scalars);
    InferenceBackend& backend_;
    ModelConstants model_;
    int tile_size_, overlap_;
};

}  // namespace rudra
