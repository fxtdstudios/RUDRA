#pragma once
// Frame -> fields through any backend: the frame pass once, then tiles, then
// stitching exactly as training/infer_sdr2hdr.py::predict_fields does.

#include "rudra/core/tiling.hpp"
#include "rudra/infer/backend.hpp"

namespace rudra {

struct FrameResult {
    FrameScalars scalars;
    Fields fields;
    bool tiled = false;
};

Result<FrameResult> infer_frame(InferenceBackend& backend, const SdrImage& frame, const TileConfig& cfg);

}  // namespace rudra
