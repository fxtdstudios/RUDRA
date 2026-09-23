#include "rudra/infer/tiler.hpp"

namespace rudra {

Result<FrameResult> infer_frame(InferenceBackend& backend, const SdrImage& frame, const TileConfig& cfg) {
    auto scalars = backend.frame_pass(frame);
    if (!scalars) return scalars.error();

    const auto tiles = plan_tiles(frame.height(), frame.width(), cfg);
    FrameResult out;
    out.scalars = *scalars;
    if (tiles.size() == 1) {
        auto f = backend.tile_pass(frame, out.scalars);
        if (!f) return f.error();
        out.fields = std::move(*f);
        return out;
    }

    Stitcher residual(3, frame.height(), frame.width());
    Stitcher highlight(1, frame.height(), frame.width());
    Stitcher shadow(1, frame.height(), frame.width());
    for (const auto& t : tiles) {
        auto f = backend.tile_pass(frame.crop(t.y, t.x, t.h, t.w), out.scalars);
        if (!f) return f.error();
        const PlanarBuffer w = tile_weight(t, cfg.overlap, frame.height(), frame.width());
        residual.add(t, f->residual, w);
        highlight.add(t, f->highlight, w);
        shadow.add(t, f->shadow, w);
    }
    out.fields = Fields{std::move(residual).finish(), std::move(highlight).finish(), std::move(shadow).finish()};
    out.tiled = true;
    return out;
}

const char* to_string(Runtime r) noexcept {
    switch (r) {
        case Runtime::LibTorch: return "libtorch";
        case Runtime::OnnxRuntime: return "onnxruntime";
    }
    return "?";
}

const char* to_string(Device d) noexcept {
    switch (d) {
        case Device::Cpu: return "cpu";
        case Device::Cuda: return "cuda";
        case Device::Mps: return "mps";
        case Device::DirectML: return "directml";
        case Device::CoreML: return "coreml";
        case Device::Rocm: return "rocm";
        case Device::OpenVino: return "openvino";
    }
    return "?";
}

std::vector<Runtime> compiled_runtimes() {
    std::vector<Runtime> r;
#ifdef RUDRA_HAVE_LIBTORCH
    r.push_back(Runtime::LibTorch);
#endif
#ifdef RUDRA_HAVE_ONNXRUNTIME
    r.push_back(Runtime::OnnxRuntime);
#endif
    return r;
}

#ifndef RUDRA_HAVE_LIBTORCH
Result<std::unique_ptr<InferenceBackend>> make_libtorch_backend(const ModelManifest&, Device) {
    return make_error(ErrorCode::Unsupported, "This build of RUDRA does not include LibTorch.",
                      "configure with -DRUDRA_WITH_LIBTORCH=ON");
}
#endif
#ifndef RUDRA_HAVE_ONNXRUNTIME
Result<std::unique_ptr<InferenceBackend>> make_onnxruntime_backend(const ModelManifest&, Device) {
    return make_error(ErrorCode::Unsupported, "This build of RUDRA does not include ONNX Runtime.",
                      "configure with -DRUDRA_WITH_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=<path>");
}
#endif

}  // namespace rudra
