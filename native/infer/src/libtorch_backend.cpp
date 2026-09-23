// LibTorch backend: loads model.ts and calls its frame_pass / tile_pass methods.
// The reference runtime: on CPU it runs eager PyTorch's own kernels, which is
// why the export measures bit-exact (tools/export_model.py).

#include <torch/cuda.h>
#include <torch/mps.h>
#include <torch/script.h>
#include <torch/version.h>

#include <cstring>

#include "rudra/infer/backend.hpp"

namespace rudra {
namespace {

torch::Tensor to_tensor(const SdrImage& img, const torch::Device& dev) {
    const auto& b = img.buffer();
    auto t = torch::from_blob(const_cast<float*>(b.span().data()), {1, 3, b.height(), b.width()},
                              torch::TensorOptions().dtype(torch::kFloat32));
    return dev.is_cpu() ? t : t.to(dev);
}

PlanarBuffer to_planar(const torch::Tensor& t) {
    auto c = t.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const auto s = c.sizes();   // (1, C, H, W)
    std::vector<float> v(static_cast<std::size_t>(c.numel()));
    std::memcpy(v.data(), c.data_ptr<float>(), v.size() * sizeof(float));
    return PlanarBuffer(static_cast<int>(s[1]), static_cast<int>(s[2]), static_cast<int>(s[3]), std::move(v));
}

class LibTorchBackend final : public InferenceBackend {
public:
    LibTorchBackend(torch::jit::Module m, torch::Device dev, Device kind)
        : module_(std::move(m)), device_(dev), kind_(kind) {}

    BackendInfo info() const override {
        return {Runtime::LibTorch, kind_, TORCH_VERSION, device_.str()};
    }

    Result<FrameScalars> frame_pass(const SdrImage& frame) override {
        try {
            c10::InferenceMode guard;
            auto out = module_.get_method("frame_pass")({to_tensor(frame, device_)}).toTuple();
            FrameScalars s;
            s.residual_scale = out->elements()[0].toTensor().to(torch::kCPU).item<float>();
            s.shadow_weight = out->elements()[1].toTensor().to(torch::kCPU).item<float>();
            auto curve = out->elements()[2].toTensor().to(torch::kCPU).to(torch::kFloat32).contiguous();
            s.curve_params.assign(curve.data_ptr<float>(), curve.data_ptr<float>() + curve.numel());
            return s;
        } catch (const std::exception& e) {
            return make_error(ErrorCode::BackendError, "LibTorch could not run the frame pass.", e.what());
        }
    }

    Result<Fields> tile_pass(const SdrImage& tile, const FrameScalars& s) override {
        try {
            c10::InferenceMode guard;
            auto scale = torch::full({1, 1, 1, 1}, s.residual_scale).to(device_);
            auto curve = torch::from_blob(const_cast<float*>(s.curve_params.data()),
                                          {1, static_cast<long>(s.curve_params.size())},
                                          torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device_);
            auto out = module_.get_method("tile_pass")({to_tensor(tile, device_), scale, curve}).toTuple();
            return Fields{to_planar(out->elements()[0].toTensor()), to_planar(out->elements()[1].toTensor()),
                          to_planar(out->elements()[2].toTensor())};
        } catch (const std::exception& e) {
            return make_error(ErrorCode::BackendError, "LibTorch could not run the tile pass.", e.what());
        }
    }

private:
    torch::jit::Module module_;
    torch::Device device_;
    Device kind_;
};

}  // namespace

Result<std::unique_ptr<InferenceBackend>> make_libtorch_backend(const ModelManifest& m, Device device) {
    torch::Device dev(torch::kCPU);
    switch (device) {
        case Device::Cpu: break;
        case Device::Cuda:
            if (!torch::cuda::is_available())
                return make_error(ErrorCode::Unsupported, "No CUDA device is available to LibTorch.");
            dev = torch::Device(torch::kCUDA, 0);
            break;
        case Device::Mps:
            if (!torch::mps::is_available())
                return make_error(ErrorCode::Unsupported, "The Apple GPU (MPS) is not available to LibTorch.");
            dev = torch::Device(torch::kMPS);
            break;
        default:
            return make_error(ErrorCode::Unsupported, "LibTorch does not run on that device.", to_string(device));
    }
    try {
        auto module = torch::jit::load((m.root / m.torchscript).string(), dev);
        module.eval();
        return std::unique_ptr<InferenceBackend>(new LibTorchBackend(std::move(module), dev, device));
    } catch (const std::exception& e) {
        return make_error(ErrorCode::BackendError, "LibTorch could not load the model.", e.what());
    }
}

}  // namespace rudra
