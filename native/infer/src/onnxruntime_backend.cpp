// ONNX Runtime backend: model.frame.onnx and model.tile.onnx, inputs fed by the
// names the manifest records (the exporter drops inputs a model does not use).
// Execution providers are appended only when this ORT build carries them; CPU
// is always the fallback.

#include <onnxruntime_cxx_api.h>

// Provider factories that are not part of the generic C++ API. Each ships only
// in the ORT package built with that provider (DirectML on Windows, Core ML on
// Apple), so their presence decides what this build can offer.
#if __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define RUDRA_ORT_HAS_DML 1
#else
#define RUDRA_ORT_HAS_DML 0
#endif
#if __has_include(<coreml_provider_factory.h>)
#include <coreml_provider_factory.h>
#define RUDRA_ORT_HAS_COREML 1
#else
#define RUDRA_ORT_HAS_COREML 0
#endif

#include <algorithm>
#include <array>
#include <cstring>

#include "rudra/infer/backend.hpp"

namespace rudra {
namespace {

Ort::Env& env() {
    // ERROR, not WARNING: ORT warns on every GPU session that it placed shape
    // ops on the CPU, which is by design and not something to act on.
    static Ort::Env e(ORT_LOGGING_LEVEL_ERROR, "rudra");
    return e;
}

bool has(const std::vector<std::string>& v, const char* name) {
    return std::find(v.begin(), v.end(), name) != v.end();
}

PlanarBuffer to_planar(Ort::Value& v) {
    const auto info = v.GetTensorTypeAndShapeInfo();
    const auto shape = info.GetShape();   // (1, C, H, W)
    const auto n = info.GetElementCount();
    std::vector<float> data(n);
    std::memcpy(data.data(), v.GetTensorData<float>(), n * sizeof(float));
    return PlanarBuffer(static_cast<int>(shape[1]), static_cast<int>(shape[2]), static_cast<int>(shape[3]),
                        std::move(data));
}

class OrtBackend final : public InferenceBackend {
public:
    OrtBackend(const ModelManifest& m, Ort::SessionOptions& so, Device device, std::string providers)
        : frame_(env(), (m.root / m.onnx_frame).c_str(), so),
          tile_(env(), (m.root / m.onnx_tile).c_str(), so),
          frame_inputs_(m.onnx_frame_inputs), tile_inputs_(m.onnx_tile_inputs),
          device_(device), providers_(std::move(providers)) {}

    BackendInfo info() const override {
        return {Runtime::OnnxRuntime, device_, Ort::GetVersionString(), providers_};
    }

    Result<FrameScalars> frame_pass(const SdrImage& frame) override {
        try {
            const auto& b = frame.buffer();
            std::array<int64_t, 4> shape{1, 3, b.height(), b.width()};
            std::vector<const char*> names;
            std::vector<Ort::Value> values;
            if (has(frame_inputs_, "sdr")) {
                names.push_back("sdr");
                values.push_back(Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(b.span().data()),
                                                                  b.span().size(), shape.data(), shape.size()));
            }
            const char* outs[] = {"residual_scale", "shadow_weight", "curve_params"};
            auto r = frame_.Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), outs, 3);
            FrameScalars s;
            s.residual_scale = r[0].GetTensorData<float>()[0];
            s.shadow_weight = r[1].GetTensorData<float>()[0];
            const auto n = r[2].GetTensorTypeAndShapeInfo().GetElementCount();
            s.curve_params.assign(r[2].GetTensorData<float>(), r[2].GetTensorData<float>() + n);
            return s;
        } catch (const std::exception& e) {
            return make_error(ErrorCode::BackendError, "ONNX Runtime could not run the frame pass.", e.what());
        }
    }

    Result<Fields> tile_pass(const SdrImage& tile, const FrameScalars& s) override {
        try {
            const auto& b = tile.buffer();
            std::array<int64_t, 4> shape{1, 3, b.height(), b.width()};
            std::array<int64_t, 4> scale_shape{1, 1, 1, 1};
            std::array<int64_t, 2> curve_shape{1, static_cast<int64_t>(s.curve_params.size())};
            float scale = s.residual_scale;
            std::vector<float> curve = s.curve_params;
            std::vector<const char*> names;
            std::vector<Ort::Value> values;
            names.push_back("sdr");
            values.push_back(Ort::Value::CreateTensor<float>(mem_, const_cast<float*>(b.span().data()),
                                                              b.span().size(), shape.data(), shape.size()));
            if (has(tile_inputs_, "residual_scale")) {
                names.push_back("residual_scale");
                values.push_back(Ort::Value::CreateTensor<float>(mem_, &scale, 1, scale_shape.data(), 4));
            }
            if (has(tile_inputs_, "curve_params")) {
                names.push_back("curve_params");
                values.push_back(Ort::Value::CreateTensor<float>(mem_, curve.data(), curve.size(),
                                                                  curve_shape.data(), 2));
            }
            const char* outs[] = {"residual", "highlight", "shadow"};
            auto r = tile_.Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), outs, 3);
            return Fields{to_planar(r[0]), to_planar(r[1]), to_planar(r[2])};
        } catch (const std::exception& e) {
            return make_error(ErrorCode::BackendError, "ONNX Runtime could not run the tile pass.", e.what());
        }
    }

private:
    Ort::MemoryInfo mem_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Session frame_, tile_;
    std::vector<std::string> frame_inputs_, tile_inputs_;
    Device device_;
    std::string providers_;
};

}  // namespace

Result<std::unique_ptr<InferenceBackend>> make_onnxruntime_backend(const ModelManifest& m, Device device) {
    try {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const auto available = Ort::GetAvailableProviders();
        auto offered = [&](const char* p) { return std::find(available.begin(), available.end(), p) != available.end(); };
        std::string providers = "CPUExecutionProvider";
        switch (device) {
            case Device::Cpu:
                break;
            case Device::Cuda:
                if (!offered("CUDAExecutionProvider"))
                    return make_error(ErrorCode::Unsupported, "This ONNX Runtime has no CUDA provider.");
                so.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
                providers = "CUDAExecutionProvider,CPUExecutionProvider";
                break;
            case Device::DirectML: {
                if (!offered("DmlExecutionProvider"))
                    return make_error(ErrorCode::Unsupported, "This ONNX Runtime build has no DirectML provider.",
                                      "use the Microsoft.ML.OnnxRuntime.DirectML package");
#if RUDRA_ORT_HAS_DML
                // DirectML needs memory patterns off and sequential execution.
                so.DisableMemPattern();
                so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
                providers = "DmlExecutionProvider,CPUExecutionProvider";
                break;
#else
                return make_error(ErrorCode::Unsupported, "This build was compiled without the DirectML header.",
                                  "dml_provider_factory.h not found under ONNXRUNTIME_ROOT/include");
#endif
            }
            case Device::CoreML: {
                if (!offered("CoreMLExecutionProvider"))
                    return make_error(ErrorCode::Unsupported, "This ONNX Runtime build has no Core ML provider.");
#if RUDRA_ORT_HAS_COREML
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(so, 0));
                providers = "CoreMLExecutionProvider,CPUExecutionProvider";
                break;
#else
                return make_error(ErrorCode::Unsupported, "This build was compiled without the Core ML header.",
                                  "coreml_provider_factory.h not found under ONNXRUNTIME_ROOT/include");
#endif
            }
            case Device::Rocm:
                if (!offered("ROCMExecutionProvider"))
                    return make_error(ErrorCode::Unsupported, "This ONNX Runtime build has no ROCm provider.");
                so.AppendExecutionProvider_ROCM(OrtROCMProviderOptions{});
                providers = "ROCMExecutionProvider,CPUExecutionProvider";
                break;
            case Device::OpenVino:
                if (!offered("OpenVINOExecutionProvider"))
                    return make_error(ErrorCode::Unsupported, "This ONNX Runtime build has no OpenVINO provider.");
                so.AppendExecutionProvider_OpenVINO_V2({});
                providers = "OpenVINOExecutionProvider,CPUExecutionProvider";
                break;
            case Device::Mps:
                return make_error(ErrorCode::Unsupported, "ONNX Runtime reaches the Apple GPU through Core ML, not MPS.");
        }
        return std::unique_ptr<InferenceBackend>(new OrtBackend(m, so, device, providers));
    } catch (const std::exception& e) {
        return make_error(ErrorCode::BackendError, "ONNX Runtime could not load the model.", e.what());
    }
}

}  // namespace rudra
