#pragma once
// InferenceBackend (NATIVE_ARCHITECTURE.md 5.4): one interface, several runtimes.
// A backend runs the two entry points of a model package; the tiler, the
// stitching and everything after the fields live outside it.

#include <memory>
#include <string>
#include <vector>

#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/core/model_manifest.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

enum class Runtime { LibTorch, OnnxRuntime };
enum class Device { Cpu, Cuda, Mps, DirectML, CoreML, Rocm, OpenVino };

const char* to_string(Runtime r) noexcept;
const char* to_string(Device d) noexcept;

struct BackendInfo {
    Runtime runtime;
    Device device;
    std::string version;       // runtime version
    std::string detail;        // device name or provider list
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual BackendInfo info() const = 0;
    virtual Result<FrameScalars> frame_pass(const SdrImage& frame) = 0;
    virtual Result<Fields> tile_pass(const SdrImage& tile, const FrameScalars& scalars) = 0;
};

// Factories. Each returns an Unsupported error when this build lacks the runtime.
Result<std::unique_ptr<InferenceBackend>> make_libtorch_backend(const ModelManifest& m, Device device);
Result<std::unique_ptr<InferenceBackend>> make_onnxruntime_backend(const ModelManifest& m, Device device);

// Which runtimes this binary was built with.
std::vector<Runtime> compiled_runtimes();

}  // namespace rudra
