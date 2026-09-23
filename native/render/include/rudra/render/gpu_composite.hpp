#pragma once
// The composite on the GPU, through QRhi (Direct3D 12/11, Metal, Vulkan,
// OpenGL). The same function as rudra/core/composite.hpp, rendered offscreen
// and read back, so the viewer's pixels can be held to the C++ reference
// (NATIVE_ARCHITECTURE.md section 12, day 8; docs/composite.spec.md).
//
// Qt-free on purpose: QRhi stays inside render/src. A QGuiApplication must
// exist before create() is called.

#include <memory>
#include <string>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

enum class GpuApi { Auto, D3D12, D3D11, Metal, Vulkan, OpenGL };
// The render target format: Fp32 for parity, Fp16 for what the viewer shows.
enum class GpuPrecision { Fp32, Fp16 };

const char* to_string(GpuApi a) noexcept;

struct GpuCompositorInfo {
    std::string backend;   // QRhi backend name
    std::string device;    // driver's device name
};

class GpuCompositor {
public:
    // Auto: D3D12 on Windows, Metal on macOS, Vulkan (then OpenGL) elsewhere.
    static Result<std::unique_ptr<GpuCompositor>> create(GpuApi api);
    virtual ~GpuCompositor() = default;
    virtual GpuCompositorInfo info() const = 0;
    virtual bool supports(GpuPrecision p) const = 0;
    // 3 x h x w in the network convention, widened to float from Fp16.
    virtual Result<PlanarBuffer> composite(const SdrImage& sdr, const Fields& fields, const FrameScalars& scalars,
                                           const ModelConstants& model, const CompositeParams& params,
                                           GpuPrecision precision) = 0;
};

}  // namespace rudra
