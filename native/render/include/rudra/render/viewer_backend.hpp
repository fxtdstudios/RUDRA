#pragma once
// render: the viewer, on QRhi (Metal, D3D12, Vulkan, OpenGL fallback). QRhi
// itself is included only by the implementation (Phase 2); this interface is
// what the engine and the UI see (NATIVE_ARCHITECTURE.md 5.4, 6.3).

#include "rudra/core/color.hpp"
#include "rudra/core/image.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

// Which path the picture takes to the glass. The pipe bar shows it verbatim.
enum class OutputPath {
    SdrPqSimulation,   // SDR swapchain, today's PQ simulation
    ScRgb,             // FP16 linear Rec.709, 1.0 = 80 nits (Windows D3D12, Linux Vulkan)
    Hdr10,             // PQ Rec.2020 10-bit (Windows option)
    Edr,               // linear Display P3, 1.0 = SDR white (macOS Metal)
};

struct DisplayInfo {
    OutputPath path = OutputPath::SdrPqSimulation;
    Nits peak{203.0f};         // what the display can show right now (EDR headroom is live)
    Nits sdr_white{203.0f};
};

// Value written to the swapchain for a scene-linear luminance, per path.
// (The gamut matrix is applied separately; this is the scalar part.)
constexpr float output_scale(OutputPath path, Nits nits, Nits sdr_white) noexcept {
    switch (path) {
        case OutputPath::ScRgb: return nits.v / kScRgbUnit.v;
        case OutputPath::Edr: return nits.v / sdr_white.v;
        case OutputPath::Hdr10:
        case OutputPath::SdrPqSimulation: return nits.v / kPqPeak.v;   // then PQ-encoded
    }
    return 0.0f;
}

class ViewerBackend {
public:
    virtual ~ViewerBackend() = default;
    virtual DisplayInfo display() const = 0;
    virtual Result<void> upload_source(const SdrImage& sdr) = 0;
    virtual Result<void> upload_fields(const PlanarBuffer& residual, const PlanarBuffer& highlight,
                                       const PlanarBuffer& shadow) = 0;
    virtual Result<PlanarBuffer> readback_composite() = 0;   // fp32, for the parity test
};

}  // namespace rudra
