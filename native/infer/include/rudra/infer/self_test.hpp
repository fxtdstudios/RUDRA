#pragma once
// The package's golden frames through a backend (Gate A's native half, and
// the app's first-load check, DESKTOP_APP_PLAN 2.1: "re-runs the goldens on
// the chosen backend and refuses a model that drifts"). Each golden frame
// untiled, then the stitch case tiled, held to the manifest's tolerance for
// that runtime and device.

#include <functional>
#include <map>
#include <string>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct SelfTestStat {
    double max_abs = 0.0;
    double excess = -1e300;   // max(|d| - (atol + rtol |ref|)); <= 0 passes
    bool pass() const { return excess <= 0.0; }
};

struct SelfTestReport {
    BackendInfo backend;
    std::string tolerance_key;        // "torchscript", "onnx" or "gpu_fp32"
    Tolerance tolerance;
    int frames = 0;
    bool stitched = false;
    std::map<std::string, SelfTestStat> outputs;   // residual_scale, ..., "stitched residual", ...
    bool pass() const;
    std::string summary() const;      // one line: "passed, 16 frames + stitch, worst 3.1e-06"
};

// Which of the manifest's tolerances a backend is held to.
std::string tolerance_key(const BackendInfo& info);

// Runs every golden; `progress` (optional) is called with frames done and the total.
Result<SelfTestReport> self_test(const ModelManifest& m, InferenceBackend& backend,
                                 const std::function<void(int, int)>& progress = {});

}  // namespace rudra
