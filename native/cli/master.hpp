#pragma once
// `rudra-native master`: one still to a master EXR and its sidecar with no
// Python, the same stage sequence as ui/server.py _render_master (Phase 1
// step 7, docs/NATIVE_ARCHITECTURE.md section 14). Decode (media), fields
// (infer), composite and master chain (core), measure (core), EXR and
// sidecar (deliver). The CLI is the only layer allowed to see all four.

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/platform/pyjson.hpp"
#include "rudra/platform/result.hpp"

namespace rudra {

struct MasterBand {
    std::string label;
    double low_nits = 0.0, high_nits = 0.0, ev = 0.0;
};

// The Studio's master parameters, with its defaults.
struct MasterRequest {
    std::string checkpoint;                 // recorded, not loaded: the package is the model
    bool preserve_outside = true;
    std::string recovery_mode = "all";      // all, highlights, shadows, off
    double strength = 1.0;
    std::vector<MasterBand> regions;        // empty: the neutral default bands
    double region_softness_stops = 1.0;
    bool anchor = true;
    double anchor_knee = 0.9;
    bool carry_chroma = true;
    double chroma_knee = 0.99;
    std::string source_space = "rec709";
    std::string container = "aces";         // aces or linear
};

struct MasterResult {
    int maxcll = 0, maxfall = 0;
    double peak_nits = 0.0;
    int width = 0, height = 0, source_bits = 0;
    std::filesystem::path exr, sidecar;
};

Result<MasterResult> render_master(const ModelManifest& package, InferenceBackend& backend,
                                   const std::filesystem::path& image, const MasterRequest& request,
                                   const std::filesystem::path& out);

// The request as the Studio's params dict (what the golden index records).
Result<MasterRequest> master_request_from_json(const std::string& json_text);

}  // namespace rudra
