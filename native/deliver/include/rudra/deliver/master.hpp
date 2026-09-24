#pragma once
// The master (Phase 3 step 9, from the CLI of Phase 1 step 7): a frame to a
// master EXR and its sidecar, the stages of ui/server.py _render_master after
// the network -- composite and the master chain (core), measure (core), EXR
// and sidecar (deliver) -- and the Studio's render plan and publishing
// (master_targets and run_master). Decoding the still and running the network
// are the caller's (the CLI, the app's master job): deliver sees neither media
// nor infer, as the layer rule says.

#include <filesystem>
#include <string>
#include <vector>

#include "rudra/core/composite.hpp"
#include "rudra/core/fields.hpp"
#include "rudra/core/image.hpp"
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

// The request as the Studio's params dict (what the golden index records).
Result<MasterRequest> master_request_from_json(const std::string& json_text);

// One frame, decoded (`source_bits` its depth) and through the network
// (fields, scalars), to `out` and its .json beside it.
Result<MasterResult> write_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                  const FrameScalars& scalars, const ModelConstants& model,
                                  const MasterRequest& request, const std::filesystem::path& out);

// ---- the render plan (master_targets) -------------------------------------

struct RenderPlan {
    std::string render_dir;           // an absolute folder
    std::string render_name = "master";
    int render_count = 1;
    int frame_start = 1;
    bool sequence = false;            // render_mode "sequence": name.000001.exr ...; else name.exr
};
// The files a render will write, in order; refused (with the Studio's
// message) for a relative folder, a bad name or range, or any file or
// sidecar that exists already.
Result<std::vector<std::filesystem::path>> master_targets(const RenderPlan& plan);

// run_master's publishing: write into a staging folder beside `out`, then
// link the EXR and its sidecar into place without replacing anything.
Result<MasterResult> publish_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                    const FrameScalars& scalars, const ModelConstants& model,
                                    const MasterRequest& request, const std::filesystem::path& out);

}  // namespace rudra
