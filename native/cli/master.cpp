#include "master.hpp"

#include "rudra/infer/tiler.hpp"
#include "rudra/media/still.hpp"

namespace rudra {

Result<MasterResult> render_master(const ModelManifest& pkg, InferenceBackend& backend,
                                   const std::filesystem::path& image, const MasterRequest& q,
                                   const std::filesystem::path& out) {
    // Full resolution, untiled: the Studio's master is one pass over the frame.
    auto decoded = decode_sdr_file(image);
    if (!decoded) return decoded.error();
    auto fr = infer_frame(backend, decoded->rgb, TileConfig{0, 0});
    if (!fr) return fr.error();
    return write_master(decoded->rgb, decoded->bits, fr->fields, fr->scalars,
                        ModelConstants{pkg.log_scale, pkg.max_hdr, pkg.corpus_ev}, q, out);
}

}  // namespace rudra
