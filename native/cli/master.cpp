#include "master.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <nlohmann/json.hpp>

#include "rudra/core/master.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/sidecars.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/media/still.hpp"

namespace rudra {
namespace {

Result<Primaries> primaries_of(const std::string& s) {
    if (s == "rec709") return Primaries::Rec709;
    if (s == "rec2020") return Primaries::Rec2020;
    if (s == "p3d65") return Primaries::P3D65;
    return make_error(ErrorCode::InvalidArgument, "Unknown source space.", s);
}

Result<RecoveryMode> mode_of(const std::string& s) {
    if (s == "all") return RecoveryMode::All;
    if (s == "highlights") return RecoveryMode::Highlights;
    if (s == "shadows") return RecoveryMode::Shadows;
    if (s == "off") return RecoveryMode::Off;
    return make_error(ErrorCode::InvalidArgument, "recovery_mode must be all, highlights, shadows, or off", s);
}

// Python round(x, 1): the correctly rounded decimal, read back.
double py_round1(double x) {
    char b[64];
    std::snprintf(b, sizeof b, "%.1f", x);
    return std::strtod(b, nullptr);
}

pyjson::Value band_json(const MasterBand& b) {
    return pyjson::Dict{{"label", b.label}, {"low_nits", b.low_nits}, {"high_nits", b.high_nits}, {"ev", b.ev}};
}

}  // namespace

Result<MasterRequest> master_request_from_json(const std::string& text) {
    MasterRequest r;
    try {
        const auto j = nlohmann::json::parse(text);
        r.checkpoint = j.value("checkpoint", std::string());
        r.preserve_outside = j.value("preserve_outside", true);
        r.recovery_mode = j.value("recovery_mode", std::string("all"));
        r.strength = j.value("strength", 1.0);
        r.region_softness_stops = j.value("region_softness_stops", 1.0);
        r.anchor = j.value("anchor", true);
        r.anchor_knee = j.value("anchor_knee", 0.9);
        r.carry_chroma = j.value("carry_chroma", true);
        r.chroma_knee = j.value("chroma_knee", 0.99);
        r.source_space = j.value("source_space", std::string("rec709"));
        r.container = j.value("container", std::string("aces"));
        if (j.contains("regions"))
            for (const auto& b : j.at("regions"))
                r.regions.push_back({b.value("label", std::string()), b.at("low_nits").get<double>(),
                                     b.at("high_nits").get<double>(), b.value("ev", 0.0)});
    } catch (const std::exception& e) {
        return make_error(ErrorCode::ParseError, "The master parameters are not valid JSON.", e.what());
    }
    return r;
}

Result<MasterResult> render_master(const ModelManifest& pkg, InferenceBackend& backend,
                                   const std::filesystem::path& image, const MasterRequest& q,
                                   const std::filesystem::path& out) {
    auto src = primaries_of(q.source_space);
    if (!src) return src.error();
    auto mode = mode_of(q.recovery_mode);
    if (!mode) return mode.error();
    if (q.container != "aces" && q.container != "linear")
        return make_error(ErrorCode::InvalidArgument, "container must be aces or linear", q.container);

    // Full resolution, untiled: the Studio's master is one pass over the frame.
    auto decoded = decode_sdr_file(image);
    if (!decoded) return decoded.error();
    auto fr = infer_frame(backend, decoded->rgb, TileConfig{0, 0});
    if (!fr) return fr.error();

    const ModelConstants model{pkg.log_scale, pkg.max_hdr, pkg.corpus_ev};
    CompositeParams cp;
    cp.mode = *mode;
    cp.strength = static_cast<float>(q.strength);
    cp.preserve_outside = q.preserve_outside;
    const NetworkLinearImage network = composite(decoded->rgb, fr->fields, fr->scalars, model, cp);

    // The master chain, stage by stage (render_master_pixels without the
    // container step: write_aces_exr does that conversion itself).
    std::vector<RegionBand> bands;
    for (const auto& b : q.regions) bands.push_back({b.low_nits, b.high_nits, b.ev});
    if (bands.empty()) bands = default_region_bands();
    const bool graded = any_graded(bands);
    NitsFrame nits = nits_from_network(network);
    apply_region_ev(nits, bands, q.region_softness_stops, double(model.max_hdr) * 10000.0);
    if (q.anchor) anchor_to_sdr(nits, decoded->rgb, q.anchor_knee);
    if (q.carry_chroma) carry_source_chroma(nits, decoded->rgb, q.chroma_knee);
    const PlanarBuffer linear = scene_linear(nits);

    const FrameStats stats = analyze_frame(nits, 0);
    const StaticMetadata md = maxcll_maxfall(std::span(&stats, 1));
    double peak = -INFINITY;
    for (double v : nits.span()) peak = std::max(peak, v);

    std::string region_text = "neutral";
    if (graded) {
        pyjson::List l;
        for (const auto& b : q.regions) l.push_back(band_json(b));
        region_text = pyjson::dumps(l);
    }
    const ExrAttributes provenance{
        {"rudra:checkpoint", q.checkpoint},
        {"rudra:maxCLL", std::to_string(md.maxcll)},
        {"rudra:maxFALL", std::to_string(md.maxfall)},
        {"rudra:recoveryMode", q.recovery_mode},
        {"rudra:preserveOutside", q.preserve_outside ? "True" : "False"},
        {"rudra:regionEV", region_text},
        {"rudra:tiled", "False"},
        {"rudra:anchored", q.anchor ? "True" : "False"},
        {"rudra:chromaCarried", q.carry_chroma ? "True" : "False"},
        {"rudra:sourceSpace", q.source_space},
    };
    const bool aces = q.container == "aces";
    auto w = aces ? write_aces_exr(out, linear, *src, 1.0, provenance) : write_exr(out, linear, true, std::nullopt, provenance);
    if (!w) return w.error();

    pyjson::List regions_json;
    for (const auto& b : q.regions) regions_json.push_back(band_json(b));
    const pyjson::Value sidecar = pyjson::Dict{
        {"maxcll_nits", md.maxcll},
        {"maxfall_nits", md.maxfall},
        {"peak_nits", py_round1(peak)},
        {"resolution", pyjson::List{decoded->rgb.width(), decoded->rgb.height()}},
        {"source_bits", decoded->bits},
        // The Studio labels its linear container Rec.2020 while writing the
        // source primaries untouched; kept as is so the two sidecars agree.
        {"container", aces ? "ACES 2065-1 (AP0)" : "scene-linear Rec.2020"},
        {"transfer", "linear"},
        {"diffuse_white_nits", 203.0},
        {"checkpoint", q.checkpoint},
        {"recovery_mode", q.recovery_mode},
        {"residual_strength", q.strength},
        {"preserve_outside", q.preserve_outside},
        {"region_ev", graded ? pyjson::Value(regions_json) : pyjson::Value(nullptr)},
        {"region_softness_stops", graded ? pyjson::Value(q.region_softness_stops) : pyjson::Value(nullptr)},
        {"tiled", false},
    };
    std::filesystem::path side = out;
    side.replace_extension(".json");
    if (auto s = write_text_file(side, pyjson::dumps(sidecar, 2)); !s) return s.error();

    MasterResult r;
    r.maxcll = md.maxcll;
    r.maxfall = md.maxfall;
    r.peak_nits = py_round1(peak);
    r.width = decoded->rgb.width();
    r.height = decoded->rgb.height();
    r.source_bits = decoded->bits;
    r.exr = out;
    r.sidecar = side;
    return r;
}

}  // namespace rudra
