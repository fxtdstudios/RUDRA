#include "rudra/deliver/master.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <cstdlib>
#include <random>
#include <regex>
#include <system_error>

#include <nlohmann/json.hpp>

#include "rudra/core/master.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/sidecars.hpp"
#include "rudra/platform/pyjson.hpp"

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

std::string master_request_json(const MasterRequest& q) {
    nlohmann::json j{{"checkpoint", q.checkpoint},     {"preserve_outside", q.preserve_outside},
                     {"recovery_mode", q.recovery_mode}, {"strength", q.strength},
                     {"region_softness_stops", q.region_softness_stops}, {"anchor", q.anchor},
                     {"anchor_knee", q.anchor_knee},     {"carry_chroma", q.carry_chroma},
                     {"chroma_knee", q.chroma_knee},     {"source_space", q.source_space},
                     {"container", q.container}};
    auto regions = nlohmann::json::array();
    for (const auto& b : q.regions)
        regions.push_back({{"label", b.label}, {"low_nits", b.low_nits}, {"high_nits", b.high_nits}, {"ev", b.ev}});
    j["regions"] = regions;
    return j.dump();
}

Result<MasterResult> write_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                  const FrameScalars& scalars, const ModelConstants& model, const MasterRequest& q,
                                  const std::filesystem::path& out) {
    auto src = primaries_of(q.source_space);
    if (!src) return src.error();
    auto mode = mode_of(q.recovery_mode);
    if (!mode) return mode.error();
    if (q.container != "aces" && q.container != "linear")
        return make_error(ErrorCode::InvalidArgument, "container must be aces or linear", q.container);

    CompositeParams cp;
    cp.mode = *mode;
    cp.strength = static_cast<float>(q.strength);
    cp.preserve_outside = q.preserve_outside;
    const NetworkLinearImage network = composite(sdr, fields, scalars, model, cp);

    // The master chain, stage by stage (render_master_pixels without the
    // container step: write_aces_exr does that conversion itself).
    std::vector<RegionBand> bands;
    for (const auto& b : q.regions) bands.push_back({b.low_nits, b.high_nits, b.ev});
    if (bands.empty()) bands = default_region_bands();
    const bool graded = any_graded(bands);
    NitsFrame nits = nits_from_network(network);
    apply_region_ev(nits, bands, q.region_softness_stops, double(model.max_hdr) * 10000.0);
    if (q.anchor) anchor_to_sdr(nits, sdr, q.anchor_knee);
    if (q.carry_chroma) carry_source_chroma(nits, sdr, q.chroma_knee);
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
        {"resolution", pyjson::List{sdr.width(), sdr.height()}},
        {"source_bits", source_bits},
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
    r.width = sdr.width();
    r.height = sdr.height();
    r.source_bits = source_bits;
    r.exr = out;
    r.sidecar = side;
    return r;
}

namespace {

std::string strip(const std::string& s) {
    const auto a = s.find_first_not_of(" \t\r\n\f\v");
    if (a == std::string::npos) return {};
    const auto b = s.find_last_not_of(" \t\r\n\f\v");
    return s.substr(a, b - a + 1);
}

// Path.expanduser(): a leading ~ is the home folder.
std::filesystem::path expand_user(const std::string& s) {
    if (s.empty() || s[0] != '~' || (s.size() > 1 && s[1] != '/' && s[1] != '\\')) return std::filesystem::path(s);
    const char* home = std::getenv("HOME");
#ifdef _WIN32
    if (!home) home = std::getenv("USERPROFILE");
#endif
    return home ? std::filesystem::path(home) / std::filesystem::path(s.size() > 2 ? s.substr(2) : std::string())
                : std::filesystem::path(s);
}

std::string six(int v) {
    char b[16];
    std::snprintf(b, sizeof b, "%06d", v);
    return b;
}

}  // namespace

Result<std::vector<std::filesystem::path>> master_targets(const RenderPlan& plan) {
    namespace fs = std::filesystem;
    fs::path folder = expand_user(strip(plan.render_dir));
    if (strip(plan.render_dir).empty() || !folder.is_absolute())
        return make_error(ErrorCode::InvalidArgument, "Choose an absolute render folder on the Studio computer");
    std::error_code ec;
    const fs::path resolved = fs::weakly_canonical(folder, ec);   // Path.resolve(): links followed, . and .. gone
    if (!ec) folder = resolved;
    const std::string name = strip(plan.render_name);
    static const std::regex bad("[^A-Za-z0-9_.-]");
    if (name.empty() || name == "." || name == ".." || std::regex_search(name, bad))
        return make_error(ErrorCode::InvalidArgument,
                          "Render name must contain only letters, numbers, dots, underscores or hyphens");
    const long long count = plan.render_count, start = plan.frame_start;
    if (count < 1 || count > 100000 || start < 0 || start + count > 100000000)
        return make_error(ErrorCode::InvalidArgument, "Invalid frame range");
    if (!plan.sequence && count != 1) return make_error(ErrorCode::InvalidArgument, "Image render requires one frame");
    std::vector<fs::path> targets;
    for (long long i = 0; i < count; ++i)
        targets.push_back(folder / (plan.sequence ? name + "." + six(int(start + i)) + ".exr" : name + ".exr"));
    for (const auto& out : targets) {
        fs::path side = out;
        side.replace_extension(".json");
        if (fs::exists(out) || fs::exists(side))
            return make_error(ErrorCode::InvalidArgument, "Refusing to overwrite existing render: " + out.string());
    }
    return targets;
}

Result<MasterResult> publish_master(const SdrImage& sdr, int source_bits, const Fields& fields,
                                    const FrameScalars& scalars, const ModelConstants& model,
                                    const MasterRequest& request, const std::filesystem::path& out) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(out.parent_path(), ec);
    if (ec) return make_error(ErrorCode::IoError, "Cannot create the render folder.", ec.message());
    // Stage both files, then publish without replacing any existing file.
    std::random_device rd;
    fs::path staging;
    for (int attempt = 0; attempt < 16; ++attempt) {
        char tag[16];
        std::snprintf(tag, sizeof tag, "%08x", unsigned(rd()));
        staging = out.parent_path() / (std::string(".rudra-render-") + tag);
        if (fs::create_directory(staging, ec) && !ec) break;
        staging.clear();
    }
    if (staging.empty()) return make_error(ErrorCode::IoError, "Cannot stage the render.", out.parent_path().string());
    struct Cleanup {
        fs::path dir;
        ~Cleanup() {
            std::error_code e;
            fs::remove_all(dir, e);
        }
    } cleanup{staging};
    const fs::path staged = staging / out.filename();
    auto r = write_master(sdr, source_bits, fields, scalars, model, request, staged);
    if (!r) return r.error();
    fs::path staged_side = staged, side = out;
    staged_side.replace_extension(".json");
    side.replace_extension(".json");
    // A hard link fails if the name exists: nothing is ever replaced. Where
    // the file system has no links, a copy that refuses to overwrite.
    auto place = [](const fs::path& from, const fs::path& to) -> std::error_code {
        std::error_code e;
        fs::create_hard_link(from, to, e);
        if (e && !fs::exists(to)) {
            e.clear();
            fs::copy_file(from, to, fs::copy_options::none, e);
        }
        return e;
    };
    if (auto e = place(staged, out))
        return make_error(ErrorCode::IoError, "Refusing to overwrite existing render: " + out.string(), e.message());
    if (auto e = place(staged_side, side)) {
        fs::remove(out, ec);
        return make_error(ErrorCode::IoError, "Refusing to overwrite existing render: " + side.string(), e.message());
    }
    r->exr = out;
    r->sidecar = side;
    return r;
}

}  // namespace rudra
