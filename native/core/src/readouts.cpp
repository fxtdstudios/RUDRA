#include "rudra/core/readouts.hpp"

#include <algorithm>
#include <cmath>

#include "rudra/core/js_format.hpp"

namespace rudra {
namespace {

constexpr double kDiffuseWhite = 203.0;
constexpr const char* kMinus = "−";
constexpr const char* kDash = "—";

// stops(nits): from diffuse white.
double stops(double nits) { return std::log2(std::max(nits, 1e-6) / kDiffuseWhite); }

// Math.round(n).toLocaleString("en-US") and fmt(Math.round(n), 0).
std::string whole(double n) { return js_fmt(js_round(n), 0); }

std::string join(const std::array<int, 3>& v, const char* sep) {
    return std::to_string(v[0]) + sep + std::to_string(v[1]) + sep + std::to_string(v[2]);
}

bool clipped(const ProbeInput& p) { return p.sdr && std::max({(*p.sdr)[0], (*p.sdr)[1], (*p.sdr)[2]}) >= 254; }

std::string mask(const std::optional<double>& v, const char* none) { return v ? js_to_fixed(*v, 2) : none; }

// "+2.60" or "-8.61": toFixed's own minus, a plus when not negative.
std::string plus_fixed(double v, int d) { return (v >= 0 ? "+" : "") + js_to_fixed(v, d); }

// `x || "—"`: an empty string is as absent as a missing one.
std::string or_dash(const std::optional<std::string>& s) { return s && !s->empty() ? *s : std::string(kDash); }

}  // namespace

ProbePanelText probe_panel(const std::optional<ProbeInput>& in) {
    ProbePanelText t;
    if (!in) {
        t.xy = kDash;
        t.nits = kDash;
        t.idle = true;
        t.delta = "pick a pixel with Probe, or hold Alt";
        t.delta_class = "pdelta idle";
        t.src = t.base = t.model = t.mask = kDash;
        return t;
    }
    const ProbeInput& p = *in;
    const double st = stops(p.model_nits), d = st - stops(p.baseline_nits);
    t.idle = false;
    t.xy = std::to_string(p.x) + ", " + std::to_string(p.y);
    t.nits = whole(p.model_nits);
    t.delta = std::string(st >= 0 ? "+" : kMinus) + js_to_fixed(std::abs(st), 2) + " stops over diffuse white";
    t.delta_class = "pdelta";
    t.src = p.sdr ? join(*p.sdr, " ") : kDash;
    t.src_class = clipped(p) ? "clip" : "";
    t.base = whole(p.baseline_nits);
    t.model = whole(p.model_nits);
    t.model_class = std::abs(d) > 0.01 ? "hi" : "";
    t.mask = "hi " + mask(p.hi_mask, kDash) + "  sh " + mask(p.sh_mask, kDash);
    return t;
}

std::vector<ProbeRow> probe_box(const ProbeInput& p) {
    const double d = stops(p.model_nits) - stops(p.baseline_nits);
    std::vector<ProbeRow> rows;
    rows.push_back({"x,y", std::to_string(p.x) + ", " + std::to_string(p.y), ""});
    rows.push_back({"baseline", whole(p.baseline_nits) + " nits  " + plus_fixed(stops(p.baseline_nits), 2) + " st", ""});
    rows.push_back({"RUDRA", whole(p.model_nits) + " nits  " + plus_fixed(stops(p.model_nits), 2) + " st", "hi"});
    rows.push_back({"delta", plus_fixed(d, 2) + " stops", std::abs(d) > 0.01 ? "hi" : ""});
    if (p.sdr) {
        const bool c = clipped(p);
        rows.push_back({"SDR", join(*p.sdr, ",") + (c ? "  clipped" : ""), c ? "bad" : ""});
    }
    rows.push_back({"mask", "hi " + mask(p.hi_mask, "-") + "  sh " + mask(p.sh_mask, "-"), ""});
    return rows;
}

MetricsText metrics_text(const FrameMetrics& m, const std::optional<FrameHeader>& header, double clipped_pct) {
    MetricsText t;
    t.a = {{"MaxCLL", js_fmt(m.maxcll, 0), "nits"},
           {"MaxFALL", js_fmt(m.maxfall, 0), "nits"},
           {"Peak", js_fmt(m.peak_nits, 1), "nits"},
           {"P99", js_fmt(m.p99_nits, 1), "nits"},
           {"Median", js_fmt(m.median_nits, 2), "nits"}};
    t.b = {{"Above diffuse white", js_fmt(m.above_diffuse_white_pct, 2), "%"},
           {"Above 1 000 nits", js_fmt(m.above_1000_nits_pct, 3), "%"},
           {"Clipped in source", js_fmt(clipped_pct, 2), "%", true},
           {"Headroom, highlights", js_signed(m.headroom_highlight_stops, 2), "st"},
           {"Headroom, shadows", js_signed(m.headroom_shadow_stops, 2), "st"},
           {"Departure RMS", js_fmt(m.departure_rms_stops, 3), "st"}};
    t.status_mask = "masks " + js_fmt(m.highlight_mask_pct, 2) + "% highlight / " + js_fmt(m.shadow_mask_pct, 2) +
                    "% shadow";
    const FrameHeader h = header.value_or(FrameHeader{});
    t.status_time = or_dash(h.source_resolution) + " · net " +
                    (h.elapsed_s ? js_fmt(*h.elapsed_s, 2) : std::string(kDash)) + " s · grade " +
                    js_fmt(m.compose_ms, 1) + " ms";
    t.src_info = or_dash(h.resolution) + (h.tiled ? " · tiled" : " · one pass");
    return t;
}

PipeText pipe_text(const std::string& container, double display_nits, std::optional<double> maxcll,
                   const std::optional<FrameHeader>& header) {
    PipeText t;
    const FrameHeader h = header.value_or(FrameHeader{});
    t.in = (h.source_bits && *h.source_bits ? std::to_string(*h.source_bits) + "-bit " : std::string()) +
           "sRGB · Rec.709";
    t.working = "scene-linear · 203 nits = 1.0";
    t.master = container != "linear" ? "ACES 2065-1 EXR, half" : "linear Rec.2020 EXR, half";
    t.view = "exposure + clip · " + js_fmt(js_round(display_nits), 0) + " nits";
    const bool has = maxcll && std::isfinite(*maxcll);
    if (has && *maxcll > display_nits * 1.001) {
        t.warn = "MaxCLL " + js_fmt(*maxcll, 0) + " over view peak " + js_fmt(display_nits, 0) +
                 " — clipped on screen, not in the master";
        t.warn_shown = true;
    }
    return t;
}

ClipBarText clip_bar(double clipped_pct, double mask_pct) {
    const double i = std::min(100.0, clipped_pct);
    const double u = std::max(0.0, std::min(100.0 - clipped_pct, mask_pct));
    return {js_number(i) + "%", js_number(i) + "%", js_number(u) + "%"};
}

}  // namespace rudra
