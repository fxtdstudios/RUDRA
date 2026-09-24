// Phase 3 steps 7 and 8: the probe, the Frame panel, the pipeline bar and the
// clip bar say what the page says.
//
// tests/golden/readouts/readouts.json is the page's own writers (showProbe,
// showMetrics, updatePipe, paintClipBar) run headless on the inputs it records
// (tools/emit_readouts_golden.py). core/readouts.cpp must write the same words,
// classes and widths from the same inputs.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/readouts.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/readouts/readouts.json");
        return json::parse(f);
    }();
    return g;
}

double num(const json& v) {
    return v.is_string() ? std::numeric_limits<double>::quiet_NaN() : v.get<double>();   // "NaN"
}

std::optional<FrameHeader> header_of(const json& h) {
    if (h.is_null()) return std::nullopt;
    FrameHeader f;
    if (h.contains("source_resolution")) f.source_resolution = h["source_resolution"].get<std::string>();
    if (h.contains("resolution")) f.resolution = h["resolution"].get<std::string>();
    if (h.contains("elapsed_s")) f.elapsed_s = h["elapsed_s"].get<double>();
    if (h.contains("tiled")) f.tiled = h["tiled"].get<bool>();
    if (h.contains("source_bits")) f.source_bits = h["source_bits"].get<int>();
    return f;
}

}  // namespace

TEST(Readouts, ProbePanelAndBoxAreThePages) {
    const auto& in = golden()["inputs"]["probes"];
    const auto& page = golden()["page"]["probes"];
    ASSERT_EQ(in.size(), page.size());
    for (std::size_t i = 0; i < in.size(); ++i) {
        std::optional<ProbeInput> p;
        if (!in[i].is_null()) {
            ProbeInput q;
            q.x = in[i]["x"];
            q.y = in[i]["y"];
            q.model_nits = in[i]["model"]["nits"];
            q.baseline_nits = in[i]["baseline"]["nits"];
            if (!in[i]["sdr"].is_null()) q.sdr = in[i]["sdr"].get<std::array<int, 3>>();
            if (!in[i]["hiMask"].is_null()) q.hi_mask = in[i]["hiMask"].get<double>();
            if (!in[i]["shMask"].is_null()) q.sh_mask = in[i]["shMask"].get<double>();
            p = q;
        }
        const auto t = probe_panel(p);
        const auto& g = page[i];
        const std::string w = "probe " + std::to_string(i);
        EXPECT_EQ(t.xy, g["xy"].get<std::string>()) << w;
        EXPECT_EQ(t.nits, g["nits"].get<std::string>()) << w;
        EXPECT_EQ(t.idle, g["idle"].get<bool>()) << w;
        EXPECT_EQ(t.delta, g["delta"].get<std::string>()) << w;
        EXPECT_EQ(t.delta_class, g["delta_class"].get<std::string>()) << w;
        EXPECT_EQ(t.src, g["src"].get<std::string>()) << w;
        EXPECT_EQ(t.src_class, g["src_class"].get<std::string>()) << w;
        EXPECT_EQ(t.base, g["base"].get<std::string>()) << w;
        EXPECT_EQ(t.model, g["model"].get<std::string>()) << w;
        EXPECT_EQ(t.model_class, g["model_class"].get<std::string>()) << w;
        EXPECT_EQ(t.mask, g["mask"].get<std::string>()) << w;
        EXPECT_EQ(!p.has_value(), g["box_hidden"].get<bool>()) << w;
        if (!p) continue;
        const auto rows = probe_box(*p);
        ASSERT_EQ(rows.size(), g["box"].size()) << w;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            EXPECT_EQ(rows[r].k, g["box"][r]["k"].get<std::string>()) << w;
            EXPECT_EQ(rows[r].v, g["box"][r]["v"].get<std::string>()) << w << " " << rows[r].k;
            EXPECT_EQ(rows[r].cls, g["box"][r]["cls"].get<std::string>()) << w << " " << rows[r].k;
        }
    }
}

TEST(Readouts, FramePanelIsThePages) {
    const auto& in = golden()["inputs"]["metrics"];
    const auto& page = golden()["page"]["metrics"];
    for (std::size_t i = 0; i < in.size(); ++i) {
        const auto& m = in[i]["m"];
        FrameMetrics f;
        f.maxcll = num(m["maxcll"]);
        f.maxfall = num(m["maxfall"]);
        f.peak_nits = num(m["peak_nits"]);
        f.p99_nits = num(m["p99_nits"]);
        f.median_nits = num(m["median_nits"]);
        f.above_diffuse_white_pct = num(m["above_diffuse_white_pct"]);
        f.above_1000_nits_pct = num(m["above_1000_nits_pct"]);
        f.headroom_highlight_stops = num(m["headroom_highlight_stops"]);
        f.headroom_shadow_stops = num(m["headroom_shadow_stops"]);
        f.departure_rms_stops = num(m["departure_rms_stops"]);
        f.highlight_mask_pct = num(m["highlight_mask_pct"]);
        f.shadow_mask_pct = num(m["shadow_mask_pct"]);
        f.compose_ms = num(m["compose_ms"]);
        const auto t = metrics_text(f, header_of(in[i]["header"]), in[i]["clipped"].get<double>());
        const auto& g = page[i];
        const std::string w = "frame " + std::to_string(i);
        for (const auto& [ours, want] : {std::pair{&t.a, &g["measA"]}, std::pair{&t.b, &g["measB"]}}) {
            ASSERT_EQ(ours->size(), want->size()) << w;
            for (std::size_t r = 0; r < ours->size(); ++r) {
                EXPECT_EQ((*ours)[r].k, (*want)[r]["k"].get<std::string>()) << w;
                EXPECT_EQ((*ours)[r].v, (*want)[r]["v"].get<std::string>()) << w << " " << (*ours)[r].k;
                EXPECT_EQ((*ours)[r].u, (*want)[r]["u"].get<std::string>()) << w;
                EXPECT_EQ((*ours)[r].warn, (*want)[r]["warn"].get<bool>()) << w;
            }
        }
        EXPECT_EQ(t.status_mask, g["statusMask"].get<std::string>()) << w;
        EXPECT_EQ(t.status_time, g["statusTime"].get<std::string>()) << w;
        EXPECT_EQ(t.src_info, g["srcInfo"].get<std::string>()) << w;
    }
}

TEST(Readouts, PipelineBarIsThePages) {
    const auto& in = golden()["inputs"]["pipes"];
    const auto& page = golden()["page"]["pipes"];
    for (std::size_t i = 0; i < in.size(); ++i) {
        const double display = 203.0 * std::pow(2.0, in[i]["peakEv"].get<double>());
        std::optional<double> maxcll;
        if (!in[i]["m"].is_null()) maxcll = num(in[i]["m"]["maxcll"]);
        const auto t = pipe_text(in[i]["container"], display, maxcll, header_of(in[i]["header"]));
        const auto& g = page[i];
        const std::string w = "pipe " + std::to_string(i);
        EXPECT_EQ(t.in, g["pipeIn"].get<std::string>()) << w;
        EXPECT_EQ(t.working, g["pipeWorking"].get<std::string>()) << w;
        EXPECT_EQ(t.view, g["viewTransform"].get<std::string>()) << w;
        EXPECT_EQ(t.master, g["pipeMaster"].get<std::string>()) << w;
        EXPECT_EQ(t.warn_shown, !g["warn_hidden"].get<bool>()) << w;
        if (t.warn_shown) {
            EXPECT_EQ(t.warn, g["warn"].get<std::string>()) << w;
        }
    }
}

TEST(Readouts, ClipBarIsThePages) {
    const auto& in = golden()["inputs"]["clips"];
    const auto& page = golden()["page"]["clips"];
    for (std::size_t i = 0; i < in.size(); ++i) {
        const auto t = clip_bar(in[i][0].get<double>(), in[i][1].get<double>());
        EXPECT_EQ(t.i_width, page[i]["i_width"].get<std::string>()) << i;
        EXPECT_EQ(t.u_left, page[i]["u_left"].get<std::string>()) << i;
        EXPECT_EQ(t.u_width, page[i]["u_width"].get<std::string>()) << i;
    }
}

// Phase 3 step 10: the HDR card the first-run wizard shows, and what it says.
#include "rudra/core/hdr_card.hpp"

TEST(HdrCard, SixPatchesOnGreyAsGateBDrawsThem) {
    const auto card = hdr_card();
    const auto& b = card.buffer();
    ASSERT_EQ(b.width(), kCardWidth);
    ASSERT_EQ(b.height(), kCardHeight);
    for (int p = 0; p < 6; ++p) {
        const int x = (2 * p + 1) * kCardWidth / 12;
        EXPECT_EQ(card_patch(x, kCardWidth), p);
        for (int c = 0; c < 3; ++c) EXPECT_EQ(b.at(c, 100, x), kCardNits[std::size_t(p)] / 10000.0f);
    }
    EXPECT_EQ(b.at(0, kCardHeight - 1, 5), 0.18f * 203.0f / 10000.0f);
}

TEST(HdrCard, TheWizardSaysWhatTheDisplayCanShow) {
    const auto sdr = display_report(DisplayTarget::sdr(), "SDR", "swapchain");
    EXPECT_FALSE(sdr.hdr);
    EXPECT_EQ(sdr.headline, "SDR display");
    EXPECT_NE(sdr.detail.find("simulated on a 203-nit display"), std::string::npos);

    const auto pa = display_report(DisplayTarget::scrgb(418.0), "scRGB", "DXGI");
    EXPECT_TRUE(pa.hdr);
    EXPECT_EQ(pa.headline, "HDR display, 418 nits");
    EXPECT_EQ(pa.detail, "scRGB, peak from DXGI. Each patch shows its own luminance up to 418 nits; the 600, "
                         "1,000 and 2,000 nit patches clip there, never tone-mapped.");

    const auto xdr = display_report(DisplayTarget::edr(1600.0), "EDR", "swapchain");
    EXPECT_EQ(xdr.headline, "HDR display, 1,600 nits");
    EXPECT_NE(xdr.detail.find("the 2,000 nit patches clip"), std::string::npos);

    const auto big = display_report(DisplayTarget::hdr10(4000.0), "HDR10", "swapchain");
    EXPECT_EQ(big.detail, "HDR10, peak from swapchain. Each patch shows its own luminance up to 4,000 nits.");

    const auto dim = display_report(DisplayTarget::scrgb(300.0), "scRGB", "DXGI");
    EXPECT_FALSE(dim.hdr);
    EXPECT_EQ(dim.headline, "HDR display with little headroom, 300 nits");

    const auto ph = display_report(DisplayTarget::scrgb(1000.0), "scRGB", "placeholder");
    EXPECT_FALSE(ph.hdr);
    EXPECT_EQ(ph.headline, "HDR display (scRGB), peak unknown");
}
