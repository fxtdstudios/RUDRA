// The composite, the master chain and the measurements against the Python they
// port (goldens from tools/emit_composite_golden.py).

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/composite.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/master.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "composite";

const json& index_json() {
    static const json j = [] {
        std::ifstream in(kDir / "index.json");
        return json::parse(in);
    }();
    return j;
}

NpyArray f32(const json& entry) {
    auto a = read_npy(kDir / entry.at("file").get<std::string>());
    EXPECT_TRUE(a) << entry.dump();
    return a ? std::move(*a) : NpyArray{};
}

NpyArrayF64 f64(const json& entry) {
    auto a = read_npy_f64(kDir / entry.at("file").get<std::string>());
    EXPECT_TRUE(a) << entry.dump();
    return a ? std::move(*a) : NpyArrayF64{};
}

// (C, H, W) float32 golden -> planar buffer.
PlanarBuffer planar(const NpyArray& a) {
    return PlanarBuffer(static_cast<int>(a.shape[0]), static_cast<int>(a.shape[1]),
                        static_cast<int>(a.shape[2]), a.data);
}

NitsFrame nits_frame(const NpyArrayF64& a) {
    NitsFrame f(static_cast<int>(a.shape[1]), static_cast<int>(a.shape[2]));
    std::copy(a.data.begin(), a.data.end(), f.span().begin());
    return f;
}

template <class A, class B>
void expect_close(const A& got, const B& want, double rtol, double atol, const std::string& what) {
    ASSERT_EQ(std::size(got), std::size(want)) << what;
    double worst = 0.0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < std::size(want); ++i) {
        const double g = double(got[i]), w = double(want[i]);
        const double excess = std::abs(g - w) - (atol + rtol * std::abs(w));
        if (excess > worst) { worst = excess; at = i; }
    }
    EXPECT_LE(worst, 0.0) << what << ": worst at " << at << " got " << double(got[at]) << " want " << double(want[at]);
}

ModelConstants model_constants() {
    const auto& m = index_json().at("model");
    return {m.at("log_scale").get<float>(), m.at("max_hdr").get<float>(), m.at("corpus_ev").get<float>()};
}

std::vector<RegionBand> bands(const json& j) {
    std::vector<RegionBand> out;
    for (const auto& b : j) out.push_back({b.at("low_nits").get<double>(), b.at("high_nits").get<double>(), b.at("ev").get<double>()});
    return out;
}

RecoveryMode mode_of(const std::string& s) {
    if (s == "highlights") return RecoveryMode::Highlights;
    if (s == "shadows") return RecoveryMode::Shadows;
    if (s == "off") return RecoveryMode::Off;
    return RecoveryMode::All;
}

struct Frame {
    SdrImage sdr;
    Fields fields;
    FrameScalars scalars;
};

Frame load_frame(const json& f) {
    Frame fr{SdrImage(planar(f32(f.at("sdr")))),
             Fields{planar(f32(f.at("residual"))), planar(f32(f.at("highlight"))), planar(f32(f.at("shadow")))},
             FrameScalars{}};
    fr.scalars.shadow_weight = f.at("shadow_weight").get<float>();
    fr.scalars.curve_params = f.at("curve_params").get<std::vector<float>>();
    return fr;
}

class PerFrame : public ::testing::TestWithParam<std::string> {
protected:
    const json& frame() const { return index_json().at("frames").at(GetParam()); }
};

}  // namespace

// predict_image is fed the whole network; the composite is fed the fields
// predict_fields returned. They agree to ~3e-6 before expm1 amplifies it
// (tests/test_frame_fields_2026_08_28.py), so this is a relative bound.
TEST_P(PerFrame, CompositeMatchesPredictImage) {
    const Frame fr = load_frame(frame());
    for (const auto& c : frame().at("composites")) {
        CompositeParams p;
        p.mode = mode_of(c.at("mode").get<std::string>());
        p.strength = c.at("strength").get<float>();
        p.preserve_outside = c.at("preserve").get<bool>();
        const auto out = composite(fr.sdr, fr.fields, fr.scalars, model_constants(), p);
        expect_close(out.buffer().span(), f32(c.at("expected")).data, 2e-4, 1e-6, c.at("expected").at("file"));
    }
}

TEST_P(PerFrame, CompositeRegionEvMatchesMasterStage) {
    const Frame fr = load_frame(frame());
    const auto& m = frame().at("master");
    for (const char* key : {"region_soft1", "region_soft0.5"}) {
        CompositeParams p;
        p.regions = bands(m.at("bands"));
        p.region_softness_stops = std::string(key) == "region_soft1" ? 1.0 : 0.5;
        const auto out = composite(fr.sdr, fr.fields, fr.scalars, model_constants(), p);
        std::vector<double> want = f64(m.at("stages").at(key)).data;
        for (double& v : want) v /= 10000.0;
        expect_close(out.buffer().span(), want, 2e-4, 1e-6, key);
    }
}

TEST_P(PerFrame, NeutralRegionsAreANoOp) {
    const Frame fr = load_frame(frame());
    CompositeParams plain, neutral;
    neutral.regions = default_region_bands();
    const auto a = composite(fr.sdr, fr.fields, fr.scalars, model_constants(), plain);
    const auto b = composite(fr.sdr, fr.fields, fr.scalars, model_constants(), neutral);
    EXPECT_TRUE(std::equal(a.buffer().span().begin(), a.buffer().span().end(), b.buffer().span().begin()));
}

// Each master stage on its predecessor's golden: a failure names its stage.
TEST_P(PerFrame, MasterStages) {
    const auto& m = frame().at("master");
    const auto& st = m.at("stages");
    const SdrImage sdr(planar(f32(frame().at("sdr"))));
    const double ceiling = double(model_constants().max_hdr) * 10000.0;
    const auto network = NetworkLinearImage(planar(f32(frame().at("composites").at(0).at("expected"))));

    for (const char* key : {"region_soft1", "region_soft0.5"}) {
        NitsFrame nits = nits_from_network(network);
        apply_region_ev(nits, bands(m.at("bands")), std::string(key) == "region_soft1" ? 1.0 : 0.5, ceiling);
        expect_close(nits.span(), f64(st.at(key)).data, 1e-12, 1e-9, key);
    }

    NitsFrame nits = nits_frame(f64(st.at("region_soft1")));
    anchor_to_sdr(nits, sdr, m.at("anchor_knee").get<double>());
    expect_close(nits.span(), f64(st.at("anchored")).data, 1e-10, 1e-9, "anchored");

    nits = nits_frame(f64(st.at("anchored")));
    carry_source_chroma(nits, sdr, m.at("chroma_knee").get<double>());
    // The blurred ramp is float32 in OpenCV and summed in a SIMD order; the
    // port sums in double. 1e-6 of the ramp, scaled by the chroma difference.
    expect_close(nits.span(), f64(st.at("carried")).data, 1e-5, 1e-6, "carried");

    nits = nits_frame(f64(st.at("carried")));
    settle_highlight_grain(nits, sdr, m.at("anchor_knee").get<double>());
    // OpenCV's box and Gaussian filters sum in their own order; the port in
    // plain double loops. What moves is the flatness weight, by ~1e-12.
    expect_close(nits.span(), f64(st.at("settled")).data, 1e-9, 1e-9, "settled");

    nits = nits_frame(f64(st.at("settled")));
    const PlanarBuffer lin = scene_linear(nits);
    expect_close(lin.span(), f32(st.at("scene_linear")).data, 0.0, 0.0, "scene_linear");
    const PlanarBuffer ap0 = convert_primaries(lin, Primaries::Rec709, Primaries::Ap0);
    expect_close(ap0.span(), f32(st.at("aces_ap0")).data, 1e-7, 1e-9, "aces_ap0");
}

TEST(Grain, BelowTheKneeAndOnStructureNothingMoves) {
    // A flat grey below the knee, a flat near-white with one-code grain above
    // it, and a hard edge in that near-white: only the grain may move.
    const int h = 24, w = 48;
    PlanarBuffer codes(3, h, w);
    NitsFrame nits(h, w);
    std::uint32_t seed = 12345;
    auto noise = [&] { seed = seed * 1664525u + 1013904223u; return double(seed >> 8) / double(1u << 24) - 0.5; };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const double code = x < 16 ? 0.5 : (x < 40 ? 0.97 : 0.6);   // an edge at x = 40
            const double grain = x < 16 ? 0.0 : noise() * 2.0 / 255.0;
            const double level = (x < 16 ? 50.0 : (x < 40 ? 1000.0 : 120.0)) * (1.0 + (x >= 16 && x < 40 ? noise() * 0.1 : 0.0));
            for (int c = 0; c < 3; ++c) {
                codes.at(c, y, x) = static_cast<float>(std::clamp(code + grain, 0.0, 1.0));
                nits.at(c, y, x) = level;
            }
        }
    const SdrImage sdr(std::move(codes));
    NitsFrame before = nits;
    settle_highlight_grain(nits, sdr);
    double spread_before = 0.0, spread_after = 0.0, mean_before = 0.0, mean_after = 0.0;
    int count = 0;
    for (int y = 4; y < h - 4; ++y)
        for (int x = 0; x < w; ++x) {
            if (x < 16 || x >= 44) {
                for (int c = 0; c < 3; ++c) EXPECT_EQ(nits.at(c, y, x), before.at(c, y, x)) << x << "," << y;
            } else if (x >= 20 && x < 36) {
                mean_before += before.at(1, y, x);
                mean_after += nits.at(1, y, x);
                spread_before += std::abs(before.at(1, y, x) - 1000.0);
                spread_after += std::abs(nits.at(1, y, x) - 1000.0);
                ++count;
            }
            EXPECT_DOUBLE_EQ(nits.at(0, y, x) / nits.at(1, y, x), before.at(0, y, x) / before.at(1, y, x));   // hue
        }
    EXPECT_LT(spread_after, 0.3 * spread_before);
    EXPECT_NEAR(mean_after / count, mean_before / count, 0.01 * mean_before / count);
}

TEST_P(PerFrame, MasterChainEndToEnd) {
    const auto& m = frame().at("master");
    const SdrImage sdr(planar(f32(frame().at("sdr"))));
    const auto network = NetworkLinearImage(planar(f32(frame().at("composites").at(0).at("expected"))));
    MasterParams p;
    p.regions = bands(m.at("bands"));
    p.region_softness_stops = m.at("region_softness").get<double>();
    const MasterPixels out = render_master_pixels(network, sdr, model_constants(), p);
    EXPECT_EQ(out.primaries, Primaries::Ap0);
    expect_close(out.pixels.span(), f32(m.at("stages").at("aces_ap0")).data, 1e-5, 1e-6, "master");
}

TEST_P(PerFrame, AnalyzeFrame) {
    const auto& a = frame().at("analyze");
    const NitsFrame nits = nits_frame(f64(frame().at("master").at("stages").at(a.at("input").get<std::string>())));
    const FrameStats s = analyze_frame(nits);
    const auto& w = a.at("stats");
    EXPECT_NEAR(s.min_nits, w.at("min_nits").get<double>(), 1e-9);
    EXPECT_EQ(s.avg_nits, w.at("avg_nits").get<double>());   // numpy pairwise sum, reproduced
    EXPECT_DOUBLE_EQ(s.max_nits, w.at("max_nits").get<double>());
    expect_close(s.maxscl_nits, w.at("maxscl_nits").get<std::vector<double>>(), 0.0, 0.0, "maxscl");
    expect_close(s.percentiles_nits, w.at("percentiles_nits").get<std::vector<double>>(), 1e-12, 1e-12, "percentiles");
    expect_close(s.log_hist, w.at("log_hist").get<std::vector<double>>(), 0.0, 0.0, "log_hist");
    const StaticMetadata md = maxcll_maxfall(std::span(&s, 1));
    EXPECT_EQ(md.maxcll, a.at("maxcll").get<int>());
    EXPECT_EQ(md.maxfall, a.at("maxfall").get<int>());
}

// ui/server.py measure() rounds for the page; the port returns raw numbers,
// checked here to within the rounding step of each field.
TEST_P(PerFrame, StudioMeasure) {
    const auto& f = frame();
    const auto& want = f.at("measure").at("result");
    const auto hdr = NetworkLinearImage(planar(f32(f.at("composites").at(0).at("expected"))));
    const auto base = NetworkLinearImage(planar(f32(f.at("baseline"))));
    const StudioMeasure r = measure(hdr, base, planar(f32(f.at("highlight"))), planar(f32(f.at("shadow"))));

    auto check = [&](const char* key, double got, int decimals) {
        const auto& w = want.at(key);
        if (w.is_null()) {
            EXPECT_TRUE(std::isnan(got)) << key;
            return;
        }
        EXPECT_NEAR(got, w.get<double>(), 0.5 * std::pow(10.0, -decimals) + 1e-9) << key;
    };
    EXPECT_EQ(r.maxcll, want.at("maxcll").get<int>());
    EXPECT_EQ(r.maxfall, want.at("maxfall").get<int>());
    check("peak_nits", r.peak_nits, 1);
    check("baseline_peak_nits", r.baseline_peak_nits, 1);
    check("headroom_stops", r.headroom_stops, 2);
    check("headroom_highlight_stops", r.headroom_highlight_stops, 2);
    check("headroom_shadow_stops", r.headroom_shadow_stops, 2);
    check("departure_rms_stops", r.departure_rms_stops, 3);
    check("p99_nits", r.p99_nits, 1);
    check("median_nits", r.median_nits, 2);
    check("above_diffuse_white_pct", r.above_diffuse_white_pct, 2);
    check("above_1000_nits_pct", r.above_1000_nits_pct, 3);
    check("highlight_mask_pct", r.highlight_mask_pct, 2);
    check("shadow_mask_pct", r.shadow_mask_pct, 2);
}

INSTANTIATE_TEST_SUITE_P(Golden, PerFrame, ::testing::Values("main", "small"));

TEST(Gamut, MatricesMatchColorspacePy) {
    auto prim = [](const std::string& s) {
        if (s == "rec709") return Primaries::Rec709;
        if (s == "rec2020") return Primaries::Rec2020;
        if (s == "p3d65") return Primaries::P3D65;
        if (s == "ap0") return Primaries::Ap0;
        return Primaries::Ap1;
    };
    for (const auto& e : index_json().at("matrices")) {
        const Mat3 m = rgb_to_rgb_matrix(prim(e.at("src")), prim(e.at("dst")));
        const auto want = e.at("m").get<std::vector<std::vector<double>>>();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                EXPECT_NEAR(m[i][j], want[i][j], 1e-13) << e.at("src") << "->" << e.at("dst");
    }
}

TEST(Gamut, IdentityAndRoundTrip) {
    const Mat3 a = rgb_to_rgb_matrix(Primaries::Rec2020, Primaries::Ap0);
    const Mat3 b = rgb_to_rgb_matrix(Primaries::Ap0, Primaries::Rec2020);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += b[i][k] * a[k][j];
            EXPECT_NEAR(s, i == j ? 1.0 : 0.0, 1e-12);
        }
    // White stays white across a white-point change (Bradford).
    const Mat3 m = rgb_to_rgb_matrix(Primaries::Rec709, Primaries::Ap0);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(m[i][0] + m[i][1] + m[i][2], 1.0, 1e-4);
}

TEST(Blur, KernelSizeAndFlatField) {
    // A flat field stays flat under replicate borders, and a unit impulse sums to one.
    std::vector<float> flat(20 * 30, 0.25f);
    for (float v : gaussian_blur_replicate(flat, 20, 30, 2.0)) EXPECT_NEAR(v, 0.25f, 1e-6f);
    std::vector<float> imp(41 * 41, 0.0f);
    imp[20 * 41 + 20] = 1.0f;
    const auto out = gaussian_blur_replicate(imp, 41, 41, 2.0);
    double s = 0.0;
    for (float v : out) s += v;
    EXPECT_NEAR(s, 1.0, 1e-5);
    EXPECT_EQ(out[20 * 41 + 20 - 9], 0.0f);   // 17 taps: radius 8
    EXPECT_GT(out[20 * 41 + 20 - 8], 0.0f);
}
