// The browser Studio's viewer as the oracle (goldens from tools/emit_viewer_golden.py,
// which runs ui/compositor.js and ui/app.js unmodified in headless Chromium).
//
// Phase 2 step 1: the frames go in exactly as /api/frame sends them, and the
// browser's own composite readback must match composite.cpp on the same
// inputs within the gpu_fp32 bound, as must its exact GPU reductions (peak,
// mean) against the same reductions on the CPU.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/half.hpp"
#include "rudra/core/scopes.hpp"
#include "rudra/core/view.hpp"
#include "rudra/engine/measure.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/platform/npy.hpp"

using namespace rudra;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "viewer";
constexpr double kPeakNits = 10000.0;

const json& index_json() {
    static const json j = [] {
        std::ifstream in(kDir / "index.json");
        return json::parse(in);
    }();
    return j;
}

struct Packet {
    int w = 0, h = 0;
    SdrImage sdr;
    Fields fields;
    FrameScalars scalars;
    ModelConstants model;
};

// ui/server.py's body: fields H*W*4 half, shadow H*W half, sdr H*W*3 uint8.
Packet unpack(const json& frame) {
    const json& hd = frame.at("header");
    Packet p;
    p.w = hd.at("width").get<int>();
    p.h = hd.at("height").get<int>();
    const std::size_t n = std::size_t(p.w) * std::size_t(p.h);
    std::ifstream in(kDir / frame.at("body").get<std::string>(), std::ios::binary);
    const std::vector<unsigned char> body((std::istreambuf_iterator<char>(in)), {});
    EXPECT_EQ(body.size(), hd.at("offsets").at("total").get<std::size_t>());
    auto half_at = [&](std::size_t byte) { return half_to_float(std::uint16_t(body[byte] | (body[byte + 1] << 8))); };
    const std::size_t off_shadow = hd.at("offsets").at("shadow").get<std::size_t>();
    const std::size_t off_sdr = hd.at("offsets").at("sdr").get<std::size_t>();
    PlanarBuffer sdr(3, p.h, p.w), residual(3, p.h, p.w), highlight(1, p.h, p.w), shadow(1, p.h, p.w);
    for (std::size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            residual.span()[std::size_t(c) * n + i] = half_at(i * 8 + std::size_t(c) * 2);
            sdr.span()[std::size_t(c) * n + i] = float(body[off_sdr + i * 3 + std::size_t(c)]) / 255.0f;
        }
        highlight.span()[i] = half_at(i * 8 + 6);
        shadow.span()[i] = half_at(off_shadow + i * 2);
    }
    p.sdr = SdrImage(std::move(sdr));
    p.fields = Fields{std::move(residual), std::move(highlight), std::move(shadow)};
    p.scalars.shadow_weight = hd.at("shadow_weight").get<float>();
    if (hd.at("curve").is_array()) p.scalars.curve_params = hd.at("curve").get<std::vector<float>>();
    p.model = {hd.at("log_scale").get<float>(), hd.at("max_hdr").get<float>(), hd.at("corpus_ev").get<float>()};
    return p;
}

// (H, W, 3) image-order golden -> planar, to compare with NetworkLinearImage.
std::vector<float> planar_of(const NpyArray& a) {
    const std::size_t h = a.shape[0], w = a.shape[1], n = h * w;
    std::vector<float> out(3 * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t c = 0; c < 3; ++c) out[c * n + i] = a.data[i * 3 + c];
    return out;
}

NpyArray npy(const std::string& name) {
    auto a = read_npy(kDir / name);
    EXPECT_TRUE(a) << name;
    return a ? std::move(*a) : NpyArray{};
}

RecoveryMode mode_of(const std::string& s) {
    if (s == "highlights") return RecoveryMode::Highlights;
    if (s == "shadows") return RecoveryMode::Shadows;
    if (s == "off") return RecoveryMode::Off;
    return RecoveryMode::All;
}

std::vector<RegionBand> bands(const json& j) {
    std::vector<RegionBand> out;
    for (const auto& b : j) out.push_back({b.at("low_nits").get<double>(), b.at("high_nits").get<double>(), b.at("ev").get<double>()});
    return out;
}

// The gpu_fp32 bound of the model-package contract: 5e-5 + 1e-5 |ref|.
void expect_gpu_close(std::span<const float> got, std::span<const float> want, const std::string& what) {
    ASSERT_EQ(got.size(), want.size()) << what;
    double worst = 0.0, at_got = 0.0, at_want = 0.0, max_abs = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        max_abs = std::max(max_abs, std::abs(double(got[i]) - double(want[i])));
        const double excess = std::abs(double(got[i]) - double(want[i])) - (5e-5 + 1e-5 * std::abs(double(want[i])));
        if (excess > worst) { worst = excess; at_got = got[i]; at_want = want[i]; }
    }
    EXPECT_LE(worst, 0.0) << what << ": got " << at_got << " want " << at_want;
    ::testing::Test::RecordProperty(what, std::to_string(max_abs));
}

// Sequential double reductions, independent of the ladder.
struct DoubleSums {
    double peak = 0.0, mean = 0.0;
};
DoubleSums double_sums(std::span<const float> planar, std::size_t n) {
    DoubleSums r;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double m = std::max({double(planar[i]), double(planar[n + i]), double(planar[2 * n + i])});
        r.peak = std::max(r.peak, m);
        sum += m;
    }
    r.peak *= kPeakNits;
    r.mean = sum * kPeakNits / double(n);
    return r;
}

}  // namespace

TEST(Viewer, HalfToFloatIsExact) {
    for (std::uint32_t h = 0; h < 65536; ++h) {
        const float f = half_to_float(std::uint16_t(h));
        if (std::isnan(f)) {
            EXPECT_EQ((h >> 10) & 0x1F, 31u);
            continue;
        }
        EXPECT_EQ(float_to_half(f), std::uint16_t(h)) << h;
    }
}

TEST(Viewer, BrowserCompositeMatchesCompositeCpp) {
    for (const auto& frame : index_json().at("frames")) {
        const std::string name = frame.at("name").get<std::string>();
        SCOPED_TRACE(name);
        const Packet p = unpack(frame);
        const std::size_t n = std::size_t(p.w) * std::size_t(p.h);

        const auto base = corrected_baseline(p.sdr, p.model.corpus_ev, p.scalars.curve_params);
        const auto want_base = planar_of(npy(frame.at("base").get<std::string>()));
        expect_gpu_close(base.buffer().span(), want_base, name + " baseline");

        const DoubleSums base_r = double_sums(base.buffer().span(), n);
        for (const auto& c : frame.at("cases")) {
            const std::string cname = c.at("name").get<std::string>();
            const json& prm = c.at("params");
            CompositeParams cp;
            cp.mode = mode_of(prm.at("mode").get<std::string>());
            cp.strength = prm.at("strength").get<float>();
            cp.preserve_outside = prm.at("preserve").get<bool>();
            cp.regions = bands(prm.contains("regions") ? prm.at("regions") : index_json().at("default_regions"));
            const auto out = composite(p.sdr, p.fields, p.scalars, p.model, cp);
            if (c.at("model").is_string())
                expect_gpu_close(out.buffer().span(), planar_of(npy(c.at("model").get<std::string>())), name + " " + cname);

            // Exact reductions on both sides of fp32 composites that agree to the bound above.
            const DoubleSums r = double_sums(out.buffer().span(), n);
            EXPECT_NEAR(r.peak, c.at("peak_nits").get<double>(), 5e-5 * kPeakNits + 1e-5 * r.peak) << cname;
            EXPECT_NEAR(r.mean, c.at("mean_nits").get<double>(), 5e-5 * kPeakNits + 1e-5 * r.mean) << cname;
            EXPECT_NEAR(base_r.peak, c.at("base_peak_nits").get<double>(), 5e-5 * kPeakNits + 1e-5 * base_r.peak) << cname;
        }
    }
}

// ---- step 3: the display pass on the CPU against the browser's canvas ------

namespace {

NetworkLinearImage image_of(const NpyArray& a) {
    return NetworkLinearImage(PlanarBuffer(3, int(a.shape[0]), int(a.shape[1]), planar_of(a)));
}

ViewParams view_params(const json& j) {
    ViewParams v;
    v.mode = static_cast<ViewMode>(j.at("view").get<int>());
    v.display_nits = j.at("displayNits").get<double>();
    v.show = j.at("show").get<std::string>() == "baseline" ? ViewSource::Baseline : ViewSource::Model;
    v.wipe = j.at("wipe").get<double>();
    if (j.contains("diffGain")) v.diff_gain = j.at("diffGain").get<double>();
    return v;
}

}  // namespace

TEST(Viewer, DisplayPassMatchesTheBrowserCanvas) {
    for (const auto& frame : index_json().at("frames")) {
        const std::string name = frame.at("name").get<std::string>();
        // The views were presented on the first composite case.
        const auto model = image_of(npy(frame.at("cases").at(0).at("model").get<std::string>()));
        const auto base = image_of(npy(frame.at("base").get<std::string>()));
        for (const auto& v : frame.at("views")) {
            const std::string vname = name + " " + v.at("name").get<std::string>();
            const Rgb8Image got = render_view_rgb8(model, base, view_params(v.at("params")));
            const NpyArray want = npy(v.at("pixels").get<std::string>());   // (H, W, 3) uint8 as float
            ASSERT_EQ(got.rgb.size(), want.data.size()) << vname;
            int worst = 0;
            std::size_t off_by_one = 0;
            for (std::size_t i = 0; i < want.data.size(); ++i) {
                const int d = std::abs(int(got.rgb[i]) - int(want.data[i]));
                worst = std::max(worst, d);
                off_by_one += d == 1;
            }
            EXPECT_LE(worst, 1) << vname;
            ::testing::Test::RecordProperty(vname, std::to_string(worst) + "/" + std::to_string(off_by_one));
        }
    }
}

// ---- step 5: the HDR output paths, against independent references ----------

namespace {

// A 1 x n picture of network-unit pixels, and the same for the baseline.
NetworkLinearImage row_of(std::vector<std::array<float, 3>> px) {
    PlanarBuffer b(3, 1, int(px.size()));
    for (std::size_t i = 0; i < px.size(); ++i)
        for (int c = 0; c < 3; ++c) b.at(c, 0, int(i)) = px[i][std::size_t(c)];
    return NetworkLinearImage(std::move(b));
}

}  // namespace

TEST(Viewer, HdrPathsWriteAbsoluteNitsClippedAtThePeak) {
    // 203 nits white, 1 000 nits white, 3 000 nits white, a 203-nit Rec.709 red.
    const auto m = row_of({{0.0203f, 0.0203f, 0.0203f}, {0.1f, 0.1f, 0.1f}, {0.3f, 0.3f, 0.3f}, {0.0203f, 0.0f, 0.0f}});
    const auto b = row_of({{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}});
    ViewParams p;
    p.display_nits = 10000.0;   // no view ceiling below the display's

    p.target = DisplayTarget::scrgb(1000.0);   // 1.0 = 80 nits, Rec.709
    auto v = render_view(m, b, p);
    EXPECT_NEAR(v.at(0, 0, 0), 203.0f / 80.0f, 1e-5f);
    EXPECT_NEAR(v.at(1, 0, 1), 1000.0f / 80.0f, 1e-4f);
    EXPECT_NEAR(v.at(2, 0, 2), 1000.0f / 80.0f, 1e-4f);   // clipped at the display's peak, not tone-mapped
    EXPECT_NEAR(v.at(0, 0, 3), 203.0f / 80.0f, 1e-5f);
    EXPECT_NEAR(v.at(1, 0, 3), 0.0f, 1e-6f);

    p.target = DisplayTarget::hdr10(1000.0);   // PQ, Rec.2020
    v = render_view(m, b, p);
    EXPECT_NEAR(v.at(0, 0, 0), pq_oetf(203.0f), 1e-6f);
    EXPECT_NEAR(v.at(0, 0, 0), 0.5807f, 5e-4f);   // BT.2408: reference white at 58 % PQ
    EXPECT_NEAR(v.at(1, 0, 2), pq_oetf(1000.0f), 1e-6f);
    const Mat3 to2020 = rgb_to_rgb_matrix(Primaries::Rec709, Primaries::Rec2020);
    for (int k = 0; k < 3; ++k)
        EXPECT_NEAR(v.at(k, 0, 3), pq_oetf(float(to2020[std::size_t(k)][0] * 203.0)), 1e-5f) << k;   // fp32 matrix, amplified by PQ

    p.target = DisplayTarget::edr(1600.0);   // 1.0 = SDR white (203), Display P3
    p.display_nits = 600.0;                  // the view peak is the lower ceiling here
    v = render_view(m, b, p);
    EXPECT_NEAR(v.at(0, 0, 0), 1.0f, 1e-5f);
    EXPECT_NEAR(v.at(0, 0, 1), 600.0f / 203.0f, 1e-5f);
    const Mat3 toP3 = rgb_to_rgb_matrix(Primaries::Rec709, Primaries::P3D65);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(v.at(k, 0, 3), float(toP3[std::size_t(k)][0]), 1e-5f) << k;
}

TEST(Viewer, HdrOverlaysAreGraphicsAtTheSdrWhite) {
    const auto m = row_of({{0.0203f, 0.0203f, 0.0203f}, {0.5f, 0.5f, 0.5f}});
    const auto b = row_of({{0.0203f, 0.0203f, 0.0203f}, {0.0f, 0.0f, 0.0f}});
    ViewParams p;
    p.mode = ViewMode::FalseColour;
    p.target = DisplayTarget::scrgb(1000.0);
    const auto v = render_view(m, b, p);
    const auto zone = false_colour(false_colour_zone(203.0f));   // diffuse white's zone, an SDR code
    for (int k = 0; k < 3; ++k) {
        const float c = zone[std::size_t(k)];
        const float lin = c > 0.04045f ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f;
        EXPECT_NEAR(v.at(k, 0, 0), 203.0f * lin / 80.0f, 1e-5f) << k;
    }
    // The wipe handle inverts the picture against the ceiling on an HDR image.
    ViewParams w;
    w.display_nits = 1000.0;
    w.target = DisplayTarget::scrgb(1000.0);
    w.wipe = 0.75;           // column 1 of 2 is at u = 0.75: on the handle
    w.wipe_half_width = 0.1;
    const auto vw = render_view(m, b, w);
    EXPECT_NEAR(vw.at(0, 0, 1), (1000.0f - 1000.0f) / 80.0f, 1e-5f);   // 5 000 nits clips to 1 000, inverted to 0
    EXPECT_NEAR(vw.at(0, 0, 0), 203.0f / 80.0f, 1e-5f);                // left of the wipe: the baseline
}

// ---- steps 6 and 7: the reduction ladder and the probe, against the browser --

TEST(Viewer, ReductionLadderEqualsTheBrowsersExactly) {
    for (const auto& frame : index_json().at("frames")) {
        const std::string name = frame.at("name").get<std::string>();
        const auto base = image_of(npy(frame.at("base").get<std::string>()));
        const std::size_t n = std::size_t(base.width()) * std::size_t(base.height());
        const Reductions rb = reduce_ladder(base.buffer());
        for (const auto& c : frame.at("cases")) {
            EXPECT_EQ(double(rb.peak) * kPeakNits, c.at("base_peak_nits").get<double>()) << name;
            if (!c.at("model").is_string()) continue;   // the browser's composite is stored for these
            const auto model = image_of(npy(c.at("model").get<std::string>()));
            const Reductions r = reduce_ladder(model.buffer());
            const std::string what = name + " " + c.at("name").get<std::string>();
            EXPECT_EQ(double(r.peak) * kPeakNits, c.at("peak_nits").get<double>()) << what;
            EXPECT_EQ(double(r.sum) * kPeakNits / double(n), c.at("mean_nits").get<double>()) << what;
            const json& m = c.at("metrics");
            EXPECT_EQ(std::ceil(double(r.peak) * kPeakNits), m.at("maxcll").get<double>()) << what;
            EXPECT_EQ(std::ceil(double(r.sum) * kPeakNits / double(n)), m.at("maxfall").get<double>()) << what;
        }
    }
}

TEST(Viewer, ProbeEqualsTheBrowsersExactly) {
    for (const auto& frame : index_json().at("frames")) {
        const std::string name = frame.at("name").get<std::string>();
        const auto model = image_of(npy(frame.at("cases").at(0).at("model").get<std::string>()));
        const auto base = image_of(npy(frame.at("base").get<std::string>()));
        for (const auto& pr : frame.at("probes")) {
            const auto at = pr.at("at").get<std::vector<double>>();
            const Probe got = probe_pixel(model, base, at[0], at[1]);
            const std::string what = name + " probe " + pr.at("at").dump();
            EXPECT_EQ(got.model.x, pr.at("x").get<int>()) << what;
            EXPECT_EQ(got.model.y, pr.at("y").get<int>()) << what;
            for (const auto& [sample, key] : {std::pair{&got.model, "model"}, std::pair{&got.baseline, "baseline"}}) {
                const json& w = pr.at(key);
                EXPECT_EQ(sample->rgb_nits[0], w.at("r").get<double>()) << what << key;
                EXPECT_EQ(sample->rgb_nits[1], w.at("g").get<double>()) << what << key;
                EXPECT_EQ(sample->rgb_nits[2], w.at("b").get<double>()) << what << key;
                EXPECT_EQ(sample->nits, w.at("nits").get<double>()) << what << key;
            }
        }
    }
}

// ---- step 8: sample, measurements and scopes, against the browser -----------

namespace {

struct Masks {
    std::vector<float> highlight, shadow;
    std::vector<std::uint8_t> sdr8;
};
Masks masks_of(const json& frame) {
    const json& hd = frame.at("header");
    const std::size_t n = hd.at("width").get<std::size_t>() * hd.at("height").get<std::size_t>();
    std::ifstream in(kDir / frame.at("body").get<std::string>(), std::ios::binary);
    const std::vector<unsigned char> body((std::istreambuf_iterator<char>(in)), {});
    const std::size_t off_shadow = hd.at("offsets").at("shadow").get<std::size_t>();
    const std::size_t off_sdr = hd.at("offsets").at("sdr").get<std::size_t>();
    Masks m;
    for (std::size_t i = 0; i < n; ++i) {
        m.highlight.push_back(half_to_float(std::uint16_t(body[i * 8 + 6] | (body[i * 8 + 7] << 8))));
        m.shadow.push_back(half_to_float(std::uint16_t(body[off_shadow + i * 2] | (body[off_shadow + i * 2 + 1] << 8))));
    }
    m.sdr8.assign(body.begin() + std::ptrdiff_t(off_sdr), body.begin() + std::ptrdiff_t(off_sdr + n * 3));
    return m;
}

// JSON numbers, with the emitter's "nan" for NaN.
double num(const json& v) {
    return v.is_string() ? std::numeric_limits<double>::quiet_NaN() : v.get<double>();
}
void expect_same(double got, const json& want, const std::string& what) {
    const double w = num(want);
    if (std::isnan(w)) EXPECT_TRUE(std::isnan(got)) << what;
    else EXPECT_EQ(got, w) << what;
}

}  // namespace

TEST(Viewer, SampleMeasurementsAndScopesEqualTheBrowsers) {
    for (const auto& frame : index_json().at("frames")) {
        const std::string name = frame.at("name").get<std::string>();
        const Masks mk = masks_of(frame);
        const MaskCoverage cov = mask_coverage(mk.highlight, mk.shadow, mk.sdr8);
        EXPECT_EQ(cov.highlight_pct, frame.at("mask_pct").at("highlight").get<double>()) << name;
        EXPECT_EQ(cov.shadow_pct, frame.at("mask_pct").at("shadow").get<double>()) << name;
        EXPECT_EQ(cov.clipped_pct, frame.at("mask_pct").at("clipped").get<double>()) << name;

        const auto base = image_of(npy(frame.at("base").get<std::string>()));
        const int w = base.width(), h = base.height();
        const SampleGrid grid = sample_grid(w, h);
        const Reductions rb = reduce_ladder(base.buffer());
        const PlanarBuffer base_sample = take_sample(base.buffer(), grid);
        for (const auto& c : frame.at("cases")) {
            if (!c.at("model").is_string()) continue;
            const std::string what = name + " " + c.at("name").get<std::string>();
            const auto model = image_of(npy(c.at("model").get<std::string>()));
            if (c.contains("sample")) {
                EXPECT_EQ(grid.width, c.at("sample").at("width").get<int>()) << what;
                EXPECT_EQ(grid.height, c.at("sample").at("height").get<int>()) << what;
                const NpyArray idx = npy(c.at("sample").at("index").get<std::string>());
                ASSERT_EQ(idx.data.size(), grid.index.size()) << what;
                for (std::size_t i = 0; i < grid.index.size(); ++i)
                    ASSERT_EQ(grid.index[i], int(idx.data[i])) << what << " sample " << i;
            }
            const PlanarBuffer model_sample = take_sample(model.buffer(), grid);
            const Measured got = measure_view(model_sample, base_sample, grid, mk.highlight, mk.shadow, cov,
                                              reduce_ladder(model.buffer()), rb, std::size_t(w) * std::size_t(h));
            const json& m = c.at("metrics");
            const ViewerMetrics& g = got.metrics;
            for (const auto& [key, v] : std::vector<std::pair<const char*, double>>{
                     {"maxcll", g.maxcll}, {"maxfall", g.maxfall}, {"peak_nits", g.peak_nits},
                     {"baseline_peak_nits", g.baseline_peak_nits}, {"headroom_stops", g.headroom_stops},
                     {"headroom_highlight_stops", g.headroom_highlight_stops},
                     {"headroom_shadow_stops", g.headroom_shadow_stops}, {"departure_rms_stops", g.departure_rms_stops},
                     {"p99_nits", g.p99_nits}, {"median_nits", g.median_nits},
                     {"above_diffuse_white_pct", g.above_diffuse_white_pct},
                     {"above_1000_nits_pct", g.above_1000_nits_pct}, {"highlight_mask_pct", g.highlight_mask_pct},
                     {"shadow_mask_pct", g.shadow_mask_pct}})
                expect_same(v, m.at(key), what + " " + key);

            const json& s = c.at("scopes");
            for (const auto& [key, v] : std::vector<std::pair<const char*, const std::vector<double>*>>{
                     {"lo", &got.scopes.lo}, {"q1", &got.scopes.q1}, {"mid", &got.scopes.mid}, {"q3", &got.scopes.q3},
                     {"hi", &got.scopes.hi}, {"histogram", &got.scopes.histogram}})
                EXPECT_EQ(*v, s.at(key).get<std::vector<double>>()) << what << " " << key;

            if (c.contains("vector")) {
                const auto img = vectorscope(model_sample);
                const NpyArray want = npy(c.at("vector").get<std::string>());
                ASSERT_EQ(img.size(), want.data.size()) << what;
                std::size_t off = 0;
                int worst = 0;
                for (std::size_t i = 0; i < img.size(); ++i) {
                    const int d = std::abs(int(img[i]) - int(want.data[i]));
                    worst = std::max(worst, d);
                    off += d != 0;
                }
                EXPECT_EQ(worst, 0) << what << " vectorscope, " << off << " values differ";
            }
        }
    }
}

// ---- Phase 3 step 8: the frame measured end to end, as the app measures it --
//
// engine/measure runs the whole chain from the frame the page received (its
// SDR, its fields) with each case's grade. The masks and the clip are the
// page's to the bit; the composite is the CPU one, which agrees with the
// browser's GPU composite within the bound above, so the measurements do to
// the same order.
TEST(Viewer, MeasureFrameIsThePagesComputeStats) {
    for (const auto& frame : index_json().at("frames")) {
        const std::string name = frame.at("name").get<std::string>();
        const Packet p = unpack(frame);
        const NetworkLinearImage base = corrected_baseline(p.sdr, p.model.corpus_ev, p.scalars.curve_params);
        for (const auto& c : frame.at("cases")) {
            if (!c.contains("metrics")) continue;
            const std::string what = name + " " + c.at("name").get<std::string>();
            const json& prm = c.at("params");
            CompositeParams cp;
            cp.mode = mode_of(prm.at("mode").get<std::string>());
            cp.strength = prm.at("strength").get<float>();
            cp.preserve_outside = prm.at("preserve").get<bool>();
            cp.regions = bands(prm.contains("regions") ? prm.at("regions") : index_json().at("default_regions"));
            const FrameMeasure fm = measure_frame(p.sdr, p.fields, p.scalars, p.model, cp, &base);
            EXPECT_EQ(fm.coverage.highlight_pct, frame.at("mask_pct").at("highlight").get<double>()) << what;
            EXPECT_EQ(fm.coverage.shadow_pct, frame.at("mask_pct").at("shadow").get<double>()) << what;
            EXPECT_EQ(fm.coverage.clipped_pct, frame.at("mask_pct").at("clipped").get<double>()) << what;
            const json& m = c.at("metrics");
            const ViewerMetrics& g = fm.measured.metrics;
            // MaxCLL and MaxFALL are ceilings of the exact reductions: a unit either way.
            EXPECT_NEAR(g.maxcll, m.at("maxcll").get<double>(), 1.0) << what;
            EXPECT_NEAR(g.maxfall, m.at("maxfall").get<double>(), 1.0) << what;
            for (const auto& [key, v] : std::vector<std::pair<const char*, double>>{
                     {"peak_nits", g.peak_nits}, {"baseline_peak_nits", g.baseline_peak_nits},
                     {"p99_nits", g.p99_nits}, {"median_nits", g.median_nits}}) {
                const double w = num(m.at(key));
                EXPECT_NEAR(v, w, 5e-5 * kPeakNits + 2e-3 * std::abs(w)) << what << " " << key;
            }
            for (const auto& [key, v] : std::vector<std::pair<const char*, double>>{
                     {"above_diffuse_white_pct", g.above_diffuse_white_pct},
                     {"above_1000_nits_pct", g.above_1000_nits_pct}, {"departure_rms_stops", g.departure_rms_stops}}) {
                EXPECT_NEAR(v, num(m.at(key)), 0.05) << what << " " << key;
            }
            // The probe reads the full-resolution composite at a pixel.
            const auto probe = fm.probe_at(p.w / 2 + 0.7, p.h / 3 + 0.2);
            ASSERT_TRUE(probe.has_value()) << what;
            EXPECT_EQ(probe->x, p.w / 2);
            EXPECT_EQ(probe->y, p.h / 3);
            EXPECT_FALSE(fm.probe_at(-0.5, 0).has_value());
            EXPECT_FALSE(fm.probe_at(p.w, 0).has_value());
            const std::size_t i = std::size_t(probe->y) * std::size_t(p.w) + std::size_t(probe->x);
            EXPECT_EQ(*probe->hi_mask, p.fields.highlight.plane(0)[i]);
            ASSERT_TRUE(probe->sdr.has_value());
            EXPECT_EQ((*probe->sdr)[0], int(std::lround(p.sdr.buffer().plane(0)[i] * 255.0f)));
            EXPECT_EQ(fm.vector_rgba.size(), std::size_t(kVectorSize) * kVectorSize * 4);
        }
    }
}
