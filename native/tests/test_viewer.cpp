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
#include <string>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/gamut.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/half.hpp"
#include "rudra/core/view.hpp"
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

struct Reductions {
    double peak = 0.0, mean = 0.0;
};
Reductions reduce(std::span<const float> planar, std::size_t n) {
    Reductions r;
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

        const Reductions base_r = reduce(base.buffer().span(), n);
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
            const Reductions r = reduce(out.buffer().span(), n);
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
