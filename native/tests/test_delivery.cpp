// Grade controls, HDR10/HLG, metadata sidecars and EXR/ACES/OCIO writers
// against the Python they port (goldens from tools/emit_delivery_golden.py).
// The sidecars, EXRs and OCIO config are compared byte for byte.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>

#include <nlohmann/json.hpp>

#include "rudra/core/grade.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/measure.hpp"
#include "rudra/core/metadata.hpp"
#include "rudra/deliver/exr.hpp"
#include "rudra/deliver/sidecars.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/platform/pyjson.hpp"

using namespace rudra;
namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const fs::path kDir = fs::path(RUDRA_GOLDEN_DIR) / "delivery";

const json& idx() {
    static const json j = [] {
        std::ifstream in(kDir / "index.json");
        return json::parse(in);
    }();
    return j;
}

NpyArray f32(const std::string& f) {
    auto a = read_npy(kDir / f);
    EXPECT_TRUE(a) << f;
    return a ? std::move(*a) : NpyArray{};
}
NpyArrayF64 f64(const std::string& f) {
    auto a = read_npy_f64(kDir / f);
    EXPECT_TRUE(a) << f;
    return a ? std::move(*a) : NpyArrayF64{};
}
PlanarBuffer planar(const NpyArray& a) {
    return PlanarBuffer(int(a.shape[0]), int(a.shape[1]), int(a.shape[2]), a.data);
}
NitsFrame nits(const NpyArrayF64& a, std::size_t offset = 0) {
    const int h = int(a.shape[a.shape.size() - 2]), w = int(a.shape[a.shape.size() - 1]);
    NitsFrame f(h, w);
    std::copy(a.data.begin() + std::ptrdiff_t(offset), a.data.begin() + std::ptrdiff_t(offset + f.span().size()),
              f.span().begin());
    return f;
}
std::string bytes_of(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

template <class A, class B>
void expect_close(const A& got, const B& want, double rtol, double atol, const std::string& what) {
    ASSERT_EQ(std::size(got), std::size(want)) << what;
    double worst = 0.0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < std::size(want); ++i) {
        const double g = double(got[i]), w = double(want[i]);
        if (std::isnan(w) && std::isnan(g)) continue;
        const double e = std::abs(g - w) - (atol + rtol * std::abs(w));
        if (e > worst) { worst = e; at = i; }
    }
    EXPECT_LE(worst, 0.0) << what << " at " << at << ": " << double(got[at]) << " vs " << double(want[at]);
}

fs::path scratch(const std::string& name) {
    const fs::path d = fs::temp_directory_path() / "rudra_delivery_test";
    fs::create_directories(d);
    return d / name;
}

}  // namespace

TEST(Grade, ApplyGradeMatchesControlsPy) {
    const auto& g = idx().at("grade");
    const NitsFrame in = nits(f64(g.at("input")));
    std::map<std::string, PlanarBuffer> masks;
    for (const auto& [k, v] : g.at("masks").items()) masks[k] = planar(f32(v));
    for (const auto& c : g.at("cases")) {
        GradeControls gc;
        gc.exposure_ev = c.at("exposure_ev");
        gc.peak_nits = c.at("peak_nits");
        if (!c.at("knee_nits").is_null()) gc.knee_nits = c.at("knee_nits").get<double>();
        gc.highlight_desat = c.at("highlight_desat");
        for (const auto& r : c.at("regions")) gc.regions.push_back({masks.at(r.at("mask")), r.at("ev"), r.at("mask")});
        auto out = apply_grade(in, gc);
        ASSERT_TRUE(out) << out.error().message;
        // float64 maths cast to float32: equal but for numpy's SIMD exp.
        expect_close(out->span(), f32(c.at("expected")).data, 2e-7, 0.0, c.at("name"));
    }
}

TEST(Grade, ItmStrengthMap) {
    const auto& g = idx().at("grade");
    std::vector<MaskRegion> regions;
    for (const auto& r : g.at("itm").at("regions"))
        regions.push_back({planar(f32(g.at("masks").at(r.at("mask").get<std::string>()))), r.at("ev"), ""});
    auto out = itm_strength_map(24, 32, g.at("itm").at("base"), regions);
    ASSERT_TRUE(out);
    expect_close(out->span(), f32(g.at("itm").at("expected")).data, 0.0, 0.0, "itm");
}

TEST(Hdr10, PqCurvesAndTwelveBitCodes) {
    const auto& h = idx().at("hdr10");
    const auto in = f64(h.at("pq_in"));
    const auto oetf = f32(h.at("pq_oetf"));
    std::vector<float> got(in.data.size());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = pq_oetf(float(in.data[i]));
    // numpy's float32 power is 1 ulp off in places (glibc's powf is correctly
    // rounded there), and PQ's exponent of 78.84 turns one ulp into 1e-5 of
    // code: a fiftieth of a 10-bit step. The 12-bit codes below are exact.
    expect_close(got, oetf.data, 0.0, 2e-5, "pq_oetf");
    const auto cin = f64(h.at("eotf_in"));
    const auto eotf = f32(h.at("pq_eotf"));
    got.resize(cin.data.size());
    for (std::size_t i = 0; i < got.size(); ++i) got[i] = pq_eotf(float(cin.data[i]));
    expect_close(got, eotf.data, 1e-4, 1e-6, "pq_eotf");
    const auto codes = h.at("pq12").get<std::vector<int>>();
    for (std::size_t i = 0; i < codes.size(); ++i) EXPECT_EQ(pq12(in.data[i]), codes[i]) << in.data[i];
}

TEST(Hdr10, MasterToPeakAndPq) {
    const auto& h = idx().at("hdr10");
    const PlanarBuffer norm = planar(f32(h.at("norm")));
    for (const auto& m : h.at("master")) {
        std::optional<double> knee;
        if (!m.at("knee").is_null()) knee = m.at("knee").get<double>();
        auto mastered = master_to_peak(norm, m.at("peak"), knee);
        ASSERT_TRUE(mastered);
        // 1 - exp(-x) cancels for small x, so a 1-ulp exp difference is 1e-4
        // relative there; 2e-4 nits absolute is far below any 10-bit step.
        expect_close(mastered->span(), f32(m.at("mastered")).data, 2e-6, 2e-4, m.at("mastered"));
        auto pq = master_to_pq(norm, m.at("peak"), knee);
        ASSERT_TRUE(pq);
        expect_close(pq->first.span(), f32(m.at("pq")).data, 0.0, 2e-5, m.at("pq"));
    }
    EXPECT_FALSE(master_to_peak(norm, 50.0));
    EXPECT_FALSE(master_to_peak(norm, 1000.0, 1000.0));
}

TEST(Hdr10, EncodeMasterEveryProfile) {
    const auto& h = idx().at("hdr10");
    const PlanarBuffer norm = planar(f32(h.at("norm")));
    for (const auto& e : h.at("encode")) {
        auto r = encode_master(norm, e.at("profile"), e.at("peak"));
        ASSERT_TRUE(r);
        expect_close(r->first.span(), f32(e.at("code")).data, 0.0, 2e-5, e.at("code"));
        expect_close(r->second.span(), f32(e.at("display")).data, 2e-6, 2e-4, e.at("display"));
    }
    const auto& prof = h.at("profiles");
    ASSERT_EQ(delivery_profiles().size(), prof.size());
    for (const auto& p : delivery_profiles()) {
        const auto& want = prof.at(p.name);
        EXPECT_EQ(p.codec, want.at("codec").get<std::string>());
        EXPECT_EQ(p.encoder, want.at("encoder").get<std::string>());
        EXPECT_EQ(p.transfer, want.at("transfer").get<std::string>());
        EXPECT_EQ(p.pixel_format, want.at("pixel_format").get<std::string>());
        EXPECT_EQ(p.prores_profile, want.value("profile", -1));
        EXPECT_EQ(p.tag, want.value("tag", std::string()));
    }
}

TEST(Metadata, ShotsAndSidecarsByteIdentical) {
    const auto& m = idx().at("metadata");
    const auto frames = f64(m.at("frames"));
    const std::size_t per = std::size_t(frames.shape[1] * frames.shape[2] * frames.shape[3]);
    std::vector<FrameStats> stats;
    for (int i = 0; i < int(frames.shape[0]); ++i) stats.push_back(analyze_frame(nits(frames, std::size_t(i) * per), i));
    const auto shots = detect_shots(stats);
    const auto want = m.at("shots").get<std::vector<std::vector<int>>>();
    ASSERT_EQ(shots.size(), want.size());
    for (std::size_t i = 0; i < shots.size(); ++i) {
        EXPECT_EQ(shots[i].first, want[i][0]);
        EXPECT_EQ(shots[i].second, want[i][1]);
    }
    auto paths = write_all_sidecars(stats, scratch("seq"), 1000.0);
    ASSERT_TRUE(paths);
    const auto& sc = m.at("sidecars");
    EXPECT_EQ(bytes_of(paths->rudra), bytes_of(kDir / sc.at("rudra").get<std::string>()));
    EXPECT_EQ(bytes_of(paths->dovi), bytes_of(kDir / sc.at("dovi").get<std::string>()));
    EXPECT_EQ(bytes_of(paths->hdr10plus), bytes_of(kDir / sc.at("hdr10plus").get<std::string>()));
}

TEST(Exr, FloatToHalfMatchesNumpy) {
    const auto& e = idx().at("exr");
    const auto in = f32(e.at("half_in")), bits = f32(e.at("half_bits"));
    std::size_t bad = 0;
    for (std::size_t i = 0; i < in.data.size(); ++i)
        if (float_to_half(in.data[i]) != std::uint16_t(bits.data[i])) ++bad;
    EXPECT_EQ(bad, 0u);
}

TEST(Exr, FilesByteIdentical) {
    const auto& e = idx().at("exr");
    const PlanarBuffer img = planar(f32(e.at("input")));
    const PlanarBuffer rgba = planar(f32(e.at("input_rgba")));
    PlanarBuffer absimg = img;
    for (float& v : absimg.span()) v = std::abs(v);
    ExrAttributes prov;
    for (const auto& [k, v] : e.at("provenance").items()) prov.emplace_back(k, v.get<std::string>());

    struct Case { const char* key; std::function<Result<void>(const fs::path&)> write; };
    const std::vector<Case> cases{
        {"half", [&](const fs::path& p) { return write_exr(p, img, true); }},
        {"float_chroma", [&](const fs::path& p) {
             return write_exr(p, img, false, kRec2020Chromaticities,
                              {{"rudra:checkpoint", "sdr2hdr_shadow_v1.pt"}, {"rudra:note", "tab\there \"quoted\""}});
         }},
        {"rgba", [&](const fs::path& p) { return write_exr(p, rgba, true); }},
        {"aces", [&](const fs::path& p) { return write_aces_exr(p, absimg, Primaries::Rec709, 1.0, prov); }},
        {"acescg", [&](const fs::path& p) { return write_acescg_exr(p, absimg, Primaries::Rec2020); }},
    };
    for (const auto& c : cases) {
        const fs::path out = scratch(std::string(c.key) + ".exr");
        auto r = c.write(out);
        ASSERT_TRUE(r) << c.key;
        EXPECT_EQ(bytes_of(out), bytes_of(kDir / e.at("files").at(c.key).get<std::string>())) << c.key;
    }
    EXPECT_EQ(ocio_config_text(), bytes_of(kDir / e.at("ocio").get<std::string>()));
}

TEST(PyJson, ReprMatchesPython) {
    EXPECT_EQ(pyjson::repr(1.0), "1.0");
    EXPECT_EQ(pyjson::repr(0.1), "0.1");
    EXPECT_EQ(pyjson::repr(1e16), "1e+16");
    EXPECT_EQ(pyjson::repr(1234567890123456.0), "1234567890123456.0");
    EXPECT_EQ(pyjson::repr(1e-5), "1e-05");
    EXPECT_EQ(pyjson::repr(0.0001), "0.0001");
    EXPECT_EQ(pyjson::repr(-0.0), "-0.0");
    EXPECT_EQ(pyjson::repr(2151.418548746334), "2151.418548746334");
    EXPECT_EQ(pyjson::repr(1.5e300), "1.5e+300");
    EXPECT_EQ(pyjson::dumps(pyjson::Dict{{"b", 1}, {"a", pyjson::List{}}}, -1, true), "{\"a\": [], \"b\": 1}");
}
