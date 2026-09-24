// rudra-gpu-parity: day 8 of NATIVE_ARCHITECTURE.md section 12. The composite
// shader on this machine's GPU against the C++ reference (composite.cpp), on
// the composite goldens tools/emit_composite_golden.py writes.
//
//   rudra-gpu-parity [--api d3d12|d3d11|metal|vulkan|gl] [--golden DIR] [--report FILE] [--bench]
//
// --bench also times one composite pass at 1080p and 4K into the viewer's
// RGBA16F target (GPU timestamps where the backend has them), for the budget
// table in NATIVE_ARCHITECTURE.md 6.6. Budgets: 4 ms at 1080p, 12 ms at 4K,
// composite and view together; this times the composite.
//
// Every case runs twice: into an RGBA32F target, held to the C++ fp32 result
// (atol 1e-6, rtol 2e-4, network units), and into RGBA16F, the viewer's
// format, held to 2 half-float ulp of the C++ result rounded to half.
//
// Then the display pass (docs/view.spec.md section 2, Phase 2 step 4): seven
// views of each frame's default composite and its baseline, into an RGBA8
// target, held to core/view.cpp within 1 code.
// Exit 0 when every case passes.

#include <QCommandLineParser>
#include <QFile>
#include <QFloat16>
#include <QGuiApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/view.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/render/gpu_composite.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

constexpr double kAtol = 1e-6, kRtol = 2e-4;
constexpr int kMaxHalfUlp = 2;

PlanarBuffer load(const fs::path& dir, const nlohmann::json& e) {
    auto a = read_npy(dir / e.at("file").get<std::string>());
    if (!a) return {};
    return PlanarBuffer(int(a->shape[0]), int(a->shape[1]), int(a->shape[2]), std::move(a->data));
}

// Distance in representable halves, with +0 and -0 equal.
int half_ulp(float got, float want) {
    auto key = [](float v) {
        const qfloat16 h(v);
        std::uint16_t b;
        std::memcpy(&b, &h, 2);
        return (b & 0x8000) ? -int(b & 0x7fff) : int(b);
    };
    return std::abs(key(got) - key(want));
}

struct Row {
    std::string label;
    double f32_abs = 0.0, f32_excess = -1.0, py_abs = 0.0;
    int f16_ulp = 0;
    bool ok() const { return f32_excess <= 0.0 && f16_ulp <= kMaxHalfUlp; }
};

}  // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);
    QCommandLineParser cli;
    cli.setApplicationDescription("RUDRA day 8: GPU composite vs the C++ reference");
    cli.addHelpOption();
    QCommandLineOption api_opt("api", "d3d12, d3d11, metal, vulkan, gl or auto", "api", "auto");
    QCommandLineOption golden_opt("golden", "composite golden folder", "dir", RUDRA_COMPOSITE_GOLDEN_DIR);
    QCommandLineOption report_opt("report", "write a JSON report here", "file");
    QCommandLineOption bench_opt("bench", "also time the composite at 1080p and 4K");
    cli.addOptions({api_opt, golden_opt, report_opt, bench_opt});
    cli.process(app);

    const QString a = cli.value(api_opt).toLower();
    const GpuApi api = a == "d3d12" ? GpuApi::D3D12 : a == "d3d11" ? GpuApi::D3D11 : a == "metal" ? GpuApi::Metal
                     : a == "vulkan" ? GpuApi::Vulkan : a == "gl" ? GpuApi::OpenGL : GpuApi::Auto;
    QTextStream out(stdout);
    auto gpu = GpuCompositor::create(api);
    if (!gpu) {
        out << "rudra-gpu-parity: " << QString::fromStdString(gpu.error().message) << " "
            << QString::fromStdString(gpu.error().detail) << "\n";
        return 2;
    }
    const auto info = (*gpu)->info();

    const fs::path dir = cli.value(golden_opt).toStdString();
    std::ifstream in(dir / "index.json");
    if (!in) {
        out << "rudra-gpu-parity: no goldens at " << QString::fromStdString(dir.string()) << "\n";
        return 2;
    }
    const nlohmann::json idx = nlohmann::json::parse(in);
    const auto& mj = idx.at("model");
    const ModelConstants model{mj.at("log_scale").get<float>(), mj.at("max_hdr").get<float>(),
                               mj.at("corpus_ev").get<float>()};

    std::vector<Row> rows;
    for (const auto& [name, f] : idx.at("frames").items()) {
        const SdrImage sdr(load(dir, f.at("sdr")));
        const Fields fields{load(dir, f.at("residual")), load(dir, f.at("highlight")), load(dir, f.at("shadow"))};
        FrameScalars sc;
        sc.shadow_weight = f.at("shadow_weight").get<float>();
        sc.curve_params = f.at("curve_params").get<std::vector<float>>();

        std::vector<std::pair<std::string, CompositeParams>> cases;
        std::vector<PlanarBuffer> python;
        for (const auto& c : f.at("composites")) {
            CompositeParams p;
            const auto m = c.at("mode").get<std::string>();
            p.mode = m == "highlights" ? RecoveryMode::Highlights : m == "shadows" ? RecoveryMode::Shadows
                   : m == "off" ? RecoveryMode::Off : RecoveryMode::All;
            p.strength = c.at("strength").get<float>();
            p.preserve_outside = c.at("preserve").get<bool>();
            cases.emplace_back(name + " " + m + " s" + std::to_string(int(std::lround(p.strength * 100))) +
                               (p.preserve_outside ? " preserve" : ""), p);
            python.push_back(load(dir, c.at("expected")));
        }
        CompositeParams graded;
        for (const auto& b : f.at("master").at("bands"))
            graded.regions.push_back({b.at("low_nits").get<double>(), b.at("high_nits").get<double>(), b.at("ev").get<double>()});
        cases.emplace_back(name + " region EV", graded);
        python.emplace_back();

        for (std::size_t k = 0; k < cases.size(); ++k) {
            const auto& [label, p] = cases[k];
            const auto ref = composite(sdr, fields, sc, model, p);
            auto g32 = (*gpu)->composite(sdr, fields, sc, model, p, GpuPrecision::Fp32);
            auto g16 = (*gpu)->composite(sdr, fields, sc, model, p, GpuPrecision::Fp16);
            if (!g32 || !g16) {
                out << "rudra-gpu-parity: " << QString::fromStdString((!g32 ? g32.error() : g16.error()).message) << "\n";
                return 2;
            }
            Row r;
            r.label = label;
            const auto want = ref.buffer().span();
            const auto got32 = g32->span(), got16 = g16->span();
            for (std::size_t i = 0; i < want.size(); ++i) {
                const double d = std::abs(double(got32[i]) - double(want[i]));
                r.f32_abs = std::max(r.f32_abs, d);
                r.f32_excess = std::max(r.f32_excess, d - (kAtol + kRtol * std::abs(double(want[i]))));
                r.f16_ulp = std::max(r.f16_ulp, half_ulp(got16[i], want[i]));
                if (!python[k].empty())
                    r.py_abs = std::max(r.py_abs, std::abs(double(got32[i]) - double(python[k].span()[i])));
            }
            rows.push_back(r);
        }
    }

    // The display pass on each frame's default composite and baseline.
    struct ViewRow {
        std::string label;
        int max_code = 0;
        std::size_t off = 0;
        bool ok() const { return max_code <= 1; }
    };
    std::vector<ViewRow> vrows;
    const std::vector<std::pair<std::string, ViewParams>> views = {
        {"image 203", view_params(ViewMode::Image, 203.0)},
        {"image 1000", view_params(ViewMode::Image, 1000.0)},
        {"baseline", view_params(ViewMode::Image, 203.0, ViewSource::Baseline)},
        {"false colour", view_params(ViewMode::FalseColour, 203.0)},
        {"difference", view_params(ViewMode::Difference, 203.0, ViewSource::Model, -1.0, 0.0012, 2000.0)},
        {"wipe 0.37", view_params(ViewMode::Image, 406.0, ViewSource::Model, 0.37, 0.02)},
        {"wipe false colour", view_params(ViewMode::FalseColour, 203.0, ViewSource::Model, 0.61, 0.02)},
    };
    for (const auto& [name, f] : idx.at("frames").items()) {
        const SdrImage sdr(load(dir, f.at("sdr")));
        const Fields fields{load(dir, f.at("residual")), load(dir, f.at("highlight")), load(dir, f.at("shadow"))};
        FrameScalars sc;
        sc.shadow_weight = f.at("shadow_weight").get<float>();
        sc.curve_params = f.at("curve_params").get<std::vector<float>>();
        const auto m = composite(sdr, fields, sc, model, CompositeParams{});
        const auto b = corrected_baseline(sdr, model.corpus_ev, sc.curve_params);
        for (const auto& [vname, vp] : views) {
            const Rgb8Image want = render_view_rgb8(m, b, vp);
            auto got = (*gpu)->view(m, b, vp);
            if (!got) {
                out << "rudra-gpu-parity: " << QString::fromStdString(got.error().message) << "\n";
                return 2;
            }
            ViewRow r;
            r.label = name + " " + vname;
            for (std::size_t i = 0; i < want.rgb.size(); ++i) {
                const int d = std::abs(int(got->rgb[i]) - int(want.rgb[i]));
                r.max_code = std::max(r.max_code, d);
                r.off += d != 0;
            }
            vrows.push_back(r);
        }
    }

    // The HDR output paths (Phase 2 step 5): the values each swapchain would
    // be written with, RGBA32F against core/view.cpp and RGBA16F within 2 ulp.
    struct HdrRow {
        std::string label;
        double f32_excess = 0.0, f32_rel = 0.0, f32_abs = 0.0;
        int f16_ulp = 0;
        bool ok() const { return f32_excess <= 0.0 && f16_ulp <= kMaxHalfUlp; }
    };
    std::vector<HdrRow> hrows;
    const std::vector<std::pair<std::string, DisplayTarget>> targets = {
        {"scRGB 1000", DisplayTarget::scrgb(1000.0)},
        {"HDR10 1000", DisplayTarget::hdr10(1000.0)},
        {"EDR P3 1600", DisplayTarget::edr(1600.0)},
        {"EDR 709 1600", DisplayTarget::edr(1600.0, Primaries::Rec709)},
    };
    const std::vector<std::pair<std::string, ViewParams>> hdr_views = {
        {"image", view_params(ViewMode::Image, 10000.0)},
        {"image view peak 600", view_params(ViewMode::Image, 600.0)},
        {"false colour", view_params(ViewMode::FalseColour, 203.0)},
        {"difference", view_params(ViewMode::Difference, 203.0)},
        {"wipe", view_params(ViewMode::Image, 10000.0, ViewSource::Model, 0.37, 0.02)},
    };
    for (const auto& [name, f] : idx.at("frames").items()) {
        const SdrImage sdr(load(dir, f.at("sdr")));
        const Fields fields{load(dir, f.at("residual")), load(dir, f.at("highlight")), load(dir, f.at("shadow"))};
        FrameScalars sc;
        sc.shadow_weight = f.at("shadow_weight").get<float>();
        sc.curve_params = f.at("curve_params").get<std::vector<float>>();
        const auto m = composite(sdr, fields, sc, model, CompositeParams{});
        const auto b = corrected_baseline(sdr, model.corpus_ev, sc.curve_params);
        for (const auto& [tname, t] : targets)
            for (auto [vname, vp] : hdr_views) {
                vp.target = t;
                const PlanarBuffer want = render_view(m, b, vp);
                auto g32 = (*gpu)->view_values(m, b, vp, GpuPrecision::Fp32);
                auto g16 = (*gpu)->view_values(m, b, vp, GpuPrecision::Fp16);
                if (!g32 || !g16) {
                    out << "rudra-gpu-parity: " << QString::fromStdString((!g32 ? g32.error() : g16.error()).message) << "\n";
                    return 2;
                }
                HdrRow r;
                r.label = name + " " + tname + " " + vname;
                // PQ codes near black: the GPU's pow on a tiny base is not
                // correctly rounded, and PQ's exponents amplify it. A twentieth
                // of a 10-bit code (the swapchain's own step) is the floor there.
                const bool pq = t.path == OutputPath::Hdr10;
                constexpr double kPqFloor = 1.0 / 1023.0 / 20.0;
                for (std::size_t i = 0; i < want.span().size(); ++i) {
                    const double wv = want.span()[i], d = std::abs(double(g32->span()[i]) - wv);
                    const double d16 = std::abs(double(g16->span()[i]) - wv);
                    const double excess = d - (1e-5 + kRtol * std::abs(wv));
                    r.f32_excess = std::max(r.f32_excess, pq ? std::min(excess, d - kPqFloor) : excess);
                    r.f32_rel = std::max(r.f32_rel, d / std::max(std::abs(wv), 1e-3));
                    r.f32_abs = std::max(r.f32_abs, d);
                    const int ulp = half_ulp(g16->span()[i], float(wv));
                    r.f16_ulp = std::max(r.f16_ulp, (pq && d16 <= kPqFloor) ? std::min(ulp, kMaxHalfUlp) : ulp);
                }
                hrows.push_back(r);
            }
    }

    // The reduction ladder (Phase 2 step 6): peak exact, sum exact when the
    // GPU adds in IEEE fp32 as the ladder is specified.
    struct ReduceRow {
        std::string label;
        double peak_d = 0.0, sum_rel = 0.0;
        bool ok() const { return peak_d == 0.0 && sum_rel <= 1e-6; }
    };
    std::vector<ReduceRow> rrows;
    for (const auto& [name, f] : idx.at("frames").items()) {
        const SdrImage sdr(load(dir, f.at("sdr")));
        const Fields fields{load(dir, f.at("residual")), load(dir, f.at("highlight")), load(dir, f.at("shadow"))};
        FrameScalars sc;
        sc.shadow_weight = f.at("shadow_weight").get<float>();
        sc.curve_params = f.at("curve_params").get<std::vector<float>>();
        const auto m = composite(sdr, fields, sc, model, CompositeParams{});
        const auto b = corrected_baseline(sdr, model.corpus_ev, sc.curve_params);
        for (const auto& [label, img] : {std::pair{std::string("model"), &m}, std::pair{std::string("baseline"), &b}}) {
            const Reductions want = reduce_ladder(img->buffer());
            auto got = (*gpu)->reduce(*img);
            if (!got) {
                out << "rudra-gpu-parity: " << QString::fromStdString(got.error().message) << "\n";
                return 2;
            }
            ReduceRow r;
            r.label = name + " " + label;
            r.peak_d = std::abs(double(got->peak) - double(want.peak));
            r.sum_rel = std::abs(double(got->sum) - double(want.sum)) / std::max(std::abs(double(want.sum)), 1e-30);
            rrows.push_back(r);
        }
    }

    out << "GPU composite parity: " << QString::fromStdString(info.backend) << " on "
        << QString::fromStdString(info.device) << "\n";
    out << QString("  %1 %2 %3 %4  %5\n").arg("case", -34).arg("fp32 max|d|", 12).arg("fp16 ulp", 9)
               .arg("vs Python", 11).arg("");
    bool all = true;
    QJsonArray jrows;
    for (const auto& r : rows) {
        all = all && r.ok();
        out << QString("  %1 %2 %3 %4  %5\n").arg(QString::fromStdString(r.label), -34)
                   .arg(r.f32_abs, 12, 'e', 2).arg(r.f16_ulp, 9)
                   .arg(r.py_abs > 0 ? QString::number(r.py_abs, 'e', 2) : QString("-"), 11)
                   .arg(r.ok() ? "pass" : "FAIL");
        jrows.append(QJsonObject{{"case", QString::fromStdString(r.label)}, {"fp32_max_abs", r.f32_abs},
                                 {"fp16_max_ulp", r.f16_ulp}, {"vs_python_max_abs", r.py_abs}, {"pass", r.ok()}});
    }
    out << "  bound: fp32 atol " << kAtol << " rtol " << kRtol << "; fp16 " << kMaxHalfUlp << " half ulp\n";
    out << "Display pass, RGBA8, against core/view.cpp\n";
    QJsonArray jviews;
    for (const auto& r : vrows) {
        all = all && r.ok();
        out << QString("  %1 %2 %3  %4\n").arg(QString::fromStdString(r.label), -34)
                   .arg(QString("max %1 code").arg(r.max_code), 12).arg(QString("%1 off").arg(r.off), 9)
                   .arg(r.ok() ? "pass" : "FAIL");
        jviews.append(QJsonObject{{"case", QString::fromStdString(r.label)}, {"max_code", r.max_code},
                                  {"values_off", qint64(r.off)}, {"pass", r.ok()}});
    }
    out << "  bound: 1 code in 8 bits\n";
    out << "HDR output paths, against core/view.cpp\n";
    QJsonArray jhdr;
    int worst_hdr_ulp = 0;
    double worst_hdr_rel = 0.0;
    bool hdr_ok = true;
    for (const auto& r : hrows) {
        hdr_ok = hdr_ok && r.ok();
        worst_hdr_ulp = std::max(worst_hdr_ulp, r.f16_ulp);
        worst_hdr_rel = std::max(worst_hdr_rel, r.f32_rel);
        if (!r.ok())
            out << QString("  %1 fp32 rel %2  fp16 %3 ulp  FAIL\n").arg(QString::fromStdString(r.label), -44)
                       .arg(r.f32_rel, 0, 'e', 2).arg(r.f16_ulp);
        jhdr.append(QJsonObject{{"case", QString::fromStdString(r.label)}, {"fp32_max_rel", r.f32_rel}, {"fp32_max_abs", r.f32_abs},
                                {"fp16_max_ulp", r.f16_ulp}, {"pass", r.ok()}});
    }
    all = all && hdr_ok;
    out << QString("  %1 cases (scRGB, HDR10, EDR P3, EDR 709 x 5 views x %2 frames): fp32 max rel %3, fp16 max %4 ulp  %5\n")
               .arg(hrows.size()).arg(idx.at("frames").size()).arg(worst_hdr_rel, 0, 'e', 2).arg(worst_hdr_ulp)
               .arg(hdr_ok ? "pass" : "FAIL");
    out << "  bound: fp32 1e-5 + " << kRtol << " |ref|; fp16 " << kMaxHalfUlp << " half ulp; HDR10 codes also pass within 1/20 of a 10-bit step\n";
    out << "Reductions (peak, fp32 sum of max RGB), against core/view.cpp reduce_ladder\n";
    QJsonArray jred;
    for (const auto& r : rrows) {
        all = all && r.ok();
        out << QString("  %1 peak |d| %2  sum rel %3  %4\n").arg(QString::fromStdString(r.label), -24)
                   .arg(r.peak_d, 0, 'e', 1).arg(r.sum_rel, 0, 'e', 1).arg(r.ok() ? "pass" : "FAIL");
        jred.append(QJsonObject{{"case", QString::fromStdString(r.label)}, {"peak_abs", r.peak_d},
                                {"sum_rel", r.sum_rel}, {"pass", r.ok()}});
    }
    out << "  bound: peak exact; sum 1e-6 relative\n";
    out << "  => " << (all ? "PASS" : "FAIL") << "\n";

    QJsonArray jbench;
    if (cli.isSet(bench_opt)) {
        out << "Composite pass, RGBA16F target, median of 50 after one warm-up\n";
        for (auto [w, h] : {std::pair{1920, 1080}, std::pair{3840, 2160}}) {
            auto t = (*gpu)->benchmark(w, h, 50);
            if (!t) {
                out << "  " << w << "x" << h << "  " << QString::fromStdString(t.error().message) << "\n";
                continue;
            }
            out << QString("  %1x%2  gpu %3  wall %4 ms\n").arg(w).arg(h)
                       .arg(t->has_gpu_timestamps ? QString::number(t->gpu_ms, 'f', 3) + " ms" : QString("n/a"))
                       .arg(t->wall_ms, 0, 'f', 3);
            out << "BENCH composite " << a << " " << w << "x" << h << " " << (t->has_gpu_timestamps ? t->gpu_ms : -1.0)
                << " " << t->wall_ms << "\n";
            jbench.append(QJsonObject{{"size", QString("%1x%2").arg(w).arg(h)},
                                      {"gpu_ms", t->has_gpu_timestamps ? QJsonValue(t->gpu_ms) : QJsonValue()},
                                      {"wall_ms", t->wall_ms}});
        }
            out << "Composite + display pass (a slider move: RGBA32F composite, RGBA16F scRGB picture), median of 50\n";
        for (auto [w, h] : {std::pair{1920, 1080}, std::pair{3840, 2160}}) {
            auto t = (*gpu)->benchmark_view(w, h, 50);
            if (!t) {
                out << "  " << w << "x" << h << "  " << QString::fromStdString(t.error().message) << "\n";
                continue;
            }
            out << QString("  %1x%2  gpu %3  wall %4 ms\n").arg(w).arg(h)
                       .arg(t->has_gpu_timestamps ? QString::number(t->gpu_ms, 'f', 3) + " ms" : QString("n/a"))
                       .arg(t->wall_ms, 0, 'f', 3);
            out << "BENCH view " << a << " " << w << "x" << h << " " << (t->has_gpu_timestamps ? t->gpu_ms : -1.0) << " "
                << t->wall_ms << "\n";
            jbench.append(QJsonObject{{"size", QString("%1x%2").arg(w).arg(h)}, {"pass", "composite+view"},
                                      {"gpu_ms", t->has_gpu_timestamps ? QJsonValue(t->gpu_ms) : QJsonValue()},
                                      {"wall_ms", t->wall_ms}});
        }
    }
    if (cli.isSet(report_opt)) {
        QFile f(cli.value(report_opt));
        if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
            f.write(QJsonDocument(QJsonObject{{"backend", QString::fromStdString(info.backend)},
                                              {"device", QString::fromStdString(info.device)},
                                              {"api", a}, {"pass", all}, {"cases", jrows}, {"views", jviews}, {"hdr", jhdr}, {"reduce", jred}, {"bench", jbench}})
                        .toJson());
    }
    return all ? 0 : 1;
}
