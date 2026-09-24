// rudra-viewer-check: the viewer window (render/viewer_window.hpp) end to end,
// through its real swapchain (Phase 2 step 9).
//
//   rudra-viewer-check [--api d3d12|d3d11|metal|vulkan|gl] [--golden DIR] [--report FILE]
//   rudra-viewer-check --card [--api ...] [--report FILE]
//
// Default: an SDR swapchain. A frame from the viewer goldens goes in as fields,
// through the window's composite, display and blit passes, and the swapchain
// is read back: at fit, 2x and 1:1 every picture pixel must equal
// core/view.cpp on core/composite.cpp within 1 code, for the image, false
// colour and wipe views, and with the guides on (core/guides.cpp: safe areas,
// centre cross, a 2.39 and a 4:3 mask); the surround must be #121212.
//
// --card: Gate B through the real display pass. An HDR swapchain if the
// display has one; a card of known luminance (10, 100, 203, 600, 1 000 and
// 2 000 nits) goes in as a composited picture at the display's own peak, and
// the swapchain is read back at each patch. PASS on an HDR swapchain whose
// peak is more than a stop above SDR white when every patch reads as its own
// luminance up to that peak and as the peak above it (within 2 %). Exit 0 on
// PASS.

#include <QCommandLineParser>
#include <QFile>
#include <QFloat16>
#include <QGuiApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTextStream>
#include <QTimer>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>

#include <nlohmann/json.hpp>

#include "rudra/core/baseline.hpp"
#include "rudra/core/composite.hpp"
#include "rudra/core/guides.hpp"
#include "rudra/core/half.hpp"
#include "rudra/core/hdr10.hpp"
#include "rudra/core/view.hpp"
#include "rudra/platform/npy.hpp"
#include "rudra/render/viewer_window.hpp"

using namespace rudra;
namespace fs = std::filesystem;

namespace {

GpuApi api_of(const QString& a) {
    if (a == "d3d12") return GpuApi::D3D12;
    if (a == "d3d11") return GpuApi::D3D11;
    if (a == "metal") return GpuApi::Metal;
    if (a == "vulkan") return GpuApi::Vulkan;
    if (a == "gl") return GpuApi::OpenGL;
    return GpuApi::Auto;
}

// A frame of the viewer goldens, as /api/frame sent it.
bool load_frame(const fs::path& dir, ViewerFrame& out) {
    std::ifstream in(dir / "index.json");
    if (!in) return false;
    const auto idx = nlohmann::json::parse(in);
    const auto& f = idx.at("frames").at(0);
    const auto& hd = f.at("header");
    const int w = hd.at("width").get<int>(), h = hd.at("height").get<int>();
    const std::size_t n = std::size_t(w) * std::size_t(h);
    std::ifstream b(dir / f.at("body").get<std::string>(), std::ios::binary);
    const std::vector<unsigned char> body((std::istreambuf_iterator<char>(b)), {});
    const std::size_t off_shadow = hd.at("offsets").at("shadow").get<std::size_t>();
    const std::size_t off_sdr = hd.at("offsets").at("sdr").get<std::size_t>();
    auto half_at = [&](std::size_t k) { return half_to_float(std::uint16_t(body[k] | (body[k + 1] << 8))); };
    PlanarBuffer sdr(3, h, w), res(3, h, w), hi(1, h, w), sh(1, h, w);
    for (std::size_t i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) {
            res.plane(c)[i] = half_at(i * 8 + std::size_t(c) * 2);
            sdr.plane(c)[i] = float(body[off_sdr + i * 3 + std::size_t(c)]) / 255.0f;
        }
        hi.plane(0)[i] = half_at(i * 8 + 6);
        sh.plane(0)[i] = half_at(off_shadow + i * 2);
    }
    out.sdr = SdrImage(std::move(sdr));
    out.fields = Fields{std::move(res), std::move(hi), std::move(sh)};
    out.scalars.shadow_weight = hd.at("shadow_weight").get<float>();
    if (hd.at("curve").is_array()) out.scalars.curve_params = hd.at("curve").get<std::vector<float>>();
    out.model = {hd.at("log_scale").get<float>(), hd.at("max_hdr").get<float>(), hd.at("corpus_ev").get<float>()};
    return true;
}

float srgb_eotf(float c) { return c > 0.04045f ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f; }

// One grab pixel as absolute nits (the green channel), from its format.
double nits_at(const ViewerWindow::Grab& g, const DisplayTarget& t, int x, int y) {
    const unsigned char* p = g.bytes.data() + (std::size_t(y) * std::size_t(g.width) + std::size_t(x)) * std::size_t(g.bytes_per_pixel);
    if (g.format == "RGBA16F") {
        qfloat16 v;
        std::memcpy(&v, p + 2, 2);
        return double(float(v)) * t.unit_nits;
    }
    if (g.format == "RGB10A2") {
        std::uint32_t word;
        std::memcpy(&word, p, 4);
        return pq_eotf(float((word >> 10) & 0x3ffu) / 1023.0f);
    }
    return double(srgb_eotf(float(p[1]) / 255.0f)) * 203.0;
}

}  // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);
    QCommandLineParser cli;
    cli.addHelpOption();
    QCommandLineOption api_opt("api", "d3d12, d3d11, metal, vulkan or gl", "api", "auto");
    QCommandLineOption golden_opt("golden", "viewer golden folder", "dir", RUDRA_VIEWER_GOLDEN_DIR);
    QCommandLineOption report_opt("report", "write a JSON report", "file");
    QCommandLineOption card_opt("card", "Gate B through the viewer's display pass (HDR swapchain)");
    QCommandLineOption dump_opt("dump", "write the grab and the expected picture of every failing case (PPM)", "dir");
    cli.addOptions({api_opt, golden_opt, report_opt, card_opt, dump_opt});
    cli.process(app);
    QTextStream out(stdout);
    const bool card = cli.isSet(card_opt);

    ViewerWindow win(api_of(cli.value(api_opt)), /*prefer_hdr=*/card);
    QJsonObject report;
    bool pass = true;
    int step = 0;

    ViewerFrame frame;
    NetworkLinearImage cpu_model, cpu_base;
    std::vector<std::pair<std::string, ViewParams>> views;
    std::vector<std::pair<std::string, double>> zooms;   // name, scale (0: fit)
    std::vector<float> card_nits = {10, 100, 203, 600, 1000, 2000};
    constexpr int kCardW = 1280, kCardH = 720;

    if (card) {
        // Six patches across the top 60 %, surround 18 % grey of 203 nits.
        PlanarBuffer px(3, kCardH, kCardW, 0.18f * 203.0f / 10000.0f);
        for (int y = 0; y < int(kCardH * 0.6); ++y)
            for (int x = 0; x < kCardW; ++x) {
                const float v = card_nits[std::size_t(x * 6 / kCardW)] / 10000.0f;
                for (int c = 0; c < 3; ++c) px.at(c, y, x) = v;
            }
        const NetworkLinearImage img(std::move(px));
        win.set_composited(img, img);
        ViewParams v;
        v.display_nits = 10000.0;   // the ceiling is the display's own peak
        win.set_view(v);
        win.resize(kCardW + 56, kCardH + 56);
    } else {
        if (!load_frame(fs::path(cli.value(golden_opt).toStdString()), frame)) {
            out << "rudra-viewer-check: no viewer goldens at " << cli.value(golden_opt) << "\n";
            return 2;
        }
        cpu_model = composite(frame.sdr, frame.fields, frame.scalars, frame.model, CompositeParams{});
        cpu_base = corrected_baseline(frame.sdr, frame.model.corpus_ev, frame.scalars.curve_params);
        win.set_frame(frame);
        win.set_composite(CompositeParams{});
        views = {{"image 203", view_params(ViewMode::Image, 203.0)},
                 {"image 1000", view_params(ViewMode::Image, 1000.0)},
                 {"false colour", view_params(ViewMode::FalseColour, 203.0)},
                 {"wipe 0.37", view_params(ViewMode::Image, 406.0, ViewSource::Model, 0.37, 0.02)},
                 {"guides 2.39", view_params(ViewMode::Image, 203.0)},
                 {"guides 4:3", view_params(ViewMode::FalseColour, 203.0)}};
        zooms = {{"fit", 0.0}, {"2x", 2.0}, {"1:1", -1.0}};   // -1: actual pixels (device 1:1)
        win.resize(400, 300);   // an 80x48 frame lands on whole pixels at fit and at 2x
    }

    const auto info = [&] {
        const ViewerStatus s = win.status();
        report["backend"] = QString::fromStdString(s.backend);
        report["swapchain"] = QString::fromStdString(s.swapchain);
        report["peak_nits"] = s.target.peak_nits;
        return s;
    };

    QJsonArray rows;
    std::function<void()> next;
    bool no_hdr = false;
    auto finish = [&] {
        report["cases"] = rows;
        report["verdict"] = no_hdr ? "NO-HDR" : pass ? "PASS" : "FAIL";
        const QByteArray json = QJsonDocument(report).toJson(QJsonDocument::Indented);
        if (cli.isSet(report_opt)) {
            QFile f(cli.value(report_opt));
            if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) f.write(json);
        }
        out << "  => " << (no_hdr ? "NO-HDR (not a failure: this API has no HDR swapchain here)" : pass ? "PASS" : "FAIL") << "\n";
        out.flush();
        QCoreApplication::exit(no_hdr ? 2 : pass ? 0 : 1);
    };

    auto check_card = [&](const ViewerWindow::Grab& g) {
        const ViewerStatus s = info();
        out << "Gate B through the viewer: " << QString::fromStdString(s.backend) << ", swapchain "
            << QString::fromStdString(s.swapchain) << " (" << QString::fromStdString(g.format) << "), display peak "
            << s.target.peak_nits << " nits (from " << QString::fromStdString(s.peak_from) << ")\n";
        const PlacedRect r = place(win.viewport(), {double(win.width()), double(win.height())}, {kCardW, kCardH});
        const double dpr = win.devicePixelRatio();
        // Each patch must read as its own luminance up to the display's peak and
        // as the peak above it: clipped, never tone-mapped (ADR-005).
        bool all_exact = true;
        QJsonArray patches;
        for (int i = 0; i < 6; ++i) {
            const int x = int((r.left + r.width * (i + 0.5) / 6.0) * dpr);
            const int y = int((r.top + r.height * 0.3) * dpr);
            const double n = nits_at(g, s.target, std::clamp(x, 0, g.width - 1), std::clamp(y, 0, g.height - 1));
            const double expect = std::min(double(card_nits[std::size_t(i)]), s.target.peak_nits);
            const bool ok = std::abs(n - expect) <= 0.02 * expect + 0.5;
            all_exact = all_exact && ok;
            out << QString("  patch %1 nits -> %2 nits (expected %3)  %4\n").arg(card_nits[std::size_t(i)], 6)
                       .arg(n, 0, 'f', 1).arg(expect, 0, 'f', 1).arg(ok ? "ok" : "off");
            patches.append(QJsonObject{{"target_nits", card_nits[std::size_t(i)]}, {"expected_nits", expect},
                                       {"swapchain_nits", std::round(n * 10) / 10}, {"ok", ok}});
        }
        report["patches"] = patches;
        const bool hdr = s.target.path != OutputPath::SdrPqSimulation;
        report["peak_from"] = QString::fromStdString(s.peak_from);
        if (!hdr) {
            no_hdr = true;
            report["verdict_note"] = "SDR swapchain: this API offers no HDR format here";
            finish();
            return;
        }
        pass = all_exact && s.target.peak_nits > 2.0 * 203.0 && s.peak_from != "placeholder";
        report["verdict_note"] = s.peak_from == "placeholder" ? "the swapchain reports Qt's placeholder peak, not the display's"
                                 : pass ? "every patch at its luminance up to the display peak and clipped above it, "
                                          "through the display pass; confirm on the glass"
                                        : "the display pass did not carry the card to the swapchain exactly";
        finish();
    };

    auto check_parity = [&](const ViewerWindow::Grab& g) {
        const auto& [vname, vp] = views[std::size_t(step / int(zooms.size()))];
        const auto& [zname, zscale] = zooms[std::size_t(step % int(zooms.size()))];
        const ViewerStatus s = info();
        const int fw = frame.sdr.width(), fh = frame.sdr.height();
        const double dpr = win.devicePixelRatio();
        const PlacedRect r = place(win.viewport(), {double(win.width()), double(win.height())}, {double(fw), double(fh)});
        ViewParams want_p = vp;
        want_p.target = s.target;
        const Rgb8Image want = render_view_rgb8(cpu_model, cpu_base, want_p);
        const bool bgra = g.format == "BGRA8";
        // Device pixel centres against the picture's texels. At a fractional
        // device pixel ratio (150 % display scale) a pixel centre can sit exactly
        // on a texel edge, where nearest sampling may take either texel, so both
        // are accepted there; likewise the surround or the edge texel on the
        // rectangle's border.
        const double L = r.left * dpr, T = r.top * dpr, W = r.width * dpr, H = r.height * dpr;
        constexpr double kTie = 1e-3;
        auto candidates = [](double u, int n, std::vector<int>& list) {   // texel indices, -1 for outside
            list.clear();
            for (double e : {-kTie, kTie}) {
                const double v = std::floor(u + e);
                const int k = (v < 0 || v >= n) ? -1 : int(v);
                if (std::find(list.begin(), list.end(), k) == list.end()) list.push_back(k);
            }
        };
        std::vector<int> cx, cy;
        int worst = 0, surround_off = 0;
        std::size_t off = 0;
        std::vector<unsigned char> got_img(std::size_t(g.width) * std::size_t(g.height) * 3),
            want_img(std::size_t(g.width) * std::size_t(g.height) * 3, 0x12);
        QJsonArray samples;   // the first mismatches, for the report
        for (int y = 0; y < g.height; ++y)
            for (int x = 0; x < g.width; ++x) {
                const unsigned char* p = g.bytes.data() + (std::size_t(y) * std::size_t(g.width) + std::size_t(x)) * 4;
                const int rgb[3] = {bgra ? p[2] : p[0], p[1], bgra ? p[0] : p[2]};
                const std::size_t o3 = (std::size_t(y) * std::size_t(g.width) + std::size_t(x)) * 3;
                for (int c = 0; c < 3; ++c) got_img[o3 + std::size_t(c)] = (unsigned char)rgb[c];
                candidates((x + 0.5 - L) / W * fw, fw, cx);
                candidates((y + 0.5 - T) / H * fh, fh, cy);
                const GuideSample gs = guide_at(win.guides(), L, T, W, H, x, y);
                int best = 1 << 20;
                bool surround_ok = false;
                for (int ty : cy)
                    for (int tx : cx) {
                        if (tx < 0 || ty < 0) {
                            int d = 0;
                            for (int c : rgb) d = std::max(d, std::abs(c - 0x12));
                            if (d == 0) surround_ok = true;
                            if (cx.size() == 1 && cy.size() == 1) surround_off = std::max(surround_off, d);
                            continue;
                        }
                        int d = 0, e3[3];
                        for (int c = 0; c < 3; ++c) {
                            const float v = float(want.rgb[(std::size_t(ty) * std::size_t(fw) + std::size_t(tx)) * 3 + std::size_t(c)]) / 255.0f;
                            const int e = int(std::lround(std::clamp(apply_guides(v, 1.0f, gs), 0.0f, 1.0f) * 255.0f));
                            e3[c] = e;
                            d = std::max(d, std::abs(rgb[c] - e));
                        }
                        if (d < best)
                            for (int c = 0; c < 3; ++c) want_img[o3 + std::size_t(c)] = (unsigned char)e3[c];
                        best = std::min(best, d);
                    }
                if (surround_ok) best = 0;
                if (best == 1 << 20) continue;   // only surround candidates: counted above
                if (best > 1 && samples.size() < 8)
                    samples.append(QJsonObject{{"x", x}, {"y", y}, {"got", QJsonArray{rgb[0], rgb[1], rgb[2]}},
                                               {"want", QJsonArray{want_img[o3], want_img[o3 + 1], want_img[o3 + 2]}},
                                               {"texel", QJsonArray{cx[0], cy[0]}}});
                worst = std::max(worst, best);
                off += best != 0;
            }
        const bool ok = worst <= 1 && surround_off == 0 && s.target.path == OutputPath::SdrPqSimulation;
        pass = pass && ok;
        out << QString("  %1 %2 max %3 code, %4 off, surround %5  %6\n").arg(QString::fromStdString(vname), -14)
                   .arg(QString::fromStdString(zname), -4).arg(worst).arg(off).arg(surround_off).arg(ok ? "pass" : "FAIL");
        QJsonObject row{{"case", QString::fromStdString(vname + " " + zname)}, {"max_code", worst},
                        {"values_off", qint64(off)}, {"surround_off", surround_off}, {"pass", ok},
                        {"grab", QJsonArray{g.width, g.height}}, {"grab_format", QString::fromStdString(g.format)},
                        {"window", QJsonArray{win.width(), win.height()}},
                        {"rect_device", QJsonArray{L, T, W, H}}, {"first_mismatches", samples}};
        rows.append(row);
        if (!ok && cli.isSet(dump_opt)) {
            const QString dir = cli.value(dump_opt);
            fs::create_directories(dir.toStdString());
            auto ppm = [&](const std::string& name, const std::vector<unsigned char>& img) {
                std::ofstream f(fs::path(dir.toStdString()) / name, std::ios::binary);
                f << "P6\n" << g.width << " " << g.height << "\n255\n";
                f.write(reinterpret_cast<const char*>(img.data()), std::streamsize(img.size()));
            };
            std::string base = vname + "_" + zname;
            for (char& ch : base)
                if (!std::isalnum(static_cast<unsigned char>(ch))) ch = '_';
            ppm(base + "_got.ppm", got_img);
            ppm(base + "_want.ppm", want_img);
        }
        ++step;
        next();
    };

    next = [&] {
        if (card) {
            win.grab(check_card);
            return;
        }
        if (step == 0) {
            const ViewerStatus s = info();
            out << "Viewer window parity: " << QString::fromStdString(s.backend) << ", swapchain "
                << QString::fromStdString(s.swapchain) << ", frame " << frame.sdr.width() << "x" << frame.sdr.height()
                << " in a " << win.width() << "x" << win.height() << " window, device pixel ratio "
                << win.devicePixelRatio() << "\n";
            report["device_pixel_ratio"] = win.devicePixelRatio();
        }
        if (step >= int(views.size() * zooms.size())) {
            out << "  bound: 1 code in 8 bits against core/view.cpp on core/composite.cpp\n";
            finish();
            return;
        }
        const std::string& vname = views[std::size_t(step / int(zooms.size()))].first;
        GuideOptions gopt;
        if (vname.rfind("guides", 0) == 0) {
            gopt.action_safe = gopt.title_safe = gopt.centre = true;
            gopt.aspect = vname == "guides 2.39" ? 2.39 : 4.0 / 3.0;
        }
        win.set_guides(gopt);
        win.set_view(views[std::size_t(step / int(zooms.size()))].second);
        const double z = zooms[std::size_t(step % int(zooms.size()))].second;
        if (z > 0) win.set_viewport({z, 0.0, 0.0});
        else if (z < 0) win.zoom_actual();
        else win.zoom_fit();
        // Two frames: the one that renders the change, then the grab.
        QTimer::singleShot(50, [&] { win.grab(check_parity); });
    };

    win.show();
    QTimer::singleShot(300, [&] { next(); });   // exposed and a few frames presented
    QTimer::singleShot(30000, [&] {
        out << "rudra-viewer-check: timed out\n";
        QCoreApplication::exit(2);
    });
    return QGuiApplication::exec();
}
