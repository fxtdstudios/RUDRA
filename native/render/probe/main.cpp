// rudra-hdr-probe: Gate B (docs/NATIVE_ARCHITECTURE.md section 12, day 5).
//
//   rudra-hdr-probe [--api d3d12|d3d11|metal|vulkan|gl] [--format scrgb|hdr10|p3|sdr]
//                   [--report file.json] [--frames N] [--screen N] [--list-screens]
//
// Opens a QRhi window with a test card at known luminance, reads the swapchain
// back and reports whether the 1 000-nit patch left the pipeline above SDR
// white. Esc quits, R reads back again (after moving the window to another
// display, say). With --frames N it exits after N frames: 0 on PASS, 1 on FAIL.

#include <QCommandLineParser>
#include <QGuiApplication>
#include <QScreen>
#include <QTextStream>

#if QT_CONFIG(vulkan)
#include <QVulkanInstance>
#endif

#include "hdr_probe_window.hpp"

using namespace rudra::probe;

namespace {

Api default_api() {
#if defined(Q_OS_WIN)
    return Api::D3D12;
#elif defined(Q_OS_MACOS)
    return Api::Metal;
#else
    return Api::Vulkan;
#endif
}

Want default_want() {
#if defined(Q_OS_MACOS)
    return Want::DisplayP3;
#else
    return Want::ScRgb;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName("rudra-hdr-probe");

    QCommandLineParser cli;
    cli.setApplicationDescription("RUDRA Gate B: HDR swapchain probe");
    cli.addHelpOption();
    QCommandLineOption api_opt("api", "d3d12, d3d11, metal, vulkan, gl or null", "api", to_string(default_api()));
    QCommandLineOption fmt_opt("format", "scrgb, hdr10, p3 or sdr", "format", to_string(default_want()));
    QCommandLineOption report_opt("report", "write the JSON report here", "file");
    QCommandLineOption frames_opt("frames", "exit after N frames (0 on PASS)", "n", "0");
    QCommandLineOption screen_opt("screen", "open on this display (index from --list-screens)", "n", "-1");
    QCommandLineOption list_opt("list-screens", "print the displays and exit");
    cli.addOptions({api_opt, fmt_opt, report_opt, frames_opt, screen_opt, list_opt});
    cli.process(app);

    Options opt;
    const QString a = cli.value(api_opt).toLower();
    opt.api = a == "d3d12" ? Api::D3D12 : a == "d3d11" ? Api::D3D11 : a == "metal" ? Api::Metal
            : a == "vulkan" ? Api::Vulkan : a == "null" ? Api::Null : Api::OpenGL;
    const QString f = cli.value(fmt_opt).toLower();
    opt.want = f == "hdr10" ? Want::Hdr10 : f == "p3" ? Want::DisplayP3 : f == "sdr" ? Want::Sdr : Want::ScRgb;
    opt.report_path = cli.value(report_opt);
    opt.exit_after_frames = cli.value(frames_opt).toInt();
    opt.screen = cli.value(screen_opt).toInt();

    const auto screens = QGuiApplication::screens();
    if (cli.isSet(list_opt)) {
        QTextStream out(stdout);
        for (int i = 0; i < screens.size(); ++i)
            out << i << "  " << screens[i]->name() << "  " << screens[i]->model() << "  "
                << screens[i]->geometry().width() << "x" << screens[i]->geometry().height()
                << (screens[i] == QGuiApplication::primaryScreen() ? "  (primary)" : "") << "\n";
        return 0;
    }

#if QT_CONFIG(vulkan)
    QVulkanInstance inst;
    if (opt.api == Api::Vulkan) {
        inst.setExtensions(QRhiVulkanInitParams::preferredInstanceExtensions());
        if (!inst.create()) {
            QTextStream(stderr) << "rudra-hdr-probe: no Vulkan instance, falling back to OpenGL\n";
            opt.api = Api::OpenGL;
        }
    }
#else
    if (opt.api == Api::Vulkan) opt.api = Api::OpenGL;
#endif

    HdrProbeWindow window(opt);
#if QT_CONFIG(vulkan)
    if (opt.api == Api::Vulkan) window.setVulkanInstance(&inst);
#endif
    if (opt.screen >= 0 && opt.screen < screens.size()) {
        QScreen* target = screens[opt.screen];
        window.setScreen(target);
        const QRect g = target->availableGeometry();
        window.setPosition(g.center() - QPoint(window.width() / 2, window.height() / 2));
    }
    window.show();
    return app.exec();
}
