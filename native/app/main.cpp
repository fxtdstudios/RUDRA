// RUDRA: the Qt application.
//
// The window is app/main_window.cpp: the Studio's menubar and actions
// (engine/actions, Phase 3 step 2) around the QRhi viewer (render/, ADR-010)
// and the frame engine. Its look is app/theme.cpp, from ui/theme.css (step 1).
//
//   RUDRA [package [still-or-folder]]   no package: the first-run check once, then the last or picked one
//   RUDRA --theme-check out.json      the fonts, weights and style as resolved here
//   RUDRA --grab out.png [...]        the window as drawn, then quit
//   RUDRA --tab grade|deliver ...     open on that inspector tab
//   RUDRA --workflow-check report.json --package P --frames DIR [--backend libtorch/cuda] [--out DIR] [--movie CLIP]
//                                     the Studio workflow, scripted, in this window (Phase 3 step 12)

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QSettings>
#include <QTimer>

#include <algorithm>

#include "main_window.hpp"
#include "workflow_check.hpp"
#include "theme.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("RUDRA");
    QApplication::setOrganizationName("FXTD Studios");
#ifdef RUDRA_VERSION_LABEL
    QApplication::setApplicationVersion(QStringLiteral(RUDRA_VERSION_LABEL));
#endif
    const rudra::app::ThemeReport theme = rudra::app::apply_theme(app);
    // RUDRA --theme-check out.json: the look as this machine resolves it
    // (fonts, weights, style), for CI and the gates; exit 1 on a problem.
    const QStringList args = QApplication::arguments();
    if (const qsizetype i = args.indexOf("--theme-check"); i >= 0) {
        QFile f(i + 1 < args.size() ? args[i + 1] : QString("theme-check.json"));
        if (f.open(QIODevice::WriteOnly | QIODevice::Truncate)) f.write(rudra::app::theme_report_json(theme));
        return theme.ok() ? 0 : 1;
    }
    // RUDRA --grab out.png [package [source]]: the window as drawn, for the
    // side-by-side review against the Studio and the Pro-direction boards.
    QString grab_to;
    QStringList rest = args.mid(1);
    if (const qsizetype i = rest.indexOf("--grab"); i >= 0) {
        grab_to = i + 1 < rest.size() ? rest[i + 1] : QString("rudra-window.png");
        rest.remove(i, std::min<qsizetype>(2, rest.size() - i));
    }
    // The scripted workflow: a window of its own settings, shown, driven, closed.
    if (const qsizetype i = args.indexOf("--workflow-check"); i >= 0) {
        auto value = [&](const char* flag, const QString& fallback = {}) {
            const qsizetype k = args.indexOf(flag);
            return k >= 0 && k + 1 < args.size() ? args[k + 1] : fallback;
        };
        rudra::app::WorkflowArgs wa;
        wa.report = i + 1 < args.size() ? args[i + 1] : QString("rudra-workflow.json");
        wa.package = value("--package");
        wa.frames = value("--frames");
        wa.backend = value("--backend");
        wa.out = value("--out", QDir::temp().filePath("rudra-workflow-masters"));
        wa.movie = value("--movie");
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, QDir::temp().filePath("rudra-workflow-settings"));
        QSettings().clear();
        QSettings().setValue("firstRun/done", true);
        rudra::app::MainWindow w;
        w.show();
        return rudra::app::run_workflow_check(w, wa);
    }
    // --grab-delay ms: time for a model and a frame to arrive (default 800).
    int grab_delay = 800;
    if (const qsizetype i = rest.indexOf("--grab-delay"); i >= 0 && i + 1 < rest.size()) {
        grab_delay = rest[i + 1].toInt();
        rest.remove(i, 2);
    }
    // --dialog welcome|export|models: grab that sheet instead of the window (the reviews).
    QString dialog;
    if (const qsizetype i = rest.indexOf("--dialog"); i >= 0 && i + 1 < rest.size()) {
        dialog = rest[i + 1];
        rest.remove(i, 2);
    }
    rudra::app::MainWindow w;
    w.restore_settings();   // the window, rails, tab, container and Render fields of the last run
    if (const qsizetype t = rest.indexOf("--tab"); t >= 0 && t + 1 < rest.size()) {   // for the review grabs
        w.show_tab(rest[t + 1]);
        rest.remove(t, 2);
    }
    w.show();
    // A package named on the command line, else the bare start: the first-run
    // check once, then the package used last or the catalog's pick (step 10).
    if (rest.size() > 0) w.open_package(rest[0]);
    else w.boot();
    if (rest.size() > 1) w.open_source(rest[1]);
    if (!grab_to.isEmpty()) {
        QTimer::singleShot(grab_delay, &w, [&w, grab_to, dialog] {
            if (!dialog.isEmpty()) {
                if (dialog == "welcome") w.open_first_run();
                else if (dialog == "export") w.open_export_sheet();
                else w.open_model_manager();
                QTimer::singleShot(400, &w, [&w, grab_to, dialog] {
                    const char* id = dialog == "welcome" ? "firstRun" : dialog == "export" ? "exportSheet" : "modelManager";
                    if (auto* d = w.findChild<QWidget*>(id)) d->grab().save(grab_to);
                    QApplication::exit(0);
                });
                return;
            }
            w.grab().save(grab_to);
            QApplication::exit(0);
        });
    }
    return QApplication::exec();
}
