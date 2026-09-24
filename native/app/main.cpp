// RUDRA: the Qt application.
//
// The window is app/main_window.cpp: the Studio's menubar and actions
// (engine/actions, Phase 3 step 2) around the QRhi viewer (render/, ADR-010)
// and the frame engine. Its look is app/theme.cpp, from ui/theme.css (step 1).
//
//   RUDRA [package [still-or-folder]]
//   RUDRA --theme-check out.json      the fonts, weights and style as resolved here
//   RUDRA --grab out.png [...]        the window as drawn, then quit
//   RUDRA --tab grade|deliver ...     open on that inspector tab

#include <QApplication>
#include <QFile>
#include <QTimer>

#include <algorithm>

#include "main_window.hpp"
#include "theme.hpp"

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("RUDRA");
    QApplication::setOrganizationName("FXTD Studios");
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
    rudra::app::MainWindow w;
    if (const qsizetype t = rest.indexOf("--tab"); t >= 0 && t + 1 < rest.size()) {   // for the review grabs
        w.show_tab(rest[t + 1]);
        rest.remove(t, 2);
    }
    if (rest.size() > 0) w.open_package(rest[0]);
    if (rest.size() > 1) w.open_source(rest[1]);
    w.show();
    if (!grab_to.isEmpty()) {
        QTimer::singleShot(800, &w, [&w, grab_to] {
            w.grab().save(grab_to);
            QApplication::exit(0);
        });
    }
    return QApplication::exec();
}
