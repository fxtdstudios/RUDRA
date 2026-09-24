#pragma once
// The app's look: the Pro boards (app/theme/pro.css). Geist and Geist Mono
// from the resources (Plex too, for the scope labels), a palette and the
// style sheet generated from pro.css at build time (cmake/rudra_theme.cmake).
// No colour is written in the app's sources; they all come from the tokens.

#include <QColor>
#include <QMap>
#include <QString>
#include <QStringList>

class QApplication;

namespace rudra::app {

struct ThemeReport {
    QStringList fonts_loaded;        // resource paths that registered
    QStringList problems;            // anything that will not look as the Studio does
    QString ui_family, mono_family;  // from the tokens
    QString style;                   // the QStyle under the sheet
    QMap<QString, QString> tokens;   // name -> value, ui/theme.css :root
    bool ok() const { return problems.isEmpty(); }
};

// Fusion style, the fonts, the palette and the style sheet on the whole app.
ThemeReport apply_theme(QApplication& app);

// One token as a colour (the scope and viewer painters use these).
QColor token_colour(const ThemeReport& t, const QString& name);
// The same from the embedded tokens, for painters that have no report.
QColor theme_colour(const QString& name);

// The report as JSON, for `RUDRA --theme-check out.json` (CI and the gates).
QByteArray theme_report_json(const ThemeReport& t);

}  // namespace rudra::app
