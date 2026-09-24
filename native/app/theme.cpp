#include "theme.hpp"

#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFontDatabase>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPalette>
#include <QSet>

#include <algorithm>
#include <QStyle>
#include <QStyleFactory>

namespace rudra::app {
namespace {

QString read_resource(const QString& path) {
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) return {};
    return QString::fromUtf8(f.readAll());
}

// The weights a family offers once registered, by style name.
QSet<int> weights_of(const QString& family) {
    QSet<int> w;
    for (const QString& style : QFontDatabase::styles(family)) {
        if (QFontDatabase::italic(family, style)) continue;
        w.insert(QFontDatabase::weight(family, style));
    }
    return w;
}

}  // namespace

QColor token_colour(const ThemeReport& t, const QString& name) {
    return QColor(t.tokens.value(name));
}

QColor theme_colour(const QString& name) {
    static const QMap<QString, QString> tokens = [] {
        QMap<QString, QString> m;
        for (const QString& raw : read_resource(":/theme/tokens.txt").split('\n', Qt::SkipEmptyParts)) {
            const QString line = raw.trimmed();
            const qsizetype sp = line.indexOf(' ');
            if (sp > 0) m.insert(line.left(sp), line.mid(sp + 1).trimmed());
        }
        return m;
    }();
    return QColor(tokens.value(name));
}

ThemeReport apply_theme(QApplication& app) {
    ThemeReport r;

    // Trimmed: a CRLF file must not put a CR into a family or colour name.
    for (const QString& raw : read_resource(":/theme/tokens.txt").split('\n', Qt::SkipEmptyParts)) {
        const QString line = raw.trimmed();
        const qsizetype sp = line.indexOf(' ');
        if (sp > 0) r.tokens.insert(line.left(sp), line.mid(sp + 1).trimmed());
    }
    if (r.tokens.isEmpty()) r.problems << "tokens.txt is missing from the resources";
    r.ui_family = r.tokens.value("ui-family");
    r.mono_family = r.tokens.value("mono-family");

    // Geist and Plex, embedded (OFL, app/fonts), so the look does not depend
    // on what the machine has installed.
    const QStringList fonts = QDir(":/fonts").entryList({"*.ttf"}, QDir::Files, QDir::Name);
    for (const QString& f : fonts) {
        const QString path = ":/fonts/" + f;
        if (QFontDatabase::addApplicationFont(path) >= 0) r.fonts_loaded << path;
        else r.problems << "font did not register: " + path;
    }
    // Each weight must resolve under the family the style sheet names. The
    // files carry per-weight legacy names ("IBM Plex Sans Cond Medm") and the
    // typographic family ("IBM Plex Sans Condensed"); a font system that
    // groups by the legacy name alone would give Regular for every weight.
    const struct { QString family; QList<int> need; } want[] = {
        {r.ui_family, {400, 500, 600, 700}},
        {r.mono_family, {400, 500}},
    };
    for (const auto& [family, need] : want) {
        if (!QFontDatabase::families().contains(family)) {
            r.problems << "family not found: " + family;
            continue;
        }
        const QSet<int> have = weights_of(family);
        for (int w : need)
            if (!have.contains(w)) r.problems << QString("%1 has no weight %2").arg(family).arg(w);
    }

    // Fusion draws every control from the palette and the sheet the same way
    // on all three systems; the native styles would tint them.
    if (QStyle* s = QStyleFactory::create("Fusion")) {
        QApplication::setStyle(s);
        r.style = "Fusion";
    } else {
        r.problems << "the Fusion style is not in this Qt";
    }

    const auto c = [&](const char* name) { return token_colour(r, name); };
    QPalette p;
    p.setColor(QPalette::Window, c("bg"));
    p.setColor(QPalette::WindowText, c("ink"));
    p.setColor(QPalette::Base, c("panel"));
    p.setColor(QPalette::AlternateBase, c("panel-2"));
    p.setColor(QPalette::Text, c("ink"));
    p.setColor(QPalette::Button, c("raise"));
    p.setColor(QPalette::ButtonText, c("ink"));
    p.setColor(QPalette::BrightText, c("white"));
    p.setColor(QPalette::Highlight, c("accent"));
    p.setColor(QPalette::HighlightedText, c("ink"));
    p.setColor(QPalette::ToolTipBase, c("panel"));
    p.setColor(QPalette::ToolTipText, c("ink"));
    p.setColor(QPalette::PlaceholderText, c("ink-4"));
    p.setColor(QPalette::Link, c("accent"));
    p.setColor(QPalette::Light, c("raise-2"));
    p.setColor(QPalette::Midlight, c("line-2"));
    p.setColor(QPalette::Mid, c("raise"));
    p.setColor(QPalette::Dark, c("panel"));
    p.setColor(QPalette::Shadow, c("bg"));
    for (auto role : {QPalette::WindowText, QPalette::Text, QPalette::ButtonText})
        p.setColor(QPalette::Disabled, role, c("ink-4"));
    QApplication::setPalette(p);

    QFont font(r.ui_family);
    font.setPixelSize(13);
    font.setWeight(QFont::Normal);
    QApplication::setFont(font);

    const QString qss = read_resource(":/theme/studio.qss");
    if (qss.isEmpty()) r.problems << "studio.qss is missing from the resources";
    app.setStyleSheet(qss);
    return r;
}

QByteArray theme_report_json(const ThemeReport& t) {
    QJsonObject o;
    o["ok"] = t.ok();
    o["ui_family"] = t.ui_family;
    o["mono_family"] = t.mono_family;
    o["fonts_loaded"] = QJsonArray::fromStringList(t.fonts_loaded);
    o["problems"] = QJsonArray::fromStringList(t.problems);
    QJsonObject weights;
    for (const QString& f : {t.ui_family, t.mono_family}) {
        QJsonArray a;
        QList<int> w = weights_of(f).values();
        std::sort(w.begin(), w.end());
        for (int x : w) a.append(x);
        weights[f] = a;
    }
    o["weights"] = weights;
    o["style"] = t.style;
    return QJsonDocument(o).toJson(QJsonDocument::Indented);
}

}  // namespace rudra::app
