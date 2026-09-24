// The app's window against the Studio page, offscreen (Phase 3).
//
// Step 2: the menubar the window builds is engine/actions' menus exactly
// (titles, entries, separators, submenus), and for the page's menus the
// Studio's labels, key hints and check states (tests/golden/actions). Every
// key the page maps reaches its action and no two actions share a key.
// Running the actions changes what the page's handlers change. Step 1: the
// look resolves (fonts, weights, the sheet) from the static library.

#include <gtest/gtest.h>

#include <QAction>
#include <QActionGroup>
#include <QApplication>
#include <QMenu>
#include <QMenuBar>
#include <QAbstractButton>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QStackedWidget>
#include <QImage>
#include <QMouseEvent>
#include <QComboBox>
#include <QPlainTextEdit>
#include <QSpinBox>
#include <QSettings>
#include <QClipboard>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QPointer>
#include <QUrl>
#include <QTest>
#include <QTreeWidget>

#include <cstdio>
#include <functional>
#include <QTextDocumentFragment>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <cctype>
#include <cmath>

#include <fstream>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "main_window.hpp"
#include "model_dialogs.hpp"
#include "workflow_check.hpp"
#include "rudra/deliver/master.hpp"
#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "widgets.hpp"
#include "rudra/core/copy_texts.hpp"
#include "rudra/engine/actions.hpp"
#include "theme.hpp"

using nlohmann::json;
using namespace rudra;

namespace {

app::ThemeReport* g_theme = nullptr;

const json& golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/actions/actions.json");
        return json::parse(f);
    }();
    return g;
}

QString qs(std::string_view s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }

std::string id_of(const QAction* a) {
    const QString n = a->objectName();
    return n.startsWith("act:") ? n.mid(4).toStdString() : std::string();
}

// A menu as the spec writes it: ids, "-", ">Title" and "<".
void flatten(const QMenu* m, std::vector<std::string>& out) {
    for (const QAction* a : m->actions()) {
        if (a->isSeparator()) out.push_back("-");
        else if (a->menu()) {
            out.push_back(">" + a->menu()->title().toStdString());
            flatten(a->menu(), out);
            out.push_back("<");
        } else out.push_back(id_of(a));
    }
}

// The page's hint as Qt prints the shortcut.
QString hint_text(const std::string& hint) {
    if (hint == "Space") return "Space";
    return QString::fromStdString(hint);
}

}  // namespace

TEST(AppLook, ResolvesFromTheLibrary) {
    ASSERT_NE(g_theme, nullptr);
    EXPECT_TRUE(g_theme->ok()) << g_theme->problems.join("; ").toStdString();
    EXPECT_EQ(g_theme->fonts_loaded.size(), 6);
    EXPECT_FALSE(qApp->styleSheet().isEmpty());
}

TEST(AppActions, MenubarIsTheSpec) {
    app::MainWindow w(false);
    const auto menus = w.menuBar()->actions();
    ASSERT_EQ(std::size_t(menus.size()), menu_specs().size());
    for (std::size_t i = 0; i < menu_specs().size(); ++i) {
        const auto& spec = menu_specs()[i];
        const QMenu* m = menus[qsizetype(i)]->menu();
        ASSERT_NE(m, nullptr);
        EXPECT_EQ(m->title(), qs(spec.title));
        std::vector<std::string> got, want;
        flatten(m, got);
        for (auto e : spec.entries) want.emplace_back(e);
        EXPECT_EQ(got, want) << m->title().toStdString();
    }
}

TEST(AppActions, PageMenusLabelsHintsChecks) {
    app::MainWindow w(false);
    for (const auto& gm : golden()["menus"]) {
        for (const auto& item : gm["items"]) {
            if (item.contains("separator")) continue;
            const std::string id = item["act"];
            QAction* a = w.action(id);
            ASSERT_NE(a, nullptr) << id;
            EXPECT_EQ(a->text().toStdString(), std::string(item["label"])) << id;
            if (!item["hint"].is_null()) {
                EXPECT_EQ(a->shortcut().toString(QKeySequence::PortableText), hint_text(item["hint"])) << id;
            }
            EXPECT_EQ(a->isCheckable(), !item["check"].is_null()) << id;
        }
    }
    // One choice of several: the modes, the containers, the zooms.
    for (const char* g : {"mode-all", "container-aces", "zoom-fit"}) {
        ASSERT_NE(w.action(g)->actionGroup(), nullptr) << g;
        EXPECT_TRUE(w.action(g)->actionGroup()->isExclusive()) << g;
    }
    EXPECT_EQ(w.action("preserve")->actionGroup(), nullptr);
}

TEST(AppActions, EveryPageKeyReachesItsAction) {
    app::MainWindow w(false);
    std::map<QString, std::string> owner;   // one owner per key sequence
    for (const auto& spec : action_specs()) {
        for (const QKeySequence& k : w.action(spec.id)->shortcuts()) {
            const QString t = k.toString(QKeySequence::PortableText);
            const auto [it, fresh] = owner.emplace(t, std::string(spec.id));
            EXPECT_TRUE(fresh) << t.toStdString() << " is on " << it->second << " and " << spec.id;
        }
    }
    for (const auto& [key, act] : golden()["keys"].items()) {
        QString t;
        if (key == " ") t = "Space";
        else if (key.size() == 1) t = QString::fromStdString(key).toUpper();
        else t = QString::fromStdString(key);
        ASSERT_TRUE(owner.count(t)) << "key '" << key << "' reaches nothing";
        EXPECT_EQ(owner[t], std::string(act)) << "key '" << key << "'";
    }
}

TEST(AppActions, KeyPressRunsTheAction) {
    app::MainWindow w(false);
    w.show();
    w.activateWindow();
    ASSERT_TRUE(QTest::qWaitForWindowActive(&w));
    QTest::keyClick(&w, Qt::Key_2);
    EXPECT_EQ(w.composite().mode, RecoveryMode::Highlights);
    EXPECT_TRUE(w.action("mode-highlights")->isChecked());
    EXPECT_FALSE(w.action("mode-all")->isChecked());
    QTest::keyClick(&w, Qt::Key_BracketRight);
    EXPECT_FLOAT_EQ(w.composite().strength, 1.1f);
    QTest::keyClick(&w, Qt::Key_P);
    EXPECT_FALSE(w.composite().preserve_outside);
    EXPECT_FALSE(w.action("preserve")->isChecked());
}

TEST(AppActions, HandlersDoWhatThePagesDo) {
    app::MainWindow w(false);
    // nudgeStrength: 0.1 a press, rounded to hundredths, clamped to 0..2.
    for (int i = 0; i < 25; ++i) w.run("strength-up");
    EXPECT_FLOAT_EQ(w.composite().strength, 2.0f);
    for (int i = 0; i < 7; ++i) w.run("strength-down");
    EXPECT_FLOAT_EQ(w.composite().strength, 1.3f);
    for (int i = 0; i < 30; ++i) w.run("strength-down");
    EXPECT_FLOAT_EQ(w.composite().strength, 0.0f);
    w.run("mode-off");
    w.run("preserve");
    EXPECT_EQ(w.composite().mode, RecoveryMode::Off);
    EXPECT_FALSE(w.composite().preserve_outside);
    // reset-recon: mode all, strength 1, preserve on.
    w.run("reset-recon");
    EXPECT_EQ(w.composite().mode, RecoveryMode::All);
    EXPECT_FLOAT_EQ(w.composite().strength, 1.0f);
    EXPECT_TRUE(w.composite().preserve_outside);
    EXPECT_TRUE(w.action("mode-all")->isChecked());
    EXPECT_TRUE(w.action("preserve")->isChecked());
    // setContainer.
    EXPECT_EQ(w.container(), "aces");
    w.run("container-linear");
    EXPECT_EQ(w.container(), "linear");
    EXPECT_TRUE(w.action("container-linear")->isChecked());
    EXPECT_FALSE(w.action("container-aces")->isChecked());
}

TEST(AppActions, EnabledAsThePageEnablesThem) {
    app::MainWindow w(false);
    w.refresh_enabled();
    // No frames: refreshMenu turns these off.
    for (const char* id : {"close", "first", "prev", "next", "last", "play", "master", "undo", "redo",
                           "copy-metrics", "copy-scopes", "copy-delivery", "remeasure"})
        EXPECT_FALSE(w.action(id)->isEnabled()) << id;
    for (const char* id : {"open", "mode-all", "preserve", "strength-up", "reset-recon", "container-aces",
                           "shortcuts", "about", "open-package", "quit"})
        EXPECT_TRUE(w.action(id)->isEnabled()) << id;
    // What waits on a later step says so.
    EXPECT_TRUE(w.action("rail-left")->isEnabled());
    // Step 11 made the last of the page's actions live: none waits on a
    // later step; the only ones off are the viewer's, in a window without one.
    for (const auto& spec : action_specs()) {
        const QString why = w.pending_reason(spec.id);
        EXPECT_TRUE(why.isEmpty() || why == "This build has no viewer.") << spec.id << ": " << why.toStdString();
    }
    for (const char* id : {"master", "copy-metrics", "copy-scopes", "copy-delivery", "models", "recent-clear"})
        EXPECT_TRUE(w.pending_reason(id).isEmpty()) << id;
    // A disabled action does not run from its key either.
    w.run("undo");
    EXPECT_EQ(w.session().undo_depth(), 0u);
}

TEST(AppSession, UndoAndRedoThroughTheMenus) {
    app::MainWindow w(false);
    w.run("strength-up");
    w.run("mode-shadows");
    w.refresh_enabled();
    EXPECT_TRUE(w.action("undo")->isEnabled());
    EXPECT_FALSE(w.action("redo")->isEnabled());
    w.action("undo")->trigger();
    EXPECT_EQ(w.composite().mode, RecoveryMode::All);
    EXPECT_TRUE(w.action("mode-all")->isChecked());
    EXPECT_TRUE(w.action("redo")->isEnabled());   // refreshed after the change
    w.action("undo")->trigger();
    EXPECT_FLOAT_EQ(w.composite().strength, 1.0f);
    EXPECT_FALSE(w.action("undo")->isEnabled());
    w.action("redo")->trigger();
    w.action("redo")->trigger();
    EXPECT_EQ(w.session().grade.mode, "shadows");
    EXPECT_FLOAT_EQ(w.composite().strength, 1.1f);
    // Region EV reset is live now, and undoable.
    w.session().region_press(0, 0.0);
    w.session().region_move(50.0, false);
    w.session().region_release();
    EXPECT_DOUBLE_EQ(w.session().grade.regions[0].ev, 0.5);
    w.run("reset-regions");
    EXPECT_DOUBLE_EQ(w.session().grade.regions[0].ev, 0.0);
    w.run("undo");
    EXPECT_DOUBLE_EQ(w.session().grade.regions[0].ev, 0.5);
    // The wipe key is the session's.
    w.run("wipe");
    ASSERT_TRUE(w.session().wipe.has_value());
    EXPECT_TRUE(w.action("wipe")->isChecked());
}

// ---- step 4: the layout of the page -------------------------------------

namespace {

const json& layout() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/layout/layout.json");
        return json::parse(f);
    }();
    return g;
}

// textContent with whitespace collapsed, as the golden records it.
QString norm(QString t) {
    if (t.contains('<')) t = QTextDocumentFragment::fromHtml(t).toPlainText();
    for (QChar& c : t)
        if (c.isSpace() || c == QChar(0x200A)) c = ' ';
    return t.simplified();
}

// A widget of one of the app's own classes by name (they have no Q_OBJECT).
template <class T>
T* find(const QWidget& w, const QString& name) {
    return dynamic_cast<T*>(w.findChild<QWidget*>(name));
}

// The page's ids the native window has no widget for, and why.
const std::map<std::string, std::string> kNotWidgets = {
    {"menubar", "the Qt menubar (its box is checked with the frame)"},
    {"gl", "the viewer draws with QRhi in its own window (ADR-010)"},
    {"plate", "the viewer's own window"},
    {"plateLabel", "drawn over the picture by the viewer: step 7"},
    {"plateHint", "drawn over the picture by the viewer: step 7"},
    {"peakBadge", "drawn over the picture by the viewer: step 7"},
    {"fcLegend", "drawn over the picture by the viewer: step 7"},
    {"probeBox", "the floating probe: step 7"},
    {"scrubHead", "painted by the scrub bar"},
};
// Texts that are the machine's, not the page's (the harness has no model).
const std::set<std::string> kLiveText = {"ckpt", "log", "menubar", "device"};

struct Window {
    app::MainWindow w{false};
    explicit Window(const std::string& state) {
        w.resize(1600, 1000);
        w.show();
        (void)QTest::qWaitForWindowExposed(&w);
        if (state == "simple") w.findChild<QPushButton*>("wsSimple")->click();
        if (state == "rails_hidden")
            for (const char* a : {"rail-left", "rail-right", "scopes"}) w.run(a);
        if (state == "tab_grade") w.findChild<QPushButton*>("tabGrade")->click();
        if (state == "tab_deliver") w.findChild<QPushButton*>("tabDeliver")->click();
        QApplication::processEvents();
    }
    QWidget* get(const std::string& id) const { return w.findChild<QWidget*>(QString::fromStdString(id)); }
    bool shown(const QWidget* x) const { return x && x->isVisibleTo(&w) && x->width() > 0 && x->height() > 0; }
    std::array<int, 4> box(const QWidget* x) const {
        const QPoint p = x->mapTo(&w, QPoint(0, 0));
        return {p.x(), p.y(), x->width(), x->height()};
    }
};

}  // namespace

TEST(AppLayout, EveryPageElementHasItsWidget) {
    Window win("full");
    for (const auto& [id, e] : layout()["states"]["full"]["ids"].items()) {
        if (kNotWidgets.count(id)) continue;
        EXPECT_NE(win.get(id), nullptr) << "no widget named " << id;
    }
}

TEST(AppLayout, ShownAndHiddenAsThePageIsInEveryState) {
    for (const auto& [state, snap] : layout()["states"].items()) {
        Window win(state);
        for (const auto& [id, e] : snap["ids"].items()) {
            if (kNotWidgets.count(id)) continue;
            QWidget* x = win.get(id);
            if (!x) continue;
            EXPECT_EQ(win.shown(x), e["shown"].get<bool>()) << state << ": " << id;
        }
        // Panel labels, toolbar labels and notes, in reading order.
        std::vector<std::pair<std::string, bool>> want, got;
        for (const auto& l : snap["panel_labels"]) want.emplace_back(l["text"], l["shown"]);
        for (QWidget* p : win.w.findChildren<QWidget*>()) {
            if (p->property("role").toString() != "plabel") continue;
            const auto* t = p->findChild<QLabel*>();
            got.emplace_back(norm(t->text()).toStdString(), win.shown(p));
        }
        EXPECT_EQ(got, want) << state << ": panel labels";
        std::vector<std::pair<std::string, bool>> nwant, ngot;
        for (const auto& l : snap["notes"]) nwant.emplace_back(l["text"], l["shown"]);
        // In the page's order: the Reconstruct, Grade and Deliver panels.
        auto* stack = win.w.findChild<QStackedWidget*>("ipanels");
        for (int i = 0; i < stack->count(); ++i)
            for (QLabel* l : stack->widget(i)->findChildren<QLabel*>())
                if (l->property("role").toString() == "note-p") {
                    ngot.emplace_back(norm(l->text()).toStdString(), win.shown(l));
                }
        EXPECT_EQ(ngot, nwant) << state << ": notes";
    }
}

TEST(AppLayout, TheWordsAreThePages) {
    Window win("full");
    const auto& snap = layout()["states"]["full"];
    for (const auto& [id, e] : snap["ids"].items()) {
        QWidget* x = win.get(id);
        if (!x || kNotWidgets.count(id) || kLiveText.count(id)) continue;
        if (!e["buttons"].is_null()) {
            std::vector<std::string> got;
            for (auto* b : x->findChildren<QPushButton*>(QString(), Qt::FindDirectChildrenOnly)) got.push_back(norm(b->text()).toStdString());
            EXPECT_EQ(got, e["buttons"].get<std::vector<std::string>>()) << id;
            continue;
        }
        if (e["text"].is_null()) continue;
        QString text;
        if (auto* l = qobject_cast<QLabel*>(x)) text = l->text();
        else if (auto* b = qobject_cast<QAbstractButton*>(x)) text = b->text();
        else continue;   // a container: its children are checked on their own
        EXPECT_EQ(norm(text).toStdString(), e["text"].get<std::string>()) << id;
    }
    for (const auto& [id, title] : snap["titles"].items()) {
        QWidget* x = win.get(id);
        if (!x) continue;
        EXPECT_EQ(x->toolTip().toStdString(), title.get<std::string>()) << id;
    }
    for (const auto& [id, ph] : snap["placeholders"].items())
        EXPECT_EQ(win.w.findChild<QLineEdit*>(QString::fromStdString(id))->placeholderText().toStdString(),
                  ph.get<std::string>()) << id;
    std::vector<std::string> tl;
    for (QLabel* l : win.w.findChildren<QLabel*>())
        if (l->property("role").toString() == "vlabel") tl.push_back(l->text().toStdString());
    std::vector<std::string> twant;
    for (const auto& l : snap["toolbar_labels"]) twant.push_back(l["text"]);
    EXPECT_EQ(tl, twant);
    for (const auto& c : snap["checks"]) {
        auto* row = find<app::CheckRow>(win.w, QString::fromStdString(c["id"].get<std::string>()));
        ASSERT_NE(row, nullptr) << c["id"];
        EXPECT_EQ(row->label()->text().toStdString(), c["label"].get<std::string>());
        EXPECT_EQ(row->hint()->text().toStdString(), c["hint"].get<std::string>());
        EXPECT_EQ(row->on(), c["on"].get<bool>()) << c["id"];
    }
    for (const auto& ic : snap["icon_titles"]) {
        auto* b = find<app::IconButton>(win.w, QString::fromStdString(ic["id"].get<std::string>()));
        ASSERT_NE(b, nullptr);
        EXPECT_EQ(b->toolTip().toStdString(), ic["title"].get<std::string>());
        EXPECT_EQ(b->on(), ic["on"].get<bool>()) << ic["id"];
    }
    // The sliders: label, value, unit and the page's range in its steps.
    const auto& sl = snap["sliders"];
    auto* strength = win.w.findChild<QSlider*>("strength");
    auto* peak = win.w.findChild<QSlider*>("peak");
    EXPECT_EQ(sl[0]["value"], win.w.findChild<QLabel*>("strengthVal")->text().toStdString());
    EXPECT_EQ(sl[1]["value"], win.w.findChild<QLabel*>("peakVal")->text().toStdString());
    EXPECT_DOUBLE_EQ(strength->minimum() / 20.0, std::stod(sl[0]["min"].get<std::string>()));
    EXPECT_DOUBLE_EQ(strength->maximum() / 20.0, std::stod(sl[0]["max"].get<std::string>()));
    EXPECT_DOUBLE_EQ(1 / 20.0, std::stod(sl[0]["step"].get<std::string>()));
    EXPECT_DOUBLE_EQ(peak->minimum() / 2.0, std::stod(sl[1]["min"].get<std::string>()));
    EXPECT_DOUBLE_EQ(peak->maximum() / 2.0, std::stod(sl[1]["max"].get<std::string>()));
    // Fields.
    std::vector<std::vector<std::string>> fwant, fgot;
    for (const auto& f : snap["fields"]) fwant.push_back(f.get<std::vector<std::string>>());
    for (QWidget* f : win.w.findChildren<QWidget*>()) {
        if (f->property("role").toString() != "field") continue;
        const auto labels = f->findChildren<QLabel*>();
        fgot.push_back({labels[0]->text().toStdString(), labels[1]->text().toStdString()});
    }
    EXPECT_EQ(fgot, fwant);
}

TEST(AppLayout, TheFrameIsWhereThePagePutsIt) {
    const std::map<std::string, std::string> names = {
        {"iconrail", "iconRail"}, {"rail_left", "railLeft"}, {"centre", "centre"}, {"vtools", "viewerTools"},
        {"viewer", "viewer"},     {"transport", "transport"}, {"rail_right", "railRight"}, {"pipe", "pipe"}};
    for (const auto& [state, snap] : layout()["states"].items()) {
        Window win(state);
        const auto mb = win.box(win.w.menuBar());
        const auto& gm = snap["frame"]["menubar"]["box"];
        EXPECT_EQ(mb[3], gm[3].get<int>()) << state << ": menubar height";
        for (const auto& [key, name] : names) {
            const auto& g = snap["frame"][key];
            QWidget* x = win.get(name);
            ASSERT_NE(x, nullptr) << name;
            EXPECT_EQ(win.shown(x), g["shown"].get<bool>()) << state << ": " << key;
            if (!g["shown"].get<bool>()) continue;
            const auto b = win.box(x);
            for (int i = 0; i < 4; ++i)
                EXPECT_LE(std::abs(b[std::size_t(i)] - g["box"][std::size_t(i)].get<int>()), 1)
                    << state << ": " << key << " box[" << i << "] " << b[std::size_t(i)] << " vs "
                    << g["box"][std::size_t(i)].get<int>();
        }
    }
}

TEST(AppLayout, ShellControlsDoWhatShellJsDoes) {
    Window win("full");
    auto& w = win.w;
    // Workspace: Simple hides the scopes and the notes, and the Scopes icon goes off.
    w.findChild<QPushButton*>("wsSimple")->click();
    EXPECT_EQ(w.workspace(), "simple");
    EXPECT_FALSE(win.shown(win.get("scopes")));
    EXPECT_FALSE(find<app::IconButton>(w, "iScopes")->on());
    w.findChild<QPushButton*>("wsFull")->click();
    EXPECT_TRUE(win.shown(win.get("scopes")));
    // The icon rail drives the Window menu's actions, so their checks follow.
    find<app::IconButton>(w, "iMedia")->click();
    EXPECT_FALSE(win.shown(win.get("railLeft")));
    EXPECT_FALSE(w.action("rail-left")->isChecked());
    find<app::IconButton>(w, "iMedia")->click();
    EXPECT_TRUE(win.shown(win.get("railLeft")));
    // A menu item for a control on a hidden tab brings the tab forward.
    w.show_tab("grade");
    w.menuBar()->actions()[4 - 1]->menu()->triggered(w.action("mode-shadows"));   // Reconstruct
    EXPECT_EQ(w.tab(), "rec");
    // The controls move the session, and the session moves them.
    find<app::Seg>(w, "mode")->button("highlights")->click();
    EXPECT_EQ(w.session().grade.mode, "highlights");
    w.run("strength-up");
    EXPECT_EQ(w.findChild<QLabel*>("strengthVal")->text(), "1.10");
    EXPECT_EQ(w.findChild<QSlider*>("strength")->value(), 22);
    w.findChild<QSlider*>("peak")->setValue(5);   // 2.5 EV
    EXPECT_EQ(w.findChild<QLabel*>("peakVal")->text(), "1,148");
    find<app::CheckRow>(w, "preserve")->clicked();
    EXPECT_EQ(w.findChild<QLabel*>("preserveHint")->text(), "raw prediction");
    w.run("container-linear");
    EXPECT_EQ(w.findChild<QLabel*>("containerField")->text().toStdString(), "OpenEXR \u2014 linear Rec.2020");
    EXPECT_EQ(w.findChild<QLabel*>("primariesField")->text(), "Rec.2020");
    w.findChild<QPushButton*>("wipeBtn")->click();
    EXPECT_TRUE(w.session().wipe.has_value());
    find<app::Seg>(w, "viewMode")->button("baseline")->click();
    EXPECT_FALSE(w.session().wipe.has_value());
    EXPECT_EQ(w.session().show, "baseline");
    find<app::Seg>(w, "viewLayer")->button("1")->click();
    EXPECT_EQ(w.session().view_layer, 1);
    EXPECT_TRUE(w.action("view-false-colour")->isChecked());
    // Undo takes the controls back too.
    w.run("undo");
    w.run("undo");
    w.run("undo");
    EXPECT_EQ(w.findChild<QLabel*>("strengthVal")->text(), "1.00");
    EXPECT_EQ(w.findChild<QLabel*>("preserveHint")->text(), "do-no-harm");
}

// ---- step 6: the panels driven as the page's are, against its golden -------

namespace {

const json& session_golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/session/scripts.json");
        return json::parse(f);
    }();
    return g;
}

void send_mouse(QWidget* target, QEvent::Type type, double x, Qt::KeyboardModifiers mods = Qt::NoModifier) {
    const QPointF local(x, 5.0);
    const QPointF global = target->mapToGlobal(local);
    const Qt::MouseButtons held = type == QEvent::MouseButtonRelease ? Qt::NoButton : Qt::LeftButton;
    QMouseEvent e(type, local, local, global, Qt::LeftButton, held, mods);
    QApplication::sendEvent(target, &e);
}

// One gesture of the golden, made on the window's own widgets.
void gesture(app::MainWindow& w, const json& op) {
    const std::string k = op[0];
    if (k == "start") return;
    if (k == "mode") {
        find<app::Seg>(w, "mode")->button(QString::fromStdString(op[1].get<std::string>()))->click();
    } else if (k == "preserve") {
        QTest::mouseClick(w.findChild<QWidget*>("preserve"), Qt::LeftButton);
    } else if (k == "strength" || k == "peak") {
        auto* s = w.findChild<QSlider*>(QString::fromStdString(k));
        const double v = std::stod(op[1].get<std::string>());
        const int n = int(std::lround(v * (k == "strength" ? 20.0 : 2.0)));
        if (k == "strength") s->setSliderDown(true);   // the pointerdown
        s->setValue(n);
        if (k == "strength") s->setSliderDown(false);
    } else if (k == "region") {
        auto* ev = find<app::RegionEditor>(w, "regions")->value(op[1].get<int>());
        double x = 5.0;
        const auto mods = op[3].get<bool>() ? Qt::ShiftModifier : Qt::NoModifier;
        send_mouse(ev, QEvent::MouseButtonPress, x);
        for (const auto& dx : op[2]) {
            x += dx.get<double>();
            send_mouse(ev, QEvent::MouseMove, x, mods);
        }
        send_mouse(ev, QEvent::MouseButtonRelease, x);
    } else if (k == "region0") {
        auto* ev = find<app::RegionEditor>(w, "regions")->value(op[1].get<int>());
        const double x = 5.0;
        // A double click: press, release, then the second press as a double click.
        send_mouse(ev, QEvent::MouseButtonPress, x);
        send_mouse(ev, QEvent::MouseButtonRelease, x);
        send_mouse(ev, QEvent::MouseButtonDblClick, x);
        send_mouse(ev, QEvent::MouseButtonRelease, x);
    } else if (k == "act") {
        w.action(op[1].get<std::string>())->trigger();
    } else if (k == "key") {
        const std::string key = op[1];
        Qt::KeyboardModifiers mods = op[2].get<bool>() ? Qt::ShiftModifier : Qt::NoModifier;
        int code = 0;
        if (key == "ArrowLeft") code = Qt::Key_Left;
        else if (key == "ArrowRight") code = Qt::Key_Right;
        else if (key == "Escape") code = Qt::Key_Escape;
        else if (key == " ") code = Qt::Key_Space;
        else if (key == "[") code = Qt::Key_BracketLeft;
        else if (key == "]") code = Qt::Key_BracketRight;
        else if (key.size() == 1 && std::isdigit(static_cast<unsigned char>(key[0]))) code = Qt::Key_0 + (key[0] - '0');
        else if (key.size() == 1 && std::isalpha(static_cast<unsigned char>(key[0]))) {
            code = Qt::Key_A + (std::toupper(static_cast<unsigned char>(key[0])) - 'A');
            if (std::isupper(static_cast<unsigned char>(key[0]))) mods |= Qt::ShiftModifier;   // "P" is Shift+P
        }
        ASSERT_NE(code, 0) << key;
        QTest::keyClick(&w, Qt::Key(code), mods);
    } else {
        FAIL() << "unknown gesture " << k;
    }
    QApplication::processEvents();
}

}  // namespace

TEST(AppPanels, EveryGestureOnTheWidgetsGivesThePagesState) {
    for (const auto& [name, steps] : session_golden()["scripts"].items()) {
        app::MainWindow w(false);
        w.resize(1600, 1000);
        w.show();
        w.activateWindow();
        ASSERT_TRUE(QTest::qWaitForWindowActive(&w));
        w.show_tab("grade");   // the Region EV rows must be laid out to be pressed
        QApplication::processEvents();
        int i = 0;
        for (const auto& st : steps) {
            gesture(w, st["op"]);
            const std::string where = name + " step " + std::to_string(i++) + " " + st["op"].dump();
            const Session& s = w.session();
            ASSERT_EQ(s.params_json(), st["params"].get<std::string>()) << where;
            EXPECT_EQ(s.undo_depth(), st["undo"].get<std::size_t>()) << where;
            EXPECT_EQ(s.redo_depth(), st["redo"].get<std::size_t>()) << where;
            if (st["wipe"].is_null()) {
                EXPECT_FALSE(s.wipe.has_value()) << where;
            } else {
                ASSERT_TRUE(s.wipe.has_value()) << where;
                EXPECT_EQ(*s.wipe, st["wipe"].get<double>()) << where;
            }
            // What the panels show.
            const auto& p = st["panel"];
            auto* regions = find<app::RegionEditor>(w, "regions");
            ASSERT_EQ(std::size_t(regions->rows()), p["regions"].size()) << where;
            for (int r = 0; r < regions->rows(); ++r) {
                const auto& g = p["regions"][std::size_t(r)];
                EXPECT_EQ(regions->range(r)->text().toStdString(), g["q"].get<std::string>()) << where;
                EXPECT_EQ(regions->value(r)->text().toStdString(), g["ev"].get<std::string>()) << where;
                EXPECT_EQ(regions->value(r)->property("live").toBool(), g["live"].get<bool>()) << where << " row " << r;
                EXPECT_EQ(regions->row(r)->property("sel").toBool(), g["sel"].get<bool>()) << where << " row " << r;
            }
            EXPECT_EQ(w.findChild<QLabel*>("regionCount")->text().toStdString(), p["regionCount"].get<std::string>()) << where;
            EXPECT_EQ(w.findChild<QLabel*>("strengthVal")->text().toStdString(), p["strengthVal"].get<std::string>()) << where;
            EXPECT_EQ(w.findChild<QLabel*>("peakVal")->text().toStdString(), p["peakVal"].get<std::string>()) << where;
            EXPECT_EQ(w.findChild<QLabel*>("preserveHint")->text().toStdString(), p["preserveHint"].get<std::string>()) << where;
            EXPECT_EQ(find<app::CheckRow>(w, "preserve")->on(), p["preserveOn"].get<bool>()) << where;
            EXPECT_EQ(find<app::Seg>(w, "mode")->on().toStdString(), p["mode"].get<std::string>()) << where;
        }
    }
}

// ---- steps 7 and 8: the probe, the Frame panel and the bars, wired --------

namespace {

// A small frame with clipped highlights, the masks on them, and a residual
// that lifts them: enough for every read-out to have something to say.
void synthetic_frame(app::MainWindow& w) {
    const int W = 96, H = 64;
    PlanarBuffer sdr(3, H, W), residual(3, H, W), hi(1, H, W), sh(1, H, W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const std::size_t i = std::size_t(y) * W + std::size_t(x);
            const float v = std::min(1.0f, float(x) / float(W - 16));
            for (int c = 0; c < 3; ++c) {
                sdr.plane(c)[i] = std::clamp(v * (c == 0 ? 1.0f : c == 1 ? 0.9f : 0.75f), 0.0f, 1.0f);
                residual.plane(c)[i] = v >= 0.99f ? 0.35f : 0.0f;
            }
            hi.plane(0)[i] = v >= 0.99f ? 1.0f : 0.0f;
            sh.plane(0)[i] = x < 4 ? 1.0f : 0.0f;
        }
    FrameScalars scalars;
    scalars.shadow_weight = 0.87f;
    FrameHeader header;
    header.source_resolution = header.resolution = "96x64";
    header.elapsed_s = 0.25;
    w.present_frame(SdrImage(std::move(sdr)), Fields{std::move(residual), std::move(hi), std::move(sh)}, scalars,
                    ModelConstants{16.0f, 4.0f, -1.0f}, header);
}

std::vector<std::vector<std::string>> rows_of(QWidget* ms) {
    std::vector<std::vector<std::string>> out;
    for (QWidget* r : ms->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly)) {
        if (r->property("role").toString() != "ro") continue;
        std::vector<std::string> row;
        for (QLabel* l : r->findChildren<QLabel*>()) row.push_back(l->text().toStdString());
        row.push_back(r->findChildren<QLabel*>()[1]->property("state").toString().toStdString());
        out.push_back(row);
    }
    return out;
}

}  // namespace

TEST(AppMeasure, TheFramePanelAndBarsShowTheMeasurement) {
    app::MainWindow w(false);
    w.resize(1600, 1000);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    EXPECT_FALSE(w.action("remeasure")->isEnabled());
    synthetic_frame(w);
    w.measure_now();
    const FrameMeasure* m = w.measurement();
    ASSERT_NE(m, nullptr);
    const MetricsText want = metrics_text(m->frame_metrics(), [] {
        FrameHeader h;
        h.source_resolution = h.resolution = "96x64";
        h.elapsed_s = 0.25;
        return h;
    }(), m->coverage.clipped_pct);
    auto expect_rows = [](const std::vector<std::vector<std::string>>& got, const std::vector<MetricRow>& rows) {
        ASSERT_EQ(got.size(), rows.size());
        for (std::size_t i = 0; i < rows.size(); ++i) {
            EXPECT_EQ(got[i][0], rows[i].k);
            EXPECT_EQ(got[i][1], rows[i].v) << rows[i].k;
            EXPECT_EQ(got[i][2], rows[i].u);
            EXPECT_EQ(got[i][3], rows[i].warn ? "warn" : "") << rows[i].k;
        }
    };
    expect_rows(rows_of(w.findChild<QWidget*>("measA")), want.a);
    expect_rows(rows_of(w.findChild<QWidget*>("measB")), want.b);
    EXPECT_EQ(w.findChild<QLabel*>("statusMask")->text().toStdString(), want.status_mask);
    EXPECT_EQ(w.findChild<QLabel*>("statusTime")->text().toStdString(), want.status_time);
    EXPECT_EQ(w.findChild<QLabel*>("srcInfo")->text().toStdString(), want.src_info);
    EXPECT_GT(m->coverage.clipped_pct, 0.0);
    EXPECT_GT(m->measured.metrics.maxcll, 203.0);
    // The clip bar: red for the clipped share, amber after it for the mask.
    auto* bar = find<app::ClipBar>(w, "clipBar");
    ASSERT_TRUE(bar->isVisible());
    EXPECT_NEAR(bar->lost()->width(), bar->width() * std::min(100.0, m->coverage.clipped_pct) / 100.0, 1.0);
    EXPECT_EQ(bar->acted()->x(), bar->lost()->width());
    // The scopes are drawn from it.
    EXPECT_TRUE(find<app::ScopePlot>(w, "wave")->has_data());
    EXPECT_TRUE(find<app::ScopePlot>(w, "hist")->has_data());
    // The pipeline bar: MaxCLL over the view peak warns, a peak above it does not.
    EXPECT_TRUE(w.findChild<QLabel*>("pipeWarn")->isVisible());
    EXPECT_EQ(w.findChild<QLabel*>("pipeWarn")->text().toStdString(),
              pipe_text("aces", 203.0, m->measured.metrics.maxcll, std::nullopt).warn);
    w.session().peak_input(5.0);
    EXPECT_EQ(w.findChild<QLabel*>("viewTransform")->text().toStdString(), "exposure + clip \u00b7 6,496 nits");
    EXPECT_FALSE(w.findChild<QLabel*>("pipeWarn")->isVisible());
    w.run("container-linear");
    EXPECT_EQ(w.findChild<QLabel*>("pipeMaster")->text().toStdString(), "linear Rec.2020 EXR, half");
    // A grade change measures again, on its own thread, after the grade settles.
    const double before = m->measured.metrics.peak_nits;
    w.run("mode-off");
    // (By value: the new measurement may well sit where the old one was freed.)
    for (int i = 0; i < 500 && !(w.measurement()->measured.metrics.peak_nits < before - 1.0); ++i) QTest::qWait(10);
    ASSERT_LT(w.measurement()->measured.metrics.peak_nits, before - 1.0) << "no new measurement within 5 s";
    EXPECT_EQ(rows_of(w.findChild<QWidget*>("measA"))[2][1],
              metrics_text(w.measurement()->frame_metrics(), std::nullopt, 0).a[2].v);   // Peak, redrawn
}

TEST(AppMeasure, TheProbeReadsThePixelUnderThePointer) {
    app::MainWindow w(false);
    w.resize(1600, 1000);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    synthetic_frame(w);
    w.measure_now();
    const FrameMeasure* m = w.measurement();
    for (const auto& at : std::vector<std::pair<double, double>>{{90.4, 10.9}, {2.0, 60.0}, {40.5, 30.5}}) {
        const auto probe = m->probe_at(at.first, at.second);
        ASSERT_TRUE(probe.has_value());
        w.probe_pixel(at, w.mapToGlobal(QPoint(500, 400)));
        const ProbePanelText t = probe_panel(probe);
        EXPECT_EQ(w.findChild<QLabel*>("probeXY")->text().toStdString(), t.xy);
        EXPECT_EQ(w.findChild<QLabel*>("probeNits")->text().toStdString(), t.nits);
        EXPECT_EQ(w.findChild<QLabel*>("probeNits")->property("state").toString(), "");
        EXPECT_TRUE(w.findChild<QLabel*>("probeNitsUnit")->isVisibleTo(&w));
        EXPECT_EQ(w.findChild<QLabel*>("probeDelta")->text().toStdString(), t.delta);
        EXPECT_EQ(w.findChild<QLabel*>("probeSrc")->text().toStdString(), t.src);
        EXPECT_EQ(w.findChild<QLabel*>("probeSrc")->property("state").toString().toStdString(), t.src_class);
        EXPECT_EQ(w.findChild<QLabel*>("probeBase")->text().toStdString(), t.base);
        EXPECT_EQ(w.findChild<QLabel*>("probeModel")->text().toStdString(), t.model);
        EXPECT_EQ(w.findChild<QLabel*>("probeModel")->property("state").toString().toStdString(), t.model_class);
        EXPECT_EQ(w.findChild<QLabel*>("probeMask")->text().toStdString(), t.mask);
        // The floating box: the page's rows, beside the pointer.
        ASSERT_TRUE(w.probe_box()->isVisible());
        const auto rows = probe_box(*probe);
        const auto labels = w.probe_box()->findChildren<QLabel*>();
        ASSERT_EQ(labels.size(), qsizetype(rows.size() * 2));
        for (std::size_t r = 0; r < rows.size(); ++r) {
            EXPECT_EQ(labels[qsizetype(r * 2)]->text().toStdString(), rows[r].k);
            EXPECT_EQ(labels[qsizetype(r * 2 + 1)]->text().toStdString(), rows[r].v);
        }
    }
    // The clipped corner reads as clipped.
    w.probe_pixel(std::pair{95.0, 0.0}, w.mapToGlobal(QPoint(500, 400)));
    EXPECT_EQ(w.findChild<QLabel*>("probeSrc")->property("state").toString(), "clip");
    // Off the frame, or the probe put away: idle, no box.
    w.probe_pixel(std::nullopt);
    EXPECT_EQ(w.findChild<QLabel*>("probeNits")->property("state").toString(), "idle");
    EXPECT_EQ(w.findChild<QLabel*>("probeDelta")->text().toStdString(), "pick a pixel with Probe, or hold Alt");
    EXPECT_FALSE(w.probe_box()->isVisible());
}

// ---- step 9: the Deliver tab's master, as a background job ----------------

namespace {

MasterFrame small_frame() {
    MasterFrame f;
    f.sdr = SdrImage(PlanarBuffer(3, 12, 20, 0.6f));
    f.fields = Fields{PlanarBuffer(3, 12, 20, 0.1f), PlanarBuffer(1, 12, 20, 0.5f), PlanarBuffer(1, 12, 20, 0.0f)};
    return f;
}

std::filesystem::path fresh_dir(const std::string& name) {
    const auto root = std::filesystem::temp_directory_path() / ("rudra-app-master-" + name);
    std::filesystem::remove_all(root);
    return root;
}

QString status_of(app::MainWindow& w) { return w.findChild<QLabel*>("renderStatus")->text(); }

bool wait_for(const std::function<bool()>& done, int ms = 10000) {
    for (int i = 0; i < ms / 10 && !done(); ++i) QTest::qWait(10);
    return done();
}

}  // namespace

TEST(AppDeliver, MasterRendersThePlanAndSaysSo) {
    app::MainWindow w(false);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    auto prepare = [](std::size_t) -> Result<MasterFrame> { return small_frame(); };
    // No folder: the page's message, nothing rendered.
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w), "Choose a render folder first.");
    EXPECT_FALSE(w.mastering());
    const auto root = fresh_dir("ok");
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(root.string()));
    w.findChild<QLineEdit*>("renderName")->setText("shot_010");
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(1);   // All loaded frames, sequence
    w.findChild<QSpinBox*>("renderStart")->setValue(1);
    w.master(prepare, 3);
    ASSERT_TRUE(wait_for([&] { return !w.mastering() && status_of(w).startsWith("Rendered"); }));
    EXPECT_EQ(status_of(w).toStdString(), "Rendered 3 frame(s) to " + root.string());
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Master EXR");
    const QString log = w.findChild<QPlainTextEdit*>("log")->toPlainText();
    const auto first = root / "shot_010.000001.exr", last = root / "shot_010.000003.exr";
    EXPECT_TRUE(log.contains(QString::fromStdString("Render destination: " + first.string() + " \u2026 " + last.string())));
    for (int i = 1; i <= 3; ++i) {
        const auto exr = root / ("shot_010.00000" + std::to_string(i) + ".exr");
        EXPECT_TRUE(std::filesystem::exists(exr)) << exr;
        EXPECT_TRUE(log.contains(QString::fromStdString("Saved " + exr.string() + " \u00b7 20x12"))) << i;
    }
    // Again: refused, nothing replaced.
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w).toStdString(), "Stopped after 0 / 3: Refusing to overwrite existing render: " + first.string());
    // A bad name and a relative folder, as the plan refuses them.
    w.findChild<QLineEdit*>("renderName")->setText("my shot");
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w), "Stopped after 0 / 3: Render name must contain only letters, numbers, dots, underscores or hyphens");
    w.findChild<QLineEdit*>("renderName")->setText("shot");
    w.findChild<QLineEdit*>("renderDir")->setText("renders/out");
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(0);   // Current image
    w.master(prepare, 3);
    EXPECT_EQ(status_of(w), "Stopped after 0 / 1: Choose an absolute render folder on the Studio computer");
    std::filesystem::remove_all(root);
}

TEST(AppDeliver, MasterEXRStopsARenderAfterTheFrameInHand) {
    app::MainWindow w(false);
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    const auto root = fresh_dir("cancel");
    w.findChild<QLineEdit*>("renderDir")->setText(QString::fromStdString(root.string()));
    w.findChild<QComboBox*>("renderMode")->setCurrentIndex(1);
    std::atomic<int> started{0};
    w.master([&](std::size_t) -> Result<MasterFrame> {
        ++started;
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        return small_frame();
    }, 20);
    ASSERT_TRUE(w.mastering());
    ASSERT_TRUE(wait_for([&] { return started >= 1; }));
    w.findChild<QPushButton*>("btnMaster")->click();   // the button stops it
    ASSERT_TRUE(wait_for([&] { return !w.mastering() && status_of(w).startsWith("Stopped"); }));
    EXPECT_TRUE(status_of(w).endsWith(": cancelled")) << status_of(w).toStdString();
    EXPECT_LT(started.load(), 20);
    EXPECT_EQ(w.findChild<QPushButton*>("btnMaster")->text(), "Master EXR");
    std::filesystem::remove_all(root);
}

// ---- step 5: the scope widgets against the page's own rasters -------------

namespace {

const json& scope_golden() {
    static const json g = [] {
        std::ifstream f(std::string(RUDRA_GOLDEN_DIR) + "/scopes/drawings.json");
        return json::parse(f);
    }();
    return g;
}

// How far two rasters of the same drawing are apart: the mean difference in
// codes, and the share of pixels whose channels all lie within `tol`.
struct Agreement {
    double mean = 1e9, within = 0.0;
    int worst = 0;
};
Agreement agreement(const QImage& a, const QImage& b, int tol) {
    Agreement r;
    const QImage x = a.convertToFormat(QImage::Format_RGB32), y = b.convertToFormat(QImage::Format_RGB32);
    if (x.size() != y.size()) return r;
    double sum = 0.0;
    long good = 0, all = 0;
    for (int row = 0; row < x.height(); ++row)
        for (int col = 0; col < x.width(); ++col) {
            const QRgb p = x.pixel(col, row), q = y.pixel(col, row);
            const int dr = std::abs(qRed(p) - qRed(q)), dg = std::abs(qGreen(p) - qGreen(q)),
                      db = std::abs(qBlue(p) - qBlue(q));
            sum += (dr + dg + db) / 3.0;
            const int d = std::max({dr, dg, db});
            r.worst = std::max(r.worst, d);
            good += d <= tol;
            ++all;
        }
    r.mean = sum / double(all);
    r.within = double(good) / double(all);
    return r;
}

// The pixels a drawing's labels change: where `with` and `without` differ.
std::vector<bool> changed(const QImage& with, const QImage& without) {
    const QImage x = with.convertToFormat(QImage::Format_RGB32), y = without.convertToFormat(QImage::Format_RGB32);
    std::vector<bool> m(std::size_t(x.width()) * std::size_t(x.height()), false);
    for (int row = 0; row < x.height(); ++row)
        for (int col = 0; col < x.width(); ++col) {
            const QRgb p = x.pixel(col, row), q = y.pixel(col, row);
            m[std::size_t(row) * std::size_t(x.width()) + std::size_t(col)] =
                std::max({std::abs(qRed(p) - qRed(q)), std::abs(qGreen(p) - qGreen(q)),
                          std::abs(qBlue(p) - qBlue(q))}) > 16;
        }
    return m;
}

// The share of `a`'s pixels with a pixel of `b` within one pixel of them.
double covered(const std::vector<bool>& a, const std::vector<bool>& b, int w, int h) {
    long n = 0, hit = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (!a[std::size_t(y) * std::size_t(w) + std::size_t(x)]) continue;
            ++n;
            bool near = false;
            for (int dy = -1; dy <= 1 && !near; ++dy)
                for (int dx = -1; dx <= 1 && !near; ++dx) {
                    const int u = x + dx, v = y + dy;
                    near = u >= 0 && v >= 0 && u < w && v < h && b[std::size_t(v) * std::size_t(w) + std::size_t(u)];
                }
            hit += near;
        }
    return n ? double(hit) / double(n) : 1.0;
}

// Where each <text> of a drawing lands in a widget of `target`: the
// transform paint_svg uses (viewBox, xMidYMid meet), a monospaced width.
struct LabelBox {
    std::string text;
    QRectF box;
    bool middle = false;
};
std::vector<LabelBox> label_boxes(const SvgDrawing& d, const QRectF& target) {
    std::vector<LabelBox> out;
    double vx = 0, vy = 0, vw = 1, vh = 1;
    std::sscanf(d.view_box.c_str(), "%lf %lf %lf %lf", &vx, &vy, &vw, &vh);
    const double k = std::min(target.width() / vw, target.height() / vh);
    const double ox = target.x() + (target.width() - vw * k) / 2.0 - vx * k;
    const double oy = target.y() + (target.height() - vh * k) / 2.0 - vy * k;
    std::function<void(const SvgElement&)> walk = [&](const SvgElement& e) {
        for (const auto& c : e.children) walk(c);
        if (e.tag != "text") return;
        const double size = std::stod(*e.attr("font-size")) * k;
        const double wd = 0.6 * size * double(e.text.size());
        double x = ox + std::stod(*e.attr("x")) * k;
        const double y = oy + std::stod(*e.attr("y")) * k;
        const std::string* a = e.attr("text-anchor");
        const bool middle = a && *a == "middle";
        if (middle) x -= wd / 2.0;
        out.push_back({e.text, QRectF(x - 2, y - size - 2, wd + 4, size * 1.3 + 4), middle});
    };
    for (const auto& e : d.elements) walk(e);
    return out;
}

// A mask inside a box: its left and right extent, its vertical centre of
// mass, and how many pixels it has there.
struct Extent {
    double left = 0, right = 0, cy = 0;
    int n = 0;
};
Extent extent(const std::vector<bool>& m, int w, int h, const QRectF& box) {
    Extent e;
    double sy = 0;
    e.left = 1e9;
    e.right = -1e9;
    for (int y = std::max(0, int(box.top())); y < std::min(h, int(std::ceil(box.bottom()))); ++y)
        for (int x = std::max(0, int(box.left())); x < std::min(w, int(std::ceil(box.right()))); ++x)
            if (m[std::size_t(y) * std::size_t(w) + std::size_t(x)]) {
                e.left = std::min(e.left, double(x));
                e.right = std::max(e.right, double(x));
                sy += y;
                ++e.n;
            }
    if (e.n) e.cy = sy / e.n;
    return e;
}

QImage grab_at(QWidget& w, int width, int height) {
    w.setFixedSize(width, height);
    w.setAttribute(Qt::WA_DontShowOnScreen, true);
    w.show();
    QApplication::processEvents();
    return w.grab().toImage();
}

}  // namespace

TEST(AppScopes, WaveformAndHistogramLookAsThePageDrawsThem) {
    for (const auto& c : scope_golden()["cases"]) {
        const std::string name = c["name"];
        ScopeData s;
        for (const char* k : {"lo", "q1", "mid", "q3", "hi"})
            (k == std::string("lo") ? s.lo : k == std::string("q1") ? s.q1 : k == std::string("mid") ? s.mid
             : k == std::string("q3") ? s.q3 : s.hi) = c["scopes"][k].get<std::vector<double>>();
        s.histogram = c["scopes"]["histogram"].get<std::vector<double>>();
        std::optional<double> maxcll;
        if (!c["metrics"].is_null()) {
            maxcll = c["metrics"]["maxcll"].is_null() ? std::nan("") : c["metrics"]["maxcll"].get<double>();
        }
        for (const char* part : {"wave", "hist"}) {
            const auto& box = c["drawn"][part]["box"];
            app::ScopePlot plot(part, part == std::string("wave") ? app::ScopePlot::Kind::Waveform
                                                                  : app::ScopePlot::Kind::Histogram,
                                box[1].get<int>());
            plot.set_data(s, maxcll);
            plot.set_text_shown(false);
            const QImage got = grab_at(plot, box[0].get<int>(), box[1].get<int>());
            const std::string base = std::string(RUDRA_GOLDEN_DIR) + "/scopes/" + name + "_" + part;
            const QImage want(QString::fromStdString(base + "_notext.png"));
            ASSERT_FALSE(want.isNull()) << name << " " << part;
            // The geometry and colour, labels hidden: equal but for how two
            // rasterisers cover an edge (a missing or misplaced band, bar or
            // line moves the mean past a code and the share under 97 %).
            const Agreement a = agreement(got, want, 8);
            if (a.mean > 1.0 || a.within < 0.97) got.save(QString::fromStdString("/tmp/native_scope_" + name + "_" + part + ".png"));
            EXPECT_LE(a.mean, 1.0) << name << " " << part;
            EXPECT_GE(a.within, 0.97) << name << " " << part << ": " << a.within * 100 << " % within 8 codes";
            // The labels: drawn where the page draws them. Glyphs differ
            // between rasterisers, so this compares where, not how.
            plot.set_text_shown(true);
            const QImage got_text = plot.grab().toImage();
            const QImage want_text(QString::fromStdString(base + ".png"));
            const int w = got.width(), h = got.height();
            const auto ours = changed(got_text, got), page = changed(want_text, want);
            EXPECT_GE(covered(ours, page, w, h), 0.8) << name << " " << part << ": our labels where the page has none";
            EXPECT_GE(covered(page, ours, w, h), 0.8) << name << " " << part << ": the page's labels missing";
            // And each label in its place: around every <text> of the display
            // list, the page's glyphs and ours start (or, centred, centre)
            // within a pixel and sit on the same line. How long a string runs
            // is the platform's (Chromium on Linux rounds each advance to a
            // whole pixel at these sizes, on Windows and macOS it does not),
            // so the far end is not compared.
            for (const auto& l : label_boxes(plot.drawing(), QRectF(1, 1, w - 2, h - 2))) {
                const Extent pe = extent(page, w, h, l.box), oe = extent(ours, w, h, l.box);
                ASSERT_GT(pe.n, 0) << name << " " << part << ": the page drew no \"" << l.text << "\"";
                ASSERT_GT(oe.n, 0) << name << " " << part << ": no \"" << l.text << "\"";
                const double px = l.middle ? (pe.left + pe.right) / 2.0 : pe.left;
                const double ox = l.middle ? (oe.left + oe.right) / 2.0 : oe.left;
                EXPECT_LE(std::abs(px - ox), 1.0) << name << " " << part << " \"" << l.text << "\" x";
                EXPECT_LE(std::abs(pe.cy - oe.cy), 1.0) << name << " " << part << " \"" << l.text << "\" y";
            }        }
    }
}

TEST(AppScopes, VectorscopeLooksAsThePageDrawsIt) {
    const auto& v = scope_golden()["vector"];
    const int w = v["sample"]["width"], h = v["sample"]["height"];
    const auto rgba = v["sample"]["rgba"].get<std::vector<double>>();
    PlanarBuffer sample(3, h, w);
    for (int i = 0; i < w * h; ++i)
        for (int ch = 0; ch < 3; ++ch) sample.plane(ch)[i] = float(rgba[std::size_t(i) * 4 + std::size_t(ch)]);
    app::VectorscopeView view;
    view.set_image(vectorscope(sample));
    const QImage got = grab_at(view, v["plot_box"][0].get<int>(), v["plot_box"][1].get<int>());
    const QImage want(QString::fromStdString(std::string(RUDRA_GOLDEN_DIR) + "/scopes/vector.png"));
    ASSERT_FALSE(want.isNull());
    // The picture itself is Phase 2's, equal to the page's canvas; what is
    // compared here is its placing, scaling and frame (labels included).
    const Agreement a = agreement(got, want, 8);
    if (a.mean > 1.0 || a.within < 0.985) got.save("/tmp/native_scope_vector.png");
    EXPECT_LE(a.mean, 1.0);
    EXPECT_GE(a.within, 0.985) << a.within * 100 << " % within 8 codes (worst " << a.worst << ")";
}


// ---------------------------------------------------------------------------
// Phase 3 step 10: the checkpoint manager and the first run
// ---------------------------------------------------------------------------

namespace {

// A backend that answers at once with fields of its own level, so a frame
// says which model made it.
class FakeBackend : public InferenceBackend {
public:
    FakeBackend(float level, std::shared_ptr<std::atomic<int>> calls) : level_(level), calls_(std::move(calls)) {}
    BackendInfo info() const override { return {Runtime::OnnxRuntime, Device::Cpu, "fake", "CPUExecutionProvider"}; }
    Result<FrameScalars> frame_pass(const SdrImage&) override { return FrameScalars{}; }
    Result<Fields> tile_pass(const SdrImage& tile, const FrameScalars&) override {
        ++*calls_;
        const int h = tile.height(), w = tile.width();
        return Fields{PlanarBuffer(3, h, w, level_), PlanarBuffer(1, h, w, 0.5f), PlanarBuffer(1, h, w, 0.0f)};
    }

private:
    float level_;
    std::shared_ptr<std::atomic<int>> calls_;
};

void write_manifest(const std::filesystem::path& dir, const std::string& name) {
    std::filesystem::create_directories(dir);
    const json tol{{"atol", 1e-5}, {"rtol", 1e-5}};
    const std::string z(64, '0');
    const json m{{"contract", "1.0"}, {"name", name}, {"source", {{"file", name + ".pt"}, {"sha256", z}}},
                 {"exported", "2026-09-24T00:00:00Z"},
                 {"network", {{"base_channels", 32}, {"corpus_ev", -1.0}, {"log_scale", 16.0}, {"max_hdr", 4.0},
                              {"heads", {{"residual_gate", false}, {"shadow_gate", true}, {"curve", false}}},
                              {"curve_params", 1}}},
                 {"tiling", {{"tile_size", 512}, {"overlap", 64}}},
                 {"files", {{"torchscript", "model.ts"}, {"onnx_frame", "f.onnx"}, {"onnx_tile", "t.onnx"},
                            {"golden", "golden/golden.json"}, {"torchscript_sha256", z}, {"onnx_frame_sha256", z},
                            {"onnx_tile_sha256", name + z.substr(name.size())}}},
                 {"onnx_inputs", {{"frame", {"sdr"}}, {"tile", {"sdr"}}}},
                 {"tolerance", {{"torchscript", tol}, {"onnx", tol}}}};
    std::ofstream(dir / "manifest.json") << m.dump(1);
}

struct Models {
    std::filesystem::path root;
    std::map<std::string, std::shared_ptr<std::atomic<int>>> calls;   // tile passes, by package name
    std::map<std::string, int> tests;                                 // self-tests run, by package name
    std::set<std::string> drifting;                                   // these fail their self-test
};

// Three packages (alpha, beta, gamma) in a models.json root; the window opens
// them through fakes and runs a fake self-test.
std::shared_ptr<Models> fake_models(app::MainWindow& w, const std::string& tag) {
    auto m = std::make_shared<Models>();
    m->root = fresh_dir("models-" + tag);
    const std::map<std::string, float> levels{{"alpha", 0.1f}, {"beta", 0.2f}, {"gamma", 0.3f}};
    for (const auto& [name, level] : levels) {
        write_manifest(m->root / name, name);
        m->calls[name] = std::make_shared<std::atomic<int>>(0);
    }
    std::ofstream(m->root / "models.json") << json{{"default", "beta.pt"},
                                                   {"models", {{{"file", "alpha.pt"}, {"kind", "sdr2hdr"}, {"title", "Alpha"}},
                                                               {{"file", "beta.pt"}, {"kind", "sdr2hdr"}, {"title", "Beta"}},
                                                               {{"file", "gamma.pt"}, {"kind", "sdr2hdr"}, {"title", "Gamma"}}}}}
                                                  .dump();
    app::MainWindow::ModelHooks h;
    h.verify_files = false;
    h.open = [m, levels](const ModelManifest& man, std::optional<BackendChoice> c) -> Result<OpenedBackend> {
        return OpenedBackend{std::make_unique<FakeBackend>(levels.at(man.name), m->calls.at(man.name)),
                             c.value_or(BackendChoice{Runtime::OnnxRuntime, Device::Cpu})};
    };
    h.test = [m](const ModelManifest& man, InferenceBackend& b) -> Result<SelfTestReport> {
        SelfTestReport r;
        r.backend = b.info();
        r.frames = 16;
        r.stitched = true;
        r.outputs["residual"].max_abs = m->drifting.count(man.name) ? 0.5 : 1e-6;
        r.outputs["residual"].excess = m->drifting.count(man.name) ? 0.4 : -1.0;
        ++m->tests[man.name];
        return r;
    };
    w.set_model_hooks(h);
    w.set_model_roots({m->root});
    return m;
}

bool use(app::MainWindow& w, const std::filesystem::path& p, QString* why = nullptr,
         std::optional<BackendChoice> c = std::nullopt) {
    std::optional<bool> result;
    w.use_model(p, c, [&](bool ok, const QString& y) {
        result = ok;
        if (why) *why = y;
    });
    EXPECT_TRUE(wait_for([&] { return result.has_value(); }));
    return result.value_or(false);
}

QString text_of(app::MainWindow& w, const char* id) {
    auto* l = w.findChild<QLabel*>(id);
    return l ? l->text() : QString("<no %1>").arg(id);
}

}  // namespace

TEST(AppModels, TheCatalogPicksTheRegistrysDefaultAndThePillsSayIt) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "pick");
    ASSERT_EQ(w.catalog().entries.size(), 3u);
    ASSERT_TRUE(w.catalog().chosen);
    EXPECT_EQ(w.catalog().entries[*w.catalog().chosen].package.filename(), "beta");
    EXPECT_EQ(text_of(w, "ckpt"), "no model");
    ASSERT_TRUE(use(w, m->root / "beta"));
    EXPECT_EQ(text_of(w, "ckpt"), "beta");
    EXPECT_EQ(text_of(w, "device"), "CPU");   // the page's info.gpu for a CPU run
    EXPECT_EQ(w.findChild<QLabel*>("lamp")->property("state").toString(), "on");
    const QString log = w.findChild<QPlainTextEdit*>("log")->toPlainText();
    EXPECT_TRUE(log.contains("self-test passed, 16 golden frames + stitch")) << log.toStdString();
    EXPECT_TRUE(log.contains("device CPU"));
    EXPECT_TRUE(log.contains("drop frames — the network runs once each, then the grade is local"));
    EXPECT_EQ(QSettings().value("model/package").toString().toStdString(), (m->root / "beta").string());
}

#ifdef RUDRA_HAVE_STILL_DECODE
TEST(AppModels, SwitchingModelsMidSessionKeepsTheSession) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "switch");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.open_source(QString::fromStdString((std::filesystem::path(RUDRA_GOLDEN_DIR) / "decode").string()));
    ASSERT_GE(w.frame_count(), 10u);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    w.run("next");
    w.run("next");
    ASSERT_TRUE(wait_for([&] { return w.current_index() == 2 && w.frame_fields() &&
                                      w.frame_fields()->residual.at(0, 0, 0) == 0.1f && *m->calls["alpha"] >= 3; }));
    // A grade with undo behind it.
    w.run("strength-up");
    w.run("mode-shadows");
    w.run("preserve");
    const std::string params = w.session().params_json();
    const auto undo = w.session().undo_depth();
    ASSERT_GE(undo, 3u);

    ASSERT_TRUE(use(w, m->root / "gamma"));
    EXPECT_EQ(text_of(w, "ckpt"), "gamma");
    // The same frames, the same frame, the same grade and undo; the new
    // model's fields for that frame.
    EXPECT_GE(w.frame_count(), 10u);
    EXPECT_EQ(w.current_index(), 2);
    EXPECT_EQ(w.session().params_json(), params);
    EXPECT_EQ(w.session().undo_depth(), undo);
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.at(0, 0, 0) == 0.3f; }));
    EXPECT_GE(*m->calls["gamma"], 1);
    // And back: alpha's fields again, from alpha, not a cache of gamma's.
    ASSERT_TRUE(use(w, m->root / "alpha"));
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.at(0, 0, 0) == 0.1f; }));
    EXPECT_EQ(w.session().params_json(), params);
}
#endif

TEST(AppModels, AModelThatDriftsIsRefusedAndTheOneInUseStays) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "drift");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    m->drifting.insert("gamma");
    QString why;
    EXPECT_FALSE(use(w, m->root / "gamma", &why));
    EXPECT_TRUE(why.startsWith("model gamma refused on ONNX Runtime on CPU: self-test FAILED (residual)")) << why.toStdString();
    EXPECT_EQ(text_of(w, "ckpt"), "alpha");
    EXPECT_EQ(w.model_package(), m->root / "alpha");
    // A folder that is not a package says so and changes nothing.
    EXPECT_FALSE(use(w, m->root, &why));
    EXPECT_TRUE(why.startsWith("This folder is not a RUDRA model package.")) << why.toStdString();
    EXPECT_EQ(text_of(w, "ckpt"), "alpha");
}

TEST(AppModels, TheGoldensRunOncePerPackageAndBackend) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "once");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    ASSERT_TRUE(use(w, m->root / "beta"));
    ASSERT_TRUE(use(w, m->root / "alpha"));
    EXPECT_EQ(m->tests["alpha"], 1);
    EXPECT_EQ(m->tests["beta"], 1);
    // Another backend is another check.
    ASSERT_TRUE(use(w, m->root / "alpha", nullptr, BackendChoice{Runtime::LibTorch, Device::Cpu}));
    EXPECT_EQ(m->tests["alpha"], 2);
    EXPECT_EQ(w.model_backend(), (std::optional<BackendChoice>{BackendChoice{Runtime::LibTorch, Device::Cpu}}));
}

TEST(AppModels, ABareStartOpensTheLastPackageOnItsBackend) {
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    {
        app::MainWindow w(false);
        auto m = fake_models(w, "boot");
        w.boot();   // nothing remembered: the registry's default
        ASSERT_TRUE(wait_for([&] { return !w.loading_model() && !w.model_package().empty(); }));
        EXPECT_EQ(w.model_package(), m->root / "beta");
        ASSERT_TRUE(use(w, m->root / "gamma", nullptr, BackendChoice{Runtime::LibTorch, Device::Cpu}));
    }
    app::MainWindow w(false);
    auto m = fake_models(w, "boot2");
    // The remembered one is from the first window's folder (still there).
    w.boot();
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && !w.model_package().empty(); }));
    EXPECT_EQ(w.model_package().filename(), "gamma");
    EXPECT_EQ(w.model_backend()->key(), "libtorch/cpu");
    EXPECT_EQ(w.findChild<QDialog*>("firstRun"), nullptr);
}

TEST(AppModels, NoPackageAnywhereSaysWhereItLooked) {
    QSettings().clear();
    QSettings().setValue("firstRun/done", true);
    app::MainWindow w(false);
    w.set_model_roots({fresh_dir("empty-a"), fresh_dir("empty-b")});
    w.boot();
    EXPECT_EQ(text_of(w, "ckpt"), "no model package found");
    EXPECT_TRUE(w.findChild<QPlainTextEdit*>("log")->toPlainText().contains(
        "no model loaded: no model package found in 2 folder(s)"));
}

TEST(AppModels, TheManagerListsSwitchesAndAddsFolders) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "manager");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.run("models");
    auto* dlg = find<app::ModelManager>(w, "modelManager");
    ASSERT_NE(dlg, nullptr);
    auto* list = dlg->list();
    ASSERT_EQ(list->topLevelItemCount(), 3);
    EXPECT_EQ(list->topLevelItem(0)->text(0), "Alpha");
    EXPECT_EQ(list->topLevelItem(0)->text(3), "in use");
    EXPECT_EQ(list->topLevelItem(1)->text(3), "opens on a bare start");
    EXPECT_EQ(list->currentItem(), list->topLevelItem(0));
    dlg->select(2);
    EXPECT_EQ(list->currentItem(), list->topLevelItem(2));
    dlg->use_selected();
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "gamma"; }));
    EXPECT_EQ(list->topLevelItem(2)->text(3), "in use");
    EXPECT_TRUE(dlg->findChild<QLabel*>("ckptStatus")->text().startsWith("In use: gamma on ONNX Runtime on CPU"));
    // A second folder joins the list.
    const auto more = fresh_dir("models-more");
    write_manifest(more / "delta", "delta");
    w.add_model_root(more);
    dlg->refresh();
    EXPECT_EQ(list->topLevelItemCount(), 4);
    EXPECT_EQ(list->topLevelItem(3)->text(0), "delta");
    // The backend list is every choice this build has, after Automatic.
    auto* combo = dlg->findChild<QComboBox*>("ckptBackend");
    EXPECT_EQ(combo->count(), int(backend_choices().size()) + 1);
}

TEST(AppModels, TheFirstRunReportsTheDisplaysRealPeak) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "first");
    w.boot();   // first start: the check, nothing loaded behind it
    auto* fr = find<app::FirstRun>(w, "firstRun");
    ASSERT_NE(fr, nullptr);
    EXPECT_TRUE(fr->isVisible());
    EXPECT_TRUE(w.model_package().empty());
    EXPECT_EQ(fr->findChild<QLabel*>("frDisplay")->text(), "Display not checked");   // no viewer in the tests
    // What the card's viewer reports on the PA279CRV through DXGI.
    fr->set_display(DisplayTarget::scrgb(418.0), "scRGB", "DXGI");
    EXPECT_EQ(fr->findChild<QLabel*>("frDisplay")->text(), "HDR display, 418 nits");
    EXPECT_TRUE(fr->findChild<QLabel*>("frDisplayDetail")->text().contains("the 600, 1,000 and 2,000 nit patches clip"));
    fr->set_display(DisplayTarget::sdr(), "SDR", "swapchain");
    EXPECT_EQ(fr->findChild<QLabel*>("frDisplay")->text(), "SDR display");
    EXPECT_TRUE(fr->findChild<QLabel*>("frModel")->text().startsWith("Beta"));
    fr->test_model();
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && !w.model_package().empty(); }));
    EXPECT_EQ(fr->findChild<QLabel*>("frModelStatus")->text(), "Ready: beta on ONNX Runtime on CPU");
    fr->start();
    EXPECT_TRUE(QSettings().value("firstRun/done").toBool());
    EXPECT_FALSE(fr->isVisible());
}


// ---------------------------------------------------------------------------
// Phase 3 step 11: the sheets, the copies, drop to open, recent shots, settings
// ---------------------------------------------------------------------------

#ifdef RUDRA_HAVE_STILL_DECODE
namespace {

std::filesystem::path decode_dir() { return std::filesystem::path(RUDRA_GOLDEN_DIR) / "decode"; }

void drop(QWidget& target, const QStringList& paths) {
    auto* mime = new QMimeData;
    QList<QUrl> urls;
    for (const auto& p : paths) urls << QUrl::fromLocalFile(p);
    mime->setUrls(urls);
    QDragEnterEvent enter(QPoint(10, 10), Qt::CopyAction, mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&target, &enter);
    QDropEvent d(QPointF(10, 10), Qt::CopyAction, mime, Qt::LeftButton, Qt::NoModifier);
    QApplication::sendEvent(&target, &d);
    delete mime;
}

QString png(const char* name) { return QString::fromStdString((decode_dir() / name).string()); }

QString last_log(app::MainWindow& w) {
    const QStringList lines = w.findChild<QPlainTextEdit*>("log")->toPlainText().split('\n', Qt::SkipEmptyParts);
    return lines.isEmpty() ? QString() : lines.last();
}

}  // namespace

TEST(AppCopy, TheCopiesAreThePagesTextsOfTheFrameOnScreen) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "copy");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.refresh_enabled();
    EXPECT_FALSE(w.action("copy-metrics")->isEnabled());   // !!state.metrics
    EXPECT_FALSE(w.action("copy-delivery")->isEnabled());  // no frame
    w.add_files({decode_dir() / "png8_rgb.png"});
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() != nullptr; }));
    w.measure_now();
    w.refresh_enabled();
    ASSERT_NE(w.measurement(), nullptr);
    EXPECT_TRUE(w.action("copy-metrics")->isEnabled());
    EXPECT_TRUE(w.action("copy-scopes")->isEnabled());
    EXPECT_TRUE(w.action("copy-delivery")->isEnabled());

    w.run("copy-metrics");
    const auto* me = w.measurement();
    EXPECT_EQ(QGuiApplication::clipboard()->text().toStdString(),
              metrics_value(me->measured.metrics, me->compose_ms).stringify());
    EXPECT_EQ(last_log(w), "copied measurements to the clipboard");
    w.run("copy-scopes");
    const json scopes = json::parse(QGuiApplication::clipboard()->text().toStdString());
    EXPECT_EQ(scopes["lo"].size(), 230u);
    EXPECT_EQ(scopes["histogram"].size(), 76u);
    EXPECT_EQ(last_log(w), "copied scope data to the clipboard");

    // Ungraded: region_ev is null; graded, it is the three regions.
    w.run("copy-delivery");
    json d = json::parse(QGuiApplication::clipboard()->text().toStdString());
    std::vector<std::string> keys;
    for (auto it = d.begin(); it != d.end(); ++it) keys.push_back(it.key());
    EXPECT_EQ(d["checkpoint"], "alpha");
    EXPECT_TRUE(d["step"].is_null());
    EXPECT_EQ(d["frame"], "png8_rgb.png");
    EXPECT_EQ(d["container"], "ACES 2065-1 (AP0)");
    EXPECT_EQ(d["diffuse_white_nits"], 203);
    EXPECT_TRUE(d["region_ev"].is_null());
    EXPECT_EQ(d["measurements"]["maxcll"], me->measured.metrics.maxcll);
    EXPECT_EQ(last_log(w), "copied delivery metadata to the clipboard");
    w.session().region_press(1, 100.0);
    w.session().region_move(130.0, false);
    w.session().region_release();
    w.run("container-linear");
    w.run("copy-delivery");
    d = json::parse(QGuiApplication::clipboard()->text().toStdString());
    ASSERT_TRUE(d["region_ev"].is_array());
    EXPECT_EQ(d["region_ev"][1]["label"], "speculars");
    EXPECT_NE(d["region_ev"][1]["ev"], 0.0);
    EXPECT_EQ(d["container"], "linear Rec.2020");
    // The text is the page's bytes for this state (core/copy_texts, held to the page).
    EXPECT_EQ(QGuiApplication::clipboard()->text().toStdString(), w.delivery_text());
}

TEST(AppCopy, SheetsAreThePagesAndAClickClosesThem) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "sheets");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.run("shortcuts");
    ASSERT_NE(w.sheet(), nullptr);
    EXPECT_TRUE(w.sheet()->isVisible());
    EXPECT_EQ(w.sheet()->findChild<QLabel*>("sheetTitle")->text(), "Keyboard");
    EXPECT_EQ(w.sheet()->findChild<QLabel*>("sheetClose")->text(), "Esc, or click anywhere, to close");
    // The title, the rows (key and what), the close line.
    EXPECT_EQ(w.sheet()->findChildren<QLabel*>().size(), qsizetype(2 + 2 * shortcut_sheet().size()));
    QPointer<QDialog> first = w.sheet();
    QTest::mouseClick(first, Qt::LeftButton);
    QApplication::processEvents();
    QApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    EXPECT_TRUE(first.isNull() || !first->isVisible());
    w.run("about");
    ASSERT_NE(w.sheet(), nullptr);
    bool device = false;
    const auto labels = w.sheet()->findChildren<QLabel*>();
    for (qsizetype i = 0; i + 1 < labels.size(); ++i)
        if (labels[i]->text() == "Device") device = labels[i + 1]->text() == "CPU";   // $("device").textContent
    EXPECT_TRUE(device);
    QTest::keyClick(w.sheet(), Qt::Key_Escape);
    QApplication::processEvents();
    EXPECT_TRUE(w.sheet() == nullptr || !w.sheet()->isVisible());
}

TEST(AppOpen, ADropAddsFramesOpensAShotOrUsesAModel) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "drop");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.show();
    ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
    // Enter lights the drop zone; the drop puts it out.
    auto* dz = w.findChild<QWidget*>("dropzone");
    ASSERT_NE(dz, nullptr);
    {
        QMimeData mime;
        mime.setUrls({QUrl::fromLocalFile(png("png8_rgb.png"))});
        QDragEnterEvent enter(QPoint(10, 10), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
        QApplication::sendEvent(&w, &enter);
        EXPECT_TRUE(enter.isAccepted());
        EXPECT_EQ(dz->property("state").toString(), "hot");
        QDragLeaveEvent leave;
        QApplication::sendEvent(&w, &leave);
        EXPECT_EQ(dz->property("state").toString(), "");
    }
    // Two stills: the frames, the first shown (addFiles).
    drop(w, {png("png8_rgb.png"), png("png16_rgb.png")});
    EXPECT_EQ(w.frame_count(), 2u);
    EXPECT_EQ(w.current_index(), 0);
    EXPECT_EQ(last_log(w), "added 2 frames");
    // One more: appended, and it is the one shown.
    drop(w, {png("bmp24.bmp")});
    EXPECT_EQ(w.frame_count(), 3u);
    EXPECT_EQ(w.current_index(), 2);
    EXPECT_EQ(last_log(w), "added 1 frame");
    ASSERT_TRUE(wait_for([&] { return w.frame_fields() && w.frame_fields()->residual.width() > 0 &&
                                      w.findChild<QLabel*>("shotCount")->text() == "3"; }));
    // A model package folder: used, the frames kept.
    drop(w, {QString::fromStdString((m->root / "beta").string())});
    ASSERT_TRUE(wait_for([&] { return !w.loading_model() && w.model_package().filename() == "beta"; }));
    EXPECT_EQ(w.frame_count(), 3u);
    // A folder of frames: that shot instead.
    drop(w, {QString::fromStdString(decode_dir().string())});
    EXPECT_GE(w.frame_count(), 10u);
    EXPECT_EQ(w.current_index(), 0);
}

TEST(AppOpen, RecentShotsNewestFirstAndClearable) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "recent");
    ASSERT_TRUE(use(w, m->root / "alpha"));
    w.open_source(QString::fromStdString(decode_dir().string()));
    w.add_files({decode_dir() / "png8_rgb.png"});
    w.open_source(QString::fromStdString(decode_dir().string()));   // again: moves to the top, not twice
    const QStringList r = w.recent_sources();
    ASSERT_EQ(r.size(), 2);
    EXPECT_EQ(r[0].toStdString(), decode_dir().lexically_normal().string());
    EXPECT_TRUE(r[1].endsWith("png8_rgb.png"));
    // The menu: File > Open recent, the shots then Clear recent.
    QMenu* file = w.menuBar()->actions()[0]->menu();
    QMenu* recent = nullptr;
    for (QAction* a : file->actions())
        if (a->menu() && a->text() == "Open recent") recent = a->menu();
    ASSERT_NE(recent, nullptr);
    w.fill_recent();
    std::vector<QAction*> items;
    for (QAction* a : recent->actions())
        if (a->objectName().startsWith("recent:")) items.push_back(a);
    ASSERT_EQ(items.size(), 2u);
    EXPECT_EQ(recent->actions().last(), w.action("recent-clear"));
    EXPECT_TRUE(items[1]->text().startsWith("png8_rgb.png"));
    items[1]->trigger();   // a still: added to the frames
    EXPECT_EQ(last_log(w), "added 1 frame");
    w.run("recent-clear");
    EXPECT_TRUE(w.recent_sources().isEmpty());
    EXPECT_FALSE(w.action("recent-clear")->isEnabled());
}
#endif

TEST(AppSettings, TheWindowComesBackAsItWasLeft) {
    QSettings().clear();
    QSize left_at;
    {
        app::MainWindow w(false);
        w.set_workspace("simple");
        w.show_tab("deliver");
        w.run("rail-left");
        w.run("scopes");
        w.run("container-linear");
        w.findChild<QLineEdit*>("renderDir")->setText("/renders/show");
        w.findChild<QLineEdit*>("renderName")->setText("sh010");
        w.findChild<QComboBox*>("renderMode")->setCurrentIndex(1);
        w.findChild<QSpinBox*>("renderStart")->setValue(1001);
        w.show();
        ASSERT_TRUE(QTest::qWaitForWindowExposed(&w));
        w.resize(720, 560);   // inside the offscreen screen (800 x 800), so nothing clamps it
        QApplication::processEvents();
        left_at = w.size();   // what the window made of it (its minimum may be wider)
        w.close();   // closeEvent saves
    }
    app::MainWindow w(false);
    EXPECT_EQ(w.workspace(), "full");   // nothing restored until asked (the tests' windows start clean)
    w.restore_settings();
    EXPECT_EQ(w.workspace(), "simple");
    EXPECT_EQ(w.tab(), "deliver");
    EXPECT_FALSE(w.session().rail_left);
    EXPECT_TRUE(w.session().rail_right);
    EXPECT_FALSE(w.session().scopes_open);
    EXPECT_FALSE(w.action("rail-left")->isChecked());
    EXPECT_EQ(w.container(), "linear");
    EXPECT_EQ(w.findChild<QLineEdit*>("renderDir")->text(), "/renders/show");
    EXPECT_EQ(w.findChild<QLineEdit*>("renderName")->text(), "sh010");
    EXPECT_EQ(w.findChild<QComboBox*>("renderMode")->currentIndex(), 1);
    EXPECT_EQ(w.findChild<QSpinBox*>("renderStart")->value(), 1001);
    // The size (restoreGeometry keeps a window on its screen: the offscreen
    // one is 800 wide, narrower than the window's minimum, so the height is
    // the part that shows the round trip here).
    EXPECT_EQ(w.size().height(), left_at.height());
    QSettings().clear();
}


#ifdef RUDRA_HAVE_STILL_DECODE
// Phase 3 step 12: the scripted workflow runs end to end (with the fakes; the
// real model and the CLI comparison run in scripts/NATIVE_PHASE3_EXIT.ps1).
TEST(AppWorkflow, TheScriptedWorkflowPassesAndReportsEveryStep) {
    QSettings().clear();
    app::MainWindow w(false);
    auto m = fake_models(w, "workflow");
    const auto shot = fresh_dir("workflow-shot");
    std::filesystem::create_directories(shot);
    int k = 0;
    for (const char* f : {"png8_rgb.png", "png16_rgb.png", "bmp24.bmp", "jpeg_q92_420.jpg", "png8_grey.png"})
        std::filesystem::copy_file(decode_dir() / f, shot / ("f" + std::to_string(k++) + "_" + f));
    app::WorkflowArgs a;
    a.report = QString::fromStdString((shot / "report.json").string());
    a.package = QString::fromStdString((m->root / "beta").string());
    a.frames = QString::fromStdString(shot.string());
    a.out = QString::fromStdString((fresh_dir("workflow-out")).string());
    a.check_every = 1;
    EXPECT_EQ(app::run_workflow_check(w, a), 0);
    std::ifstream in(shot / "report.json");
    const json r = json::parse(in);
    EXPECT_EQ(r["verdict"], "PASS");
    for (const char* s : {"model", "open", "scrub", "grade", "compare", "measure", "master"})
        EXPECT_TRUE(r[s]["ok"].get<bool>()) << s;
    EXPECT_EQ(r["scrub"]["delivered"], 5);
    EXPECT_EQ(r["scrub"]["checked_against_decode"], 5);
    EXPECT_EQ(r["master"]["masters"].size(), 3u);
    // Each master names the frame, the files and the parameters the CLI takes.
    const auto& first = r["master"]["masters"][0];
    EXPECT_TRUE(std::filesystem::exists(first["exr"].get<std::string>()));
    auto q = master_request_from_json(first["params"].get<std::string>());
    ASSERT_TRUE(q);
    EXPECT_EQ(q->checkpoint, "beta");
    EXPECT_EQ(q->recovery_mode, "highlights");
    EXPECT_NEAR(q->strength, 1.2, 1e-12);
}
#endif

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    // Settings of their own, never the user's.
    QCoreApplication::setOrganizationName("FXTD Studios (tests)");
    QCoreApplication::setApplicationName("rudra_app_tests");
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat, QSettings::UserScope,
                       QString::fromStdString((std::filesystem::temp_directory_path() /
                                                 ("rudra-app-tests-settings-" + std::to_string(QCoreApplication::applicationPid())))
                                                    .string()));
    app::ThemeReport theme = app::apply_theme(app);
    g_theme = &theme;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
