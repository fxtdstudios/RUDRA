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
#include <QTest>
#include <QTextDocumentFragment>

#include <array>
#include <cmath>

#include <fstream>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "main_window.hpp"
#include "widgets.hpp"
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
    EXPECT_TRUE(w.action("master")->toolTip().contains("step 9"));
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

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    app::ThemeReport theme = app::apply_theme(app);
    g_theme = &theme;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
