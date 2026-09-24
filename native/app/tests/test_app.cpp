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
#include <QTest>

#include <fstream>
#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include "main_window.hpp"
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
    EXPECT_TRUE(w.action("rail-left")->toolTip().contains("step 4"));
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

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);
    app::ThemeReport theme = app::apply_theme(app);
    g_theme = &theme;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
