#include "main_window.hpp"

#include "rudra/video/ffmpeg_check.hpp"

#include <QActionGroup>
#include <QApplication>
#include <QDialog>
#include <QFileDialog>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QGridLayout>
#include <QKeySequence>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPointer>
#include <QStatusBar>
#include <QFrame>
#include <QKeyEvent>
#include <QClipboard>
#include <QCloseEvent>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <QLayout>
#include <QComboBox>
#include <QLineEdit>
#include <QLocale>
#include <QSpinBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QSettings>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QStyle>
#include <QVBoxLayout>

#include "export_sheet.hpp"
#include "model_dialogs.hpp"
#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "widgets.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <thread>

#include "rudra/core/baseline.hpp"
#include "rudra/core/copy_texts.hpp"
#include "rudra/core/readouts.hpp"
#include "rudra/engine/actions.hpp"
#include "rudra/engine/frame_engine.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/infer/tiler.hpp"
#include "rudra/media/sequence.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif
#ifdef RUDRA_HAVE_STILL_DECODE
#include "rudra/media/still.hpp"
#endif

namespace rudra::app {
namespace {

QString qs(std::string_view s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }

QString describe(const ModelManifest& m) {
    return QStringLiteral("%1  ·  contract %2  ·  residual gate %3  ·  shadow gate %4  ·  curve %5  ·  corpus EV %6")
        .arg(QString::fromStdString(m.name), QString::fromStdString(m.contract), m.has_residual_gate ? "yes" : "no",
             m.has_shadow_gate ? "yes" : "no", m.has_curve ? "yes" : "no")
        .arg(m.corpus_ev, 0, 'f', 1);
}

// The page's key (KeyboardEvent.key) as a Qt shortcut. Letters without
// Shift: the page maps both cases to one action.
QKeySequence key_sequence(std::string_view k) {
    if (k == " ") return QKeySequence(Qt::Key_Space);
    if (k == ",") return QKeySequence(Qt::Key_Comma);
    if (k == ".") return QKeySequence(Qt::Key_Period);
    if (k == "?") return QKeySequence(Qt::Key_Question);
    if (k == "[") return QKeySequence(Qt::Key_BracketLeft);
    if (k == "]") return QKeySequence(Qt::Key_BracketRight);
    if (k == "Home") return QKeySequence(Qt::Key_Home);
    if (k == "End") return QKeySequence(Qt::Key_End);
    if (k.size() == 1) return QKeySequence(qs(k).toUpper());
    return QKeySequence(qs(k));
}

// Which actions are one choice among several: the part of data-check before
// the colon ("mode:all" is in the "mode" group).
QString check_group(std::string_view check) {
    const auto colon = check.find(':');
    return colon == std::string_view::npos ? QString() : qs(check.substr(0, colon));
}

}  // namespace

MainWindow::MainWindow(bool with_viewer) {
    setWindowTitle("RUDRA");
    resize(1600, 1000);
    session_.on_change([this](std::uint32_t what) { session_changed(what); });
    build_ui(with_viewer);
    setAcceptDrops(true);   // a drop anywhere opens (the page's window drop)
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        viewer_->on_status([this](const ViewerStatus&) {
            // A wipe dragged or dropped on the plate itself: the session follows.
            const double w = viewer_->view().wipe;
            const std::optional<double> now = w >= 0.0 ? std::optional<double>(w) : std::nullopt;
            if (now != session_.wipe) {
                session_.wipe = now;
                sync_checks();
            }
            show_status();
        });
    }
#endif
    bind_handlers();
    build_menus();
    sync_checks();
    sync_ui();
    refresh_enabled();
    update_pipe();

    // The trailing edge of the page's scheduleStats: measure 90 ms after the
    // grade last moved, on another thread; the picture never waits for it.
    stats_timer_.setSingleShot(true);
    stats_timer_.setInterval(90);
    connect(&stats_timer_, &QTimer::timeout, this, [this] { run_stats(); });
    // 24 fps, the Studio's default; a frame that is not ready is held, not skipped.
    // Connected once here: Qt::UniqueConnection does not apply to a lambda, so
    // connecting on every Play added a handler each time and playback ran fast.
    play_.setInterval(1000 / 24);
    connect(&play_, &QTimer::timeout, this, [this] {
        if (!waiting_) step_to(current_ + 1);
    });

    // The floating probe box (#probeBox): it follows the cursor over the viewer,
    // a window of its own so it can sit over the viewer's swapchain.
    probe_box_ = new QFrame(this, Qt::ToolTip | Qt::FramelessWindowHint);
    probe_box_->setObjectName("probeBox");
    probe_box_->setAttribute(Qt::WA_ShowWithoutActivating);
    probe_box_->setAttribute(Qt::WA_TransparentForMouseEvents);
    auto* pbl = new QGridLayout(probe_box_);
    pbl->setContentsMargins(8, 6, 8, 6);
    pbl->setHorizontalSpacing(10);
    pbl->setVerticalSpacing(1);
    probe_box_->hide();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        viewer_->on_drop([this](const QStringList& paths) { open_paths(paths); });
        viewer_->on_hover([this](const ViewerWindow::Hover& h) {
            if (!(probe_on_ || h.alt) || !h.x) {
                probe_pixel(std::nullopt);
                return;
            }
            QWidget* host = findChild<QWidget*>("viewerHost");
            probe_pixel(std::pair{*h.x, *h.y}, host ? host->mapToGlobal(h.window.toPoint()) : QPoint());
        });
    }
#endif

    QStringList runtimes;
    for (auto r : compiled_runtimes()) runtimes << to_string(r);
    runtimes_ = runtimes.isEmpty() ? "no inference runtime" : runtimes.join(", ");
    log("runtimes: " + runtimes_);
    log(QString("container: ") + container_field_->text());
}

MainWindow::~MainWindow() {
    queues_->finished = nullptr;
    queues_->stop();      // a running export is left interrupted, as Ctrl+C leaves it
    queues_.reset();
    if (model_worker_.joinable()) model_worker_.join();   // a package still loading
    if (stats_worker_.joinable()) stats_worker_.join();   // a measurement in flight uses the frame
    master_job_.reset();   // cancels and waits: it uses the backend
    play_.stop();
    engine_.reset();   // joins the worker before the backend goes
}

QAction* MainWindow::action(std::string_view id) const {
    const auto it = actions_.find(id);
    return it == actions_.end() ? nullptr : it->second;
}

QString MainWindow::pending_reason(std::string_view id) const {
    const auto it = pending_.find(id);
    return it == pending_.end() ? QString() : it->second;
}

void MainWindow::run(std::string_view id) {
    QAction* a = action(id);
    if (a && !a->isEnabled()) return;
    const auto it = handlers_.find(id);
    if (it != handlers_.end()) it->second();
    sync_checks();
}

void MainWindow::bind_handlers() {
    auto& h = handlers_;
    h["open"] = [this] {
        // The page's file input: several stills at once, added to the frames.
        const QStringList files = QFileDialog::getOpenFileNames(
            this, "Open SDR frames", {}, "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)");
        std::vector<std::filesystem::path> paths;
        for (const auto& f : files) paths.emplace_back(f.toStdString());
        if (!paths.empty()) add_files(paths);
    };
    h["open-folder"] = [this] { open_source({}, true); };
    h["open-package"] = [this] { open_package(); };
    h["models"] = [this] { open_model_manager(); };
    h["first-run"] = [this] { open_first_run(); };
    h["quit"] = [] { QApplication::quit(); };
    h["close"] = [this] { close_frames(); };
    h["first"] = [this] { step_to(0); };
    h["prev"] = [this] { step_to(current_ - 1); };
    h["next"] = [this] { step_to(current_ + 1); };
    h["last"] = [this] { step_to(int(frames_.size()) - 1); };
    h["play"] = [this] { toggle_play(); };
    // The page's grading state and its undo: engine/session.
    for (const auto& spec : action_specs())
        if (Session::owns(spec.id)) h[std::string(spec.id)] = [this, id = std::string(spec.id)] { session_.run(id); };
    h["shortcuts"] = [this] {
        std::vector<std::pair<QString, QString>> rows;
        for (const auto& r : shortcut_sheet()) rows.emplace_back(qs(r.keys), qs(r.what));
        show_sheet("Keyboard", rows);
    };
    h["about"] = [this] {
        const QString device = device_->text();   // the page's $("device").textContent
        QString composite = "CPU";
#ifdef RUDRA_APP_VIEWER
        if (viewer_) composite = QStringLiteral("GPU, QRhi %1 — render/shaders/composite.frag")
                                     .arg(QString::fromStdString(viewer_->status().backend));
#endif
        // head.checkpoint: the model that made the frame on screen.
        show_sheet("RUDRA Studio", {{"Checkpoint", manifest_ && current_frame_ ? QString::fromStdString(manifest_->name) : "—"},
                                    {"Step", "—"},
                                    {"Device", device},
                                    {"Composite", composite},
                                    {"Storage", "log2_extended, 0.005–1 000 000 nits"},
                                    {"Master", "ACES 2065-1 (AP0), ST 2065-4 chromaticities"},
                                    {"Built by", "FXTD Studios / Radiance Research"}});
    };

    h["view-image"] = [this] { session_.set_view_layer(0); };
    h["view-false-colour"] = [this] { session_.set_view_layer(1); };
    h["view-difference"] = [this] { session_.set_view_layer(2); };

    // Waiting on later steps of Phase 3: off, and they say why.
    // copy(): JSON.stringify(value, null, 2) to the clipboard (core/copy_texts).
    h["copy-metrics"] = [this] {
        if (measure_) copy_text("measurements", metrics_value(measure_->measured.metrics, measure_->compose_ms).stringify());
    };
    h["copy-scopes"] = [this] {
        if (measure_) copy_text("scope data", scopes_value(measure_->measured.scopes).stringify());
    };
    h["copy-delivery"] = [this] { copy_text("delivery metadata", delivery_text()); };
    h["recent-clear"] = [this] {
        QSettings().remove("recent/sources");
        fill_recent();
    };
    h["remeasure"] = [this] {
        measure_now();
        log("re-measured");
    };
    h["master"] = [this] {
        if (mastering()) {   // while it renders, the button stops it after this frame
            master_job_->cancel();
            findChild<QLabel*>("renderStatus")->setText("Stopping after this frame\u2026");
            return;
        }
        master();
    };

#ifdef RUDRA_APP_VIEWER
    {
        auto guides = [this](auto&& edit) {
            return [this, edit] {
                if (!viewer_) return;
                auto g = viewer_->guides();
                edit(g);
                viewer_->set_guides(g);
            };
        };
        h["zoom-fit"] = [this] { if (viewer_) viewer_->zoom_fit(); };
        h["zoom-actual"] = [this] { if (viewer_) viewer_->zoom_actual(); };
        h["guides-action"] = guides([](GuideOptions& g) { g.action_safe = !g.action_safe; });
        h["guides-title"] = guides([](GuideOptions& g) { g.title_safe = !g.title_safe; });
        h["guides-centre"] = guides([](GuideOptions& g) { g.centre = !g.centre; });
        const std::pair<const char*, double> aspects[] = {{"aspect-none", 0.0}, {"aspect-2.39", 2.39},
                                                          {"aspect-1.85", 1.85}, {"aspect-16:9", 16.0 / 9.0},
                                                          {"aspect-4:3", 4.0 / 3.0}, {"aspect-1:1", 1.0}};
        for (const auto& [id, r] : aspects) h[id] = guides([aspect = r](GuideOptions& g) { g.aspect = aspect; });
    }
#endif
    for (const auto& a : action_specs()) {
        const std::string id(a.id);
        if (!handlers_.count(id) && !pending_.count(id)) pending_[id] = "This build has no viewer.";
    }
}

void MainWindow::build_menus() {
    std::map<QString, QActionGroup*> groups;
    for (const auto& spec : action_specs()) {
        auto* a = new QAction(qs(spec.label), this);
        a->setObjectName(QStringLiteral("act:") + qs(spec.id));
        QList<QKeySequence> keys;
        for (auto k : mapped_keys()) {
            if (action_for_key(k) != spec.id) continue;
            const QKeySequence seq = key_sequence(k);
            if (!keys.contains(seq)) keys << seq;
            // The page maps "z" and "Z" alike: a letter works with Shift (or
            // Caps Lock) too.
            if (k.size() == 1 && std::isalpha(static_cast<unsigned char>(k[0]))) {
                const QKeySequence shifted(QStringLiteral("Shift+") + seq.toString(QKeySequence::PortableText));
                if (!keys.contains(shifted)) keys << shifted;
            }
        }
        if (!spec.native_key.empty()) keys << QKeySequence(qs(spec.native_key));
        // The hint the page prints comes first, so the menu shows it.
        if (!spec.hint.empty()) {
            const QKeySequence hint = key_sequence(spec.hint == "Space" ? " " : spec.hint);
            keys.removeAll(hint);
            keys.prepend(hint);
        }
        a->setShortcuts(keys);
        a->setShortcutContext(Qt::WindowShortcut);
        if (!spec.check.empty()) {
            a->setCheckable(true);
            const QString g = check_group(spec.check);
            if (!g.isEmpty()) {
                auto*& group = groups[g];
                if (!group) group = new QActionGroup(this);
                group->addAction(a);
            }
        }
        const std::string id(spec.id);
        connect(a, &QAction::triggered, this, [this, id] { run(id); });
        actions_[id] = a;
        addAction(a);   // shortcuts work with the menubar hidden too
    }
    for (const auto& m : menu_specs()) {
        QMenu* menu = menuBar()->addMenu(qs(m.title));
        std::vector<QMenu*> stack{menu};
        for (auto e : m.entries) {
            if (e == "-") stack.back()->addSeparator();
            else if (e == "<") stack.pop_back();
            else if (e.substr(0, 1) == ">") {
                QMenu* sub = stack.back()->addMenu(qs(e.substr(1)));
                if (e == ">Open recent") {
                    recent_menu_ = sub;
                    connect(sub, &QMenu::aboutToShow, this, [this] { fill_recent(); });
                }
                stack.push_back(sub);
            }
            else stack.back()->addAction(actions_.at(std::string(e)));
        }
        connect(menu, &QMenu::aboutToShow, this, [this] {
            sync_checks();
            refresh_enabled();
        });
        // shell.js TAB_FOR_ACT: a menu item that changes something on a tab
        // that is not showing brings that tab forward (menus only, not keys).
        connect(menu, &QMenu::triggered, this, [this](QAction* a) {
            static const std::map<QString, QString> tab_for = {
                {"act:reset-regions", "grade"},  {"act:container-aces", "deliver"}, {"act:container-linear", "deliver"},
                {"act:copy-delivery", "deliver"}, {"act:mode-all", "rec"},         {"act:mode-highlights", "rec"},
                {"act:mode-shadows", "rec"},      {"act:mode-off", "rec"},         {"act:preserve", "rec"},
                {"act:strength-up", "rec"},       {"act:strength-down", "rec"},    {"act:reset-recon", "rec"}};
            const auto it = tab_for.find(a->objectName());
            if (it != tab_for.end()) show_tab(it->second);
        });
    }
}

void MainWindow::refresh_enabled() {
    const bool any = !frames_.empty(), many = frames_.size() > 1;
    for (const auto& spec : action_specs()) {
        QAction* a = action(spec.id);
        const QString why = pending_reason(spec.id);
        bool on = true;
        switch (spec.enable) {
            case EnableRule::Always: break;
            case EnableRule::AnyFrames: on = any; break;
            case EnableRule::ManyFrames: on = many; break;
            case EnableRule::CanMaster: on = (any && backend_ != nullptr) || mastering(); break;
            case EnableRule::CanUndo: on = session_.undo_depth() > 0; break;
            case EnableRule::CanRedo: on = session_.redo_depth() > 0; break;
            case EnableRule::HasMetrics:
            case EnableRule::HasScopes: on = measure_ != nullptr; break;
        }
        a->setEnabled(on && why.isEmpty());
        a->setToolTip(why);
        a->setStatusTip(why);
    }
}

void MainWindow::sync_checks() {
    auto set = [this](std::string_view id, bool on) {
        if (QAction* a = action(id)) a->setChecked(on);
    };
    const auto& grade = session_.grade;
    set("mode-all", grade.mode == "all");
    set("mode-highlights", grade.mode == "highlights");
    set("mode-shadows", grade.mode == "shadows");
    set("mode-off", grade.mode == "off");
    set("preserve", grade.preserve);
    set("container-aces", session_.container == "aces");
    set("container-linear", session_.container == "linear");
    set("wipe", session_.wipe.has_value());
    set("view-image", session_.view_layer == 0);
    set("view-false-colour", session_.view_layer == 1);
    set("view-difference", session_.view_layer == 2);
    set("rail-left", session_.rail_left);
    set("rail-right", session_.rail_right);
    set("scopes", session_.scopes_open);
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const auto g = viewer_->guides();
        const bool fit = viewer_->viewport().scale <= 0.0;
        set("zoom-fit", fit);
        set("zoom-actual", !fit);
        set("guides-action", g.action_safe);
        set("guides-title", g.title_safe);
        set("guides-centre", g.centre);
        set("aspect-none", g.aspect <= 0.0);
        set("aspect-2.39", std::abs(g.aspect - 2.39) < 1e-9);
        set("aspect-1.85", std::abs(g.aspect - 1.85) < 1e-9);
        set("aspect-16:9", std::abs(g.aspect - 16.0 / 9.0) < 1e-9);
        set("aspect-4:3", std::abs(g.aspect - 4.0 / 3.0) < 1e-9);
        set("aspect-1:1", std::abs(g.aspect - 1.0) < 1e-9);
        return;
    }
#endif
    set("zoom-fit", true);
}

namespace {

// The page's overlay sheet: Esc, or a click anywhere, closes it.
class Sheet : public QDialog {
public:
    using QDialog::QDialog;

protected:
    void mousePressEvent(QMouseEvent*) override { accept(); }
};

}  // namespace

void MainWindow::show_sheet(const QString& title, const std::vector<std::pair<QString, QString>>& rows) {
    if (sheet_) sheet_->close();
    auto* d = new Sheet(this);
    d->setAttribute(Qt::WA_DeleteOnClose);
    d->setObjectName("sheet");
    d->setWindowTitle(title);
    auto* grid = new QGridLayout(d);
    auto* head = new QLabel(title, d);
    head->setObjectName("sheetTitle");
    head->setProperty("role", "plabel");
    grid->addWidget(head, 0, 0, 1, 2);
    int r = 1;
    for (const auto& [k, v] : rows) {
        auto* kl = new QLabel(k, d);
        kl->setProperty("role", "value");
        auto* vl = new QLabel(v, d);
        vl->setTextInteractionFlags(Qt::TextSelectableByMouse);
        grid->addWidget(kl, r, 0);
        grid->addWidget(vl, r, 1);
        ++r;
    }
    auto* close = new QLabel("Esc, or click anywhere, to close", d);
    close->setObjectName("sheetClose");
    close->setProperty("role", "note");
    grid->addWidget(close, r, 0, 1, 2);
    sheet_ = d;
    d->show();
}

void MainWindow::session_changed(std::uint32_t what) {
    if (what & Session::Grade) schedule_stats();
    if (what & (Session::Peak | Session::Delivery)) update_pipe();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        if (what & Session::Grade) viewer_->set_composite(session_.composite_params());
        if (what & (Session::Peak | Session::Wipe | Session::View)) {
            auto v = viewer_->view();
            v.display_nits = session_.display_nits();
            v.wipe = session_.wipe ? *session_.wipe : -1.0;
            v.mode = session_.view_layer == 1 ? ViewMode::FalseColour
                     : session_.view_layer == 2 ? ViewMode::Difference : ViewMode::Image;
            v.show = session_.show == "baseline" ? ViewSource::Baseline : ViewSource::Model;
            viewer_->set_view(v);
        }
    }
#endif
    sync_checks();
    sync_ui();
    refresh_enabled();
}

void MainWindow::open_package(const QString& preset) {
    const QString dir = preset.isEmpty() ? QFileDialog::getExistingDirectory(this, "Open model package") : preset;
    if (dir.isEmpty()) return;
    use_model(dir.toStdString(), backend_choice_);
}

// ---------------------------------------------------------------------------
// The checkpoint manager (step 10)
// ---------------------------------------------------------------------------

struct MainWindow::LoadedModel {
    ModelManifest manifest;
    OpenedBackend opened;
    std::optional<SelfTestReport> report;   // nullopt: this package passed on this backend before
    std::optional<Error> error;
};

namespace {

// The package's bytes (the checkpoint's and every file's hash) and the backend.
QString verified_key(const ModelManifest& m, const BackendChoice& c) {
    std::string k = m.source_sha256;
    for (const auto& [file, sha] : m.file_sha256) k += "|" + sha;
    return QString::fromStdString(k + "|" + c.key());
}

}  // namespace

void MainWindow::set_model_roots(std::vector<std::filesystem::path> roots) {
    roots_override_ = std::move(roots);
    rescan_models();
}

std::vector<std::filesystem::path> MainWindow::model_roots() const {
    if (roots_override_) return *roots_override_;
    QSettings st;
    std::vector<std::filesystem::path> extra;
    for (const auto& r : st.value("model/roots").toStringList()) extra.emplace_back(r.toStdString());
    const auto app_models = std::filesystem::path(QCoreApplication::applicationDirPath().toStdString()) / "models";
    const QString user = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    return package_roots(extra, app_models,
                         user.isEmpty() ? std::filesystem::path() : std::filesystem::path(user.toStdString()) / "models");
}

void MainWindow::add_model_root(const std::filesystem::path& root) {
    if (roots_override_) {
        if (std::find(roots_override_->begin(), roots_override_->end(), root) == roots_override_->end())
            roots_override_->push_back(root);
    } else {
        QSettings st;
        QStringList r = st.value("model/roots").toStringList();
        const QString q = QString::fromStdString(root.string());
        if (!r.contains(q)) r << q;
        st.setValue("model/roots", r);
    }
    rescan_models();
}

void MainWindow::rescan_models() { catalog_ = scan_packages(model_roots()); }

void MainWindow::use_model(const std::filesystem::path& package, std::optional<BackendChoice> choice,
                           std::function<void(bool, QString)> done) {
    auto refuse = [&](const QString& why) {
        log(why);
        if (done) done(false, why);
    };
    if (mastering()) return refuse("A render is running: stop it before switching models.");
    if (loading_model_) return refuse("A model is still loading.");
    auto m = read_manifest(package);
    if (!m) return refuse(QString::fromStdString(m.error().message + " " + m.error().detail));
    // Which pairs of this package and a backend have passed before.
    QSettings st;
    const QStringList verified = st.value("model/verified").toStringList();
    loading_model_ = true;
    ckpt_->setText(QStringLiteral("loading %1…").arg(QString::fromStdString(m->name)));
    log("loading " + describe(*m));
    refresh_enabled();
    if (model_worker_.joinable()) model_worker_.join();
    QPointer<MainWindow> self(this);
    auto hooks = hooks_;
    model_worker_ = std::thread([self, man = *m, choice, verified, hooks, done]() mutable {
        auto loaded = std::make_shared<LoadedModel>();
        loaded->manifest = man;
        auto run = [&]() -> std::optional<Error> {
            if (hooks.verify_files)
                if (auto v = verify_package_files(man); !v) return v.error();
            auto o = hooks.open ? hooks.open(man, choice) : open_backend(man, choice);
            if (!o) return o.error();
            loaded->opened = std::move(*o);
            if (!verified.contains(verified_key(man, loaded->opened.choice))) {
                auto r = hooks.test ? hooks.test(man, *loaded->opened.backend) : self_test(man, *loaded->opened.backend);
                if (!r) return r.error();
                loaded->report = std::move(*r);
            }
            return std::nullopt;
        };
        loaded->error = run();
        QMetaObject::invokeMethod(qApp, [self, loaded, done] {
            if (self) self->adopt_model(loaded, done);
        });
    });
}

void MainWindow::adopt_model(std::shared_ptr<LoadedModel> loaded, const std::function<void(bool, QString)>& done) {
    if (model_worker_.joinable()) model_worker_.join();
    loading_model_ = false;
    const ModelManifest& m = loaded->manifest;
    QString why;
    if (loaded->error) {
        why = describe(m) + "  ·  " + QString::fromStdString(loaded->error->message) +
              (loaded->error->detail.empty() ? "" : " " + QString::fromStdString(loaded->error->detail));
    } else if (loaded->report && !loaded->report->pass()) {
        why = QStringLiteral("model %1 refused on %2: self-test %3")
                  .arg(QString::fromStdString(m.name), QString::fromStdString(loaded->opened.choice.label()),
                       QString::fromStdString(loaded->report->summary()));
    }
    if (!why.isEmpty()) {
        log(why);
        set_model_pills();   // the model in use, if any, is still in use
        refresh_enabled();
        if (done) done(false, why);
        return;
    }
    if (loaded->report) {
        log("self-test " + QString::fromStdString(loaded->report->summary()));
        QSettings st;
        QStringList v = st.value("model/verified").toStringList();
        v << verified_key(m, loaded->opened.choice);
        st.setValue("model/verified", v);
    }
    // The engine's worker uses the backend: it goes first. The session and
    // the frames stay; the frame on screen is asked for again.
    const bool had_frames = !frames_.empty();
    play_.stop();
    btn_play_->set_glyph(IconButton::Glyph::Play);
    {
        std::lock_guard lock(*backend_mutex_);
        engine_.reset();
        manifest_ = std::make_unique<ModelManifest>(m);
        backend_ = std::move(loaded->opened.backend);
        backend_choice_ = loaded->opened.choice;
    }
    QSettings st;
    st.setValue("model/package", QString::fromStdString(m.root.string()));
    st.setValue("model/backend", QString::fromStdString(backend_choice_->key()));
    set_model_pills();
    log("model " + describe(m));
    log("device " + device_->text());
    if (had_frames) start_engine(frames_, current_);
    else if (!pending_source_) log("drop frames — the network runs once each, then the grade is local");
    if (pending_source_) {
        const auto [path, folder] = *pending_source_;
        pending_source_.reset();
        QTimer::singleShot(0, this, [this, path, folder] { open_source(path, folder); });
    }
    sync_ui();
    refresh_enabled();
    if (done) done(true, {});
}

const Fields* MainWindow::frame_fields() const { return current_frame_ ? &current_frame_->fields : nullptr; }

void MainWindow::set_model_pills() {
    const bool on = backend_ != nullptr && manifest_ != nullptr;
    ckpt_->setText(on ? QString::fromStdString(manifest_->name) : QStringLiteral("no model"));
    if (on) {
        const auto info = backend_->info();
        device_->setText(QString::fromStdString(device_pill(info)));
        device_->setToolTip(QString::fromStdString(backend_choice_ ? backend_choice_->label() : std::string()) + " " +
                            QString::fromStdString(info.version) + (info.detail.empty() ? "" : " (" + QString::fromStdString(info.detail) + ")"));
    } else {
        device_->setText("—");
    }
    if (auto* st = findChild<QLabel*>("modelCardState")) {
        st->setText(on ? QStringLiteral("Verified \u00b7 ") + QString::fromStdString(backend_choice_ ? backend_choice_->label() : "")
                       : QStringLiteral("File > Model packages\u2026"));
        st->setProperty("state", on ? "ok" : "");
        st->style()->unpolish(st);
        st->style()->polish(st);
    }
    lamp_->setProperty("state", on ? "on" : "off");
    lamp_->style()->unpolish(lamp_);
    lamp_->style()->polish(lamp_);
}

void MainWindow::boot() {
    rescan_models();
    QSettings st;
    if (!st.value("firstRun/done", false).toBool()) {
        open_first_run();
        return;
    }
    const std::filesystem::path last = st.value("model/package").toString().toStdString();
    const auto choice = BackendChoice::from_key(st.value("model/backend").toString().toStdString());
    std::error_code ec;
    if (!last.empty() && std::filesystem::is_regular_file(last / "manifest.json", ec)) {
        use_model(last, choice);
        return;
    }
    if (const auto i = catalog_.pick()) {
        use_model(catalog_.entries[*i].package, choice);
        return;
    }
    ckpt_->setText("no model package found");
    log("no model loaded: no model package found in " + QString::number(catalog_.roots.size()) +
        " folder(s); File > Model packages… adds one");
}

void MainWindow::open_source(const QString& preset, bool folder) {
#if defined(RUDRA_APP_VIEWER) && defined(RUDRA_HAVE_STILL_DECODE)
    if (!backend_ && loading_model_ && !preset.isEmpty()) {
        pending_source_ = {preset, folder};   // opened when the model is in
        return;
    }
    if (!backend_) {
        log("Open a model package first.");
        return;
    }
    QString path = preset;
    if (path.isEmpty())
        path = folder ? QFileDialog::getExistingDirectory(this, "Open a folder of SDR frames")
                      : QFileDialog::getOpenFileName(this, "Open an SDR still or movie", {},
                                                     "Stills and movies (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp "
                                                     "*.mov *.mp4 *.mxf *.mkv *.avi *.m2ts *.ts *.webm *.m4v)");
    if (path.isEmpty()) return;
    std::vector<std::filesystem::path> frames;
    std::optional<FrameSequence> video;
    const std::string suffix = QFileInfo(path).suffix().toLower().prepend('.').toStdString();
    const bool movie = QFileInfo(path).isFile() &&
                       std::find(video_suffixes().begin(), video_suffixes().end(), suffix) != video_suffixes().end();
    if (QFileInfo(path).isDir() || movie) {
        // A folder of frames, or a movie read a frame at a time as the Studio reads it.
        auto seq = open_sequence(path.toStdString());
        if (!seq) {
            log(QString::fromStdString(seq.error().message));
            return;
        }
        frames = seq->frames;
        if (seq->kind == "video") video = std::move(*seq);
    } else {
        frames = {std::filesystem::path(path.toStdString())};
    }
    remember_source(path);
    shot_video_ = std::move(video);
    start_engine(std::move(frames));
    if (shot_video_)
        log(QStringLiteral("opened %1: %2 frames at %3 fps").arg(shot_title()).arg(frame_count()).arg(*shot_video_->fps, 0, 'g', 6));
#else
    (void)preset;
    (void)folder;
    QMessageBox::information(this, "RUDRA", "This build has no viewer or no still decoder (Qt 6.6 with Shader "
                                            "Tools, and RUDRA_WITH_OPENCV).");
#endif
}

void MainWindow::close_frames() {
    play_.stop();
    engine_.reset();
    ++engine_gen_;
    frames_.clear();
    shot_video_.reset();
    current_ = 0;
    frame_info_.clear();
    current_frame_.reset();
    measure_.reset();
    ++stats_gen_;
    clear_scopes();
    clip_bar_->hide();
    scrub_->set_count(0);
    refresh_library();
    probe_pixel(std::nullopt);
    update_pipe();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->clear_frame();
#endif
    refresh_enabled();
    sync_ui();
    show_status();
}

void MainWindow::start_engine(std::vector<std::filesystem::path> frames, int at) {
    play_.stop();
    engine_.reset();   // joins the old worker before the new one starts
    frames_ = std::move(frames);
    auto* backend = backend_.get();
    const ModelConstants model{manifest_->log_scale, manifest_->max_hdr, manifest_->corpus_ev};
    auto files = frames_;
    source_sizes_ = std::make_shared<SourceSizes>();
    source_sizes_->size.assign(files.size(), {0, 0});
    engine_ = std::make_unique<FrameEngine>(
        [files, sizes = source_sizes_, side = preview_max_side_](int i) -> Result<SdrImage> {
#ifdef RUDRA_HAVE_STILL_DECODE
            auto file = ensure_frame_file(files[std::size_t(i)]);   // a movie's frame is extracted on first use
            if (!file) return file.error();
            auto d = decode_sdr_file(*file);
            if (!d) return d.error();
            {
                std::lock_guard lock(sizes->mu);
                sizes->size[std::size_t(i)] = {d->rgb.width(), d->rgb.height()};
            }
            // The Studio's preview: 1600 on the long side (ui/server.py _fit);
            // masters decode again at full size.
            return fit_max_side(d->rgb, side);
#else
            // A build without media's still decode (no OpenCV) opens no
            // frames; the viewer and the rest of the shell still work.
            (void)files;
            (void)i;
            return make_error(ErrorCode::Unsupported, "This build has no still decode (RUDRA_WITH_OPENCV=OFF).");
#endif
        },
        [backend, mu = backend_mutex_](const SdrImage& sdr) {
            // One untiled pass, as the Studio's preview; the master job shares the backend.
            std::lock_guard lock(*mu);
            return infer_frame(*backend, sdr, TileConfig{0, 0});
        });
    QPointer<MainWindow> self(this);
    // A frame the engine before this one finished (another model's, or
    // frames since closed) is not shown.
    const int gen = ++engine_gen_;
    engine_->on_ready([self, model, gen](const ReadyFrame& f) {
        QMetaObject::invokeMethod(qApp, [self, f, model, gen] {
            if (self && self->engine_gen_ == gen) self->frame_ready(f, model);
        });
    });
    engine_->set_sequence(int(frames_.size()));
    scrub_->set_count(int(frames_.size()));
    first_thumb_ = QImage();
    refresh_library();
    refresh_enabled();
    sync_ui();
    step_to(at);
}

void MainWindow::step_to(int i) {
    if (!engine_ || frames_.empty()) return;
    const int n = int(frames_.size());
    current_ = ((i % n) + n) % n;
    waiting_ = true;
    engine_->show(current_);
    sync_ui();
}

void MainWindow::toggle_play() {
    if (play_.isActive()) {
        play_.stop();
        btn_play_->set_glyph(IconButton::Glyph::Play);
        return;
    }
    if (frames_.size() < 2) return;
    btn_play_->set_glyph(IconButton::Glyph::Pause);
    play_.start();
}

void MainWindow::frame_ready(const ReadyFrame& f, const ModelConstants& model) {
    if (f.index != current_) return;   // the engine already dropped stale ones; this is the UI's own check
    waiting_ = false;
    const QString name = QString::fromStdString(frame_name(std::size_t(f.index)));
    if (f.error) {
        log(name + ": " + QString::fromStdString(f.error->message));
        return;
    }
    FrameHeader header;
    header.resolution = QStringLiteral("%1x%2").arg(f.sdr->width()).arg(f.sdr->height()).toStdString();
    header.source_resolution = header.resolution;
    if (source_sizes_) {
        std::lock_guard lock(source_sizes_->mu);
        const auto [sw, sh] = source_sizes_->size[std::size_t(f.index)];
        if (sw > 0) header.source_resolution = QStringLiteral("%1x%2").arg(sw).arg(sh).toStdString();
    }
    header.tiled = false;   // the preview is one untiled pass, as the Studio's
    if (!f.from_cache) header.elapsed_s = f.infer_ms / 1000.0;
    else if (current_frame_ && current_frame_->header.elapsed_s) header.elapsed_s = current_frame_->header.elapsed_s;
    present_frame(*f.sdr, f.fields->fields, f.fields->scalars, model, header);
    const auto st = engine_->stats();
    frame_info_ = QStringLiteral("%1  %2/%3  ·  %4")
                      .arg(name)
                      .arg(f.index + 1)
                      .arg(frames_.size())
                      .arg(f.from_cache ? QStringLiteral("cached")
                                        : QStringLiteral("decode %1 ms, infer %2 ms")
                                              .arg(f.decode_ms, 0, 'f', 0)
                                              .arg(f.infer_ms, 0, 'f', 0));
    frame_info_ += QStringLiteral("  ·  %1 in cache").arg(st.cached);
    show_status();
}

void MainWindow::show_status() {
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const ViewerStatus s = viewer_->status();
        zoom_val_->setText(QStringLiteral("%1%").arg(s.zoom_percent));
        zoom_seg_->set_on(viewer_->viewport().scale <= 0.0 ? "fit" : (s.zoom_percent == 100 ? "actual" : ""));
        const bool hdr = s.target.path != OutputPath::SdrPqSimulation;
        findChild<QLabel*>("statusTime")->setText(QString::fromStdString(s.backend) + " · " +
                                                  QString::fromStdString(s.swapchain) +
                                                  (hdr ? QStringLiteral(" · peak %1 nits").arg(std::lround(s.target.peak_nits))
                                                       : QString()));
    }
#endif
    src_info_->setText(frame_info_.isEmpty() ? QStringLiteral("—") : frame_info_);
    update_badges();
}

// The page's timecode(): non-drop, from 01:00:00:00.
static QString timecode(int frame, double fps) {
    const int rate = std::max(1, int(std::floor(fps + 0.5)));
    const long long f = frame + static_cast<long long>(rate) * 3600;
    auto pad = [](long long v) { return QString::number(v).rightJustified(2, '0'); };
    return pad((f / (rate * 3600LL)) % 24) + ":" + pad((f / (rate * 60LL)) % 60) + ":" + pad((f / rate) % 60) + ":" +
           pad(f % rate);
}

void MainWindow::sync_ui() {
    const auto& g = session_.grade;
    mode_seg_->set_on(QString::fromStdString(g.mode));
    {
        const QSignalBlocker b1(strength_), b2(peak_);
        strength_->setValue(int(std::lround(g.strength * 20.0)));
        peak_->setValue(int(std::lround(session_.peak_ev * 2.0)));
    }
    strength_val_->setText(QString::number(g.strength, 'f', 2));
    peak_val_->setText(QLocale(QLocale::English).toString(qlonglong(std::floor(session_.display_nits() + 0.5))));
    preserve_->set_on(g.preserve);
    preserve_->set_hint(g.preserve ? "do-no-harm" : "raw prediction");
    anchor_->set_on(session_.anchor);
    anchor_->set_hint(session_.anchor ? "conform" : "raw ITM level");
    carry_chroma_->set_on(session_.carry_chroma);
    carry_chroma_->set_hint(session_.carry_chroma ? "below the clip" : "per-channel");
    const bool aces = session_.container == "aces";
    container_field_->setText(aces ? "OpenEXR — ACES 2065-1" : "OpenEXR — linear Rec.2020");
    primaries_field_->setText(aces ? "AP0 (ST 2065-4)" : "Rec.2020");
    region_count_->setText(QString::number(g.regions.size()));
    regions_->sync();
    view_mode_->set_on(session_.wipe ? "#wipeBtn" : QString::fromStdString(session_.show));
    view_layer_->set_on(QString::number(session_.view_layer));
    guide_btn_->setChecked(guides_on_);
    // Window
    rail_left_->setVisible(session_.rail_left);
    rail_right_->setVisible(session_.rail_right);
    scopes_->setVisible(session_.scopes_open && workspace_ != "simple");
    i_media_->set_on(session_.rail_left);
    i_inspector_->set_on(session_.rail_right);
    i_scopes_->set_on(session_.scopes_open && workspace_ != "simple");
    ws_->set_on(workspace_);
    tabs_->set_on(tab_);
    const int page = tab_ == "grade" ? 1 : tab_ == "deliver" ? 2 : 0;
    for (int i = 0; i < int(pages_.size()); ++i) pages_[std::size_t(i)]->setVisible(i == page);
    for (auto* n : notes_) n->setVisible(workspace_ != "simple");
    // Frames (drawFrames)
    const int n = int(frames_.size());
    {
        auto* title = findChild<QLabel*>("shotTitle");
        auto* sub = findChild<QLabel*>("shotSub");
        if (n == 0) {
            title->setText("No shot open");
            sub->setText("Drop frames, or open a folder");
        } else {
            title->setText(shot_title());
            QString res;
            if (current_frame_ && current_frame_->header.source_resolution)
                res = QString::fromStdString(*current_frame_->header.source_resolution).replace('x', QChar(0xd7)) +
                      QStringLiteral(" \u00b7 ");
            const double fps = shot_video_ ? shot_video_->fps.value_or(24.0) : 24.0;
            sub->setText(res + QStringLiteral("%1 frame%2 \u00b7 %3 fps \u00b7 Rec.709")
                                   .arg(n)
                                   .arg(n == 1 ? "" : "s")
                                   .arg(QString::number(fps, 'g', 5)));
        }
    }
    shot_count_->setText(QString::number(n));
    frames_empty_->setVisible(n == 0);
    if (auto* dz = findChild<QWidget*>("dropzone")) dz->setVisible(n == 0);   // shell.js: tucked once frames open
    tc_->setText(timecode(n ? current_ : 0, 24.0));
    scrub_->set_position(n > 1 ? double(current_) / double(n - 1) : 0.0);
    for (auto* b : {btn_prev_, btn_play_, btn_next_}) b->setEnabled(n > 1);
    const bool ready = n > 0 && backend_ != nullptr;
    btn_master_->setEnabled((ready && pending_reason("master").isEmpty()) || mastering());
    btn_reprocess_->setEnabled(ready);
    if (viewer_stack_->count() > 1) viewer_stack_->setCurrentIndex(n > 0 ? 1 : 0);
}

void MainWindow::keyPressEvent(QKeyEvent* e) {
    const bool modified = e->modifiers() & (Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier);
    const bool shift = e->modifiers() & Qt::ShiftModifier;
    std::string key;
    switch (e->key()) {
        case Qt::Key_Left: key = "ArrowLeft"; break;
        case Qt::Key_Right: key = "ArrowRight"; break;
        case Qt::Key_Escape: key = "Escape"; break;
        case Qt::Key_B: key = "b"; break;
        default: break;
    }
    if (key.empty() || modified) {
        QMainWindow::keyPressEvent(e);
        return;
    }
    session_.key_down(key, shift, false, e->isAutoRepeat());
}

void MainWindow::keyReleaseEvent(QKeyEvent* e) {
    if (e->key() == Qt::Key_B && !e->isAutoRepeat()) session_.key_up("b");
    QMainWindow::keyReleaseEvent(e);
}

void MainWindow::set_workspace(const QString& mode) {
    workspace_ = mode == "simple" ? "simple" : "full";
    sync_ui();
}

void MainWindow::show_tab(const QString& tab) {
    tab_ = tab;
    sync_ui();
}

void MainWindow::show_scopes(const ScopeData& s, std::optional<double> maxcll,
                             const std::vector<std::uint8_t>& vector_rgba) {
    wave_->set_data(s, maxcll);
    hist_->set_data(s, maxcll);
    vector_->set_image(vector_rgba);
}

void MainWindow::clear_scopes() {
    wave_->clear();
    hist_->clear();
    vector_->clear();
}

void MainWindow::present_frame(SdrImage sdr, Fields fields, FrameScalars scalars, ModelConstants model,
                               FrameHeader header) {
    auto cur = std::make_shared<Current>();
    cur->baseline = std::make_shared<const NetworkLinearImage>(
        corrected_baseline(sdr, model.corpus_ev, scalars.curve_params));
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        viewer_->set_frame({sdr, fields, scalars, model});
        viewer_->set_composite(session_.composite_params());
    }
#endif
    cur->sdr = std::move(sdr);
    cur->fields = std::move(fields);
    cur->scalars = std::move(scalars);
    cur->model = model;
    cur->header = std::move(header);
    current_frame_ = std::move(cur);
    note_frame_in_strip(current_);
    run_stats();   // adopt(): computeStats at once for a new frame
}

void MainWindow::schedule_stats() {
    if (current_frame_) stats_timer_.start();
}

void MainWindow::run_stats() {
    if (!current_frame_) return;
    if (stats_running_) {   // the page's statsPending: once more when this one is done
        stats_again_ = true;
        return;
    }
    stats_running_ = true;
    const int gen = ++stats_gen_;
    auto frame = current_frame_;
    const CompositeParams params = session_.composite_params();
    QPointer<MainWindow> self(this);
    // Joined, not detached: a detached measurement outlived its window and, at
    // exit, the libraries it uses (a destroyed mutex aborts on macOS).
    if (stats_worker_.joinable()) stats_worker_.join();   // done: stats_running_ was cleared after it posted
    stats_worker_ = std::thread([self, frame, params, gen] {
        auto m = std::make_shared<const FrameMeasure>(
            measure_frame(frame->sdr, frame->fields, frame->scalars, frame->model, params, frame->baseline.get()));
        QMetaObject::invokeMethod(qApp, [self, m, gen] {
            if (!self) return;
            self->stats_running_ = false;
            if (gen == self->stats_gen_) self->apply_measure(m);
            if (self->stats_again_) {
                self->stats_again_ = false;
                self->run_stats();
            }
        });
    });
}

void MainWindow::measure_now() {
    if (!current_frame_) return;
    const auto& f = *current_frame_;
    ++stats_gen_;   // anything in flight is now stale
    apply_measure(std::make_shared<const FrameMeasure>(
        measure_frame(f.sdr, f.fields, f.scalars, f.model, session_.composite_params(), f.baseline.get())));
}

void MainWindow::fill_rows(QWidget* ms, const std::vector<MetricRow>& rows) {
    auto* v = ms->layout();
    while (QLayoutItem* it = v->takeAt(0)) {
        delete it->widget();
        delete it;
    }
    for (const auto& r : rows) {
        auto* row = new QWidget(ms);
        row->setProperty("role", "ro");
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(0, 2, 0, 2);
        h->setSpacing(3);
        auto* k = new QLabel(QString::fromStdString(r.k), row);
        k->setProperty("role", "key");
        auto* val = new QLabel(QString::fromStdString(r.v), row);
        val->setProperty("role", "value");
        if (r.warn) val->setProperty("state", "warn");
        val->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        auto* u = new QLabel(QString::fromStdString(r.u), row);
        u->setProperty("role", "unit");
        h->addWidget(k);
        h->addStretch(1);
        h->addWidget(val);
        h->addWidget(u);
        v->addWidget(row);
    }
}

void MainWindow::apply_measure(std::shared_ptr<const FrameMeasure> m) {
    measure_ = std::move(m);
    const auto& f = *measure_;
    const MetricsText t = metrics_text(f.frame_metrics(), current_frame_ ? std::optional(current_frame_->header)
                                                                          : std::nullopt,
                                       f.coverage.clipped_pct);
    fill_rows(findChild<QWidget*>("measA"), t.a);
    fill_rows(findChild<QWidget*>("measB"), t.b);
    findChild<QLabel*>("statusMask")->setText(QString::fromStdString(t.status_mask));
    findChild<QLabel*>("statusTime")->setText(QString::fromStdString(t.status_time));
    src_info_->setText(QString::fromStdString(t.src_info));
    clip_bar_->set(f.coverage.clipped_pct, f.coverage.highlight_pct);
    clip_bar_->show();
    show_scopes(f.measured.scopes, f.measured.metrics.maxcll, f.vector_rgba);
    // The clipping lane: gold where the SDR clipped, violet where the network
    // found crushed shadows (its shadow mask over 1 % of the frame).
    if (!frames_.empty()) {
        const auto lane = f.coverage.clipped_pct > 0.01 ? ScrubBar::Lane::Highlights
                          : f.coverage.shadow_pct > 1.0 ? ScrubBar::Lane::Shadows
                                                        : ScrubBar::Lane::None;
        scrub_->set_lane(current_, lane);
        int hi = 0, sh = 0;
        for (int i = 0; i < scrub_->count(); ++i) {
            hi += scrub_->lane(i) == ScrubBar::Lane::Highlights;
            sh += scrub_->lane(i) == ScrubBar::Lane::Shadows;
        }
        findChild<QLabel*>("colHighlights")->setText(QString::number(hi));
        findChild<QLabel*>("colShadows")->setText(QString::number(sh));
    }
    update_pipe();
    update_badges();
    refresh_enabled();
}

void MainWindow::update_pipe() {
    std::optional<double> maxcll;
    if (measure_) maxcll = measure_->measured.metrics.maxcll;
    const PipeText t = pipe_text(session_.container, session_.display_nits(), maxcll,
                                 current_frame_ ? std::optional(current_frame_->header) : std::nullopt);
    findChild<QLabel*>("pipeIn")->setText(QString::fromStdString(t.in));
    findChild<QLabel*>("pipeWorking")->setText(QString::fromStdString(t.working));
    view_transform_->setText(QString::fromStdString(t.view));
    findChild<QLabel*>("pipeMaster")->setText(QString::fromStdString(t.master));
    auto* warn = findChild<QLabel*>("pipeWarn");
    warn->setText(QString::fromStdString(t.warn));
    warn->setVisible(t.warn_shown);
}

void MainWindow::probe_pixel(std::optional<std::pair<double, double>> px, QPoint global) {
    std::optional<ProbeInput> p;
    if (px && measure_) p = measure_->probe_at(px->first, px->second);
    const ProbePanelText t = probe_panel(p);
    auto set = [this](const char* id, const std::string& text, const char* state) {
        auto* l = findChild<QLabel*>(id);
        l->setText(QString::fromStdString(text));
        if (l->property("state").toString() != state) {
            l->setProperty("state", state);
            l->style()->unpolish(l);
            l->style()->polish(l);
        }
    };
    set("probeXY", t.xy, "");
    set("probeNits", t.nits, t.idle ? "idle" : "");
    findChild<QLabel*>("probeNitsUnit")->setVisible(!t.idle);
    set("probeDelta", t.delta, t.delta_class.find("idle") != std::string::npos ? "idle" : "");
    set("probeSrc", t.src, t.src_class.c_str());
    set("probeBase", t.base, "");
    set("probeModel", t.model, t.model_class.c_str());
    set("probeMask", t.mask, "");
    // The floating box.
    if (!p) {
        probe_box_->hide();
        return;
    }
    auto* grid = static_cast<QGridLayout*>(probe_box_->layout());
    while (QLayoutItem* it = grid->takeAt(0)) {
        delete it->widget();
        delete it;
    }
    int r = 0;
    for (const auto& row : rudra::probe_box(*p)) {
        auto* k = new QLabel(QString::fromStdString(row.k), probe_box_);
        k->setProperty("role", "key");
        auto* v = new QLabel(QString::fromStdString(row.v), probe_box_);
        v->setProperty("role", "value");
        if (!row.cls.empty()) v->setProperty("state", QString::fromStdString(row.cls));
        grid->addWidget(k, r, 0);
        grid->addWidget(v, r, 1);
        ++r;
    }
    probe_box_->adjustSize();
    // showProbe's placing: 16 px right of and below the cursor, flipped to the
    // other side at the viewer's edges, at least 4 px in.
    QWidget* area = viewer_stack_;
    const QPoint origin = area->mapToGlobal(QPoint(0, 0));
    const int bw = std::max(196, probe_box_->width()), bh = probe_box_->height();
    int lx = global.x() - origin.x() + 16, ly = global.y() - origin.y() + 16;
    if (lx + bw > area->width()) lx = global.x() - origin.x() - bw - 16;
    if (ly + bh > area->height()) ly = global.y() - origin.y() - bh - 16;
    probe_box_->move(origin + QPoint(std::max(4, lx), std::max(4, ly)));
    probe_box_->show();
}

MasterRequest MainWindow::master_request() const {
    // The settings of the moment: params(), the Deliver checks, the container.
    MasterRequest q;
    q.checkpoint = manifest_ ? manifest_->name : std::string();
    q.preserve_outside = session_.grade.preserve;
    q.recovery_mode = session_.grade.mode;
    q.strength = session_.grade.strength;
    for (const auto& r : session_.grade.regions) q.regions.push_back({r.label, r.low_nits, r.high_nits, r.ev});
    q.anchor = session_.anchor;
    q.carry_chroma = session_.carry_chroma;
    q.container = session_.container;
    return q;
}

const SdrImage* MainWindow::frame_sdr() const { return current_frame_ ? &current_frame_->sdr : nullptr; }

void MainWindow::master(PrepareMasterFrame prepare, std::size_t count) {
    auto* status = findChild<QLabel*>("renderStatus");
    auto say = [status](const QString& t) {
        status->setText(t);
        status->setVisible(!t.isEmpty());
    };
    const bool injected = bool(prepare);
    if (!injected && (!backend_ || frames_.empty())) return;   // !state.live || !current()
    if (mastering()) return;                                    // state.busy
    auto* dir = findChild<QLineEdit*>("renderDir");
    const QString folder = dir->text().trimmed();
    if (folder.isEmpty()) {
        say("Choose a render folder first.");
        dir->setFocus();
        return;
    }
    const bool sequence = findChild<QComboBox*>("renderMode")->currentData().toString() == "sequence";
    std::vector<std::filesystem::path> sources;
    if (!injected) sources = sequence ? frames_ : std::vector<std::filesystem::path>{frames_[std::size_t(current_)]};
    const std::size_t n = injected ? (sequence ? count : 1) : sources.size();
    RenderPlan plan;
    plan.render_dir = folder.toStdString();
    plan.render_name = findChild<QLineEdit*>("renderName")->text().trimmed().toStdString();
    plan.render_count = int(n);
    plan.frame_start = findChild<QSpinBox*>("renderStart")->value();
    plan.sequence = sequence;
    auto targets = master_targets(plan);
    if (!targets) {
        say(QStringLiteral("Stopped after 0 / %1: %2").arg(n).arg(QString::fromStdString(targets.error().message)));
        log("Render stopped: " + QString::fromStdString(targets.error().message));
        return;
    }
    log("Render destination: " + QString::fromStdString(targets->front().string()) +
        (targets->size() > 1 ? " \u2026 " + QString::fromStdString(targets->back().string()) : QString()));

    const MasterRequest q = master_request();
    const ModelConstants model = manifest_ ? ModelConstants{manifest_->log_scale, manifest_->max_hdr, manifest_->corpus_ev}
                                           : (current_frame_ ? current_frame_->model : ModelConstants{16.0f, 4.0f, -1.0f});
    if (!prepare) {
#ifdef RUDRA_HAVE_STILL_DECODE
        prepare = [sources, backend = backend_.get(), mu = backend_mutex_](std::size_t i) -> Result<MasterFrame> {
            auto file = ensure_frame_file(sources[i]);
            if (!file) return file.error();
            auto d = decode_sdr_file(*file);
            if (!d) return d.error();
            std::lock_guard lock(*mu);
            auto fr = infer_frame(*backend, d->rgb, TileConfig{0, 0});   // full resolution, untiled
            if (!fr) return fr.error();
            return MasterFrame{std::move(d->rgb), d->bits, std::move(fr->fields), std::move(fr->scalars)};
        };
#else
        say("This build has no still decode (RUDRA_WITH_OPENCV=OFF).");
        return;
#endif
    }
    play_.stop();   // stopPlay()
    btn_master_->setText("Rendering\u2026");
    btn_master_->setToolTip("Click to stop after the frame being written");
    QPointer<MainWindow> self(this);
    master_job_ = std::make_unique<MasterJob>(model, q, *targets, std::move(prepare));
    master_job_->start(
        [self, n](std::size_t i) {
            QMetaObject::invokeMethod(qApp, [self, i, n] {
                if (self) self->findChild<QLabel*>("renderStatus")->setText(QStringLiteral("Rendering %1 / %2").arg(i + 1).arg(n));
                if (self) self->findChild<QLabel*>("renderStatus")->show();
            });
        },
        [self](const MasterProgress& p) {
            const QString line = "Saved " + QString::fromStdString(p.last.exr.string()) +
                                 QStringLiteral(" \u00b7 %1x%2").arg(p.last.width).arg(p.last.height);
            QMetaObject::invokeMethod(qApp, [self, line] {
                if (self) self->log(line);
            });
        },
        [self, folder](const MasterOutcome& o) {
            QMetaObject::invokeMethod(qApp, [self, o, folder] {
                if (self) self->master_finished(o, folder);
            });
        });
    refresh_enabled();
    sync_ui();   // the button stays live: a click stops the render
}

void MainWindow::master_finished(const MasterOutcome& o, const QString& folder) {
    auto* status = findChild<QLabel*>("renderStatus");
    if (o.error || o.cancelled) {
        const QString why = o.error ? QString::fromStdString(o.error->message) : QStringLiteral("cancelled");
        status->setText(QStringLiteral("Stopped after %1 / %2: %3").arg(o.completed).arg(o.total).arg(why));
        log("Render stopped: " + why);
    } else {
        status->setText(QStringLiteral("Rendered %1 frame(s) to %2").arg(o.completed).arg(folder));
    }
    status->show();
    delivered_ += int(o.completed);
    findChild<QLabel*>("colDelivered")->setText(QString::number(delivered_));
    btn_master_->setText("Master EXR");   // busy(false)
    btn_master_->setToolTip({});
    refresh_enabled();
    sync_ui();
}

// ---------------------------------------------------------------------------
// Step 11: copies, drop to open, recent shots, settings
// ---------------------------------------------------------------------------

void MainWindow::copy_text(const QString& label, const std::string& text) {
    QGuiApplication::clipboard()->setText(QString::fromStdString(text));
    log("copied " + label + " to the clipboard");
}

std::string MainWindow::delivery_text() const {
    DeliveryRecord d;
    if (current_frame_) {
        // state.header: what came back with the frame on screen.
        if (manifest_) d.checkpoint = manifest_->name;
        d.resolution = current_frame_->header.resolution;
    }
    if (!frames_.empty()) d.frame = frame_name(std::size_t(current_));
    d.aces = session_.container == "aces";
    d.mode = session_.grade.mode;
    d.strength = session_.grade.strength;
    d.preserve = session_.grade.preserve;
    for (const auto& r : session_.grade.regions) d.regions.push_back({r.label, r.low_nits, r.high_nits, r.ev});
    if (measure_) d.metrics = std::pair{measure_->measured.metrics, measure_->compose_ms};
    return delivery_value(d).stringify();
}

void MainWindow::add_files(const std::vector<std::filesystem::path>& files) {
    if (files.empty()) return;
    if (!backend_) {
        log("Open a model package first.");
        return;
    }
    auto all = frames_;
    const std::size_t start = all.size();
    all.insert(all.end(), files.begin(), files.end());
    log(QStringLiteral("added %1 frame%2").arg(files.size()).arg(files.size() == 1 ? "" : "s"));
    for (const auto& f : files) remember_source(QString::fromStdString(f.string()));
    // select(state.frames.length === files.length ? 0 : start)
    start_engine(std::move(all), all.size() == files.size() ? 0 : int(start));
}

void MainWindow::open_paths(const QStringList& paths) {
    std::vector<std::filesystem::path> files;
    for (const auto& p : paths) {
        const QFileInfo fi(p);
        if (fi.isDir()) {
            if (QFileInfo(QDir(p).filePath("manifest.json")).isFile()) use_model(p.toStdString(), backend_choice_);
            else open_source(p, true);
            return;   // one folder is one shot, or one model
        }
        if (fi.isFile()) {
            const std::string suf = fi.suffix().toLower().prepend('.').toStdString();
            if (std::find(video_suffixes().begin(), video_suffixes().end(), suf) != video_suffixes().end()) {
                open_source(p);   // a movie is a shot of its own
                return;
            }
            files.emplace_back(p.toStdString());
        }
    }
    add_files(files);
}

void MainWindow::set_drop_hot(bool hot) {
    if (auto* dz = findChild<QWidget*>("dropzone")) {
        dz->setProperty("state", hot ? "hot" : "");
        dz->style()->unpolish(dz);
        dz->style()->polish(dz);
        for (auto* c : dz->findChildren<QWidget*>()) {
            c->style()->unpolish(c);
            c->style()->polish(c);
        }
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (!e->mimeData()->hasUrls()) return;
    e->acceptProposedAction();
    set_drop_hot(true);
}

void MainWindow::dragLeaveEvent(QDragLeaveEvent*) { set_drop_hot(false); }

void MainWindow::dropEvent(QDropEvent* e) {
    set_drop_hot(false);
    QStringList paths;
    for (const auto& u : e->mimeData()->urls())
        if (u.isLocalFile()) paths << u.toLocalFile();
    if (paths.isEmpty()) return;
    e->acceptProposedAction();
    open_paths(paths);
}

namespace {
constexpr int kRecentMax = 10;
}

QStringList MainWindow::recent_sources() const { return QSettings().value("recent/sources").toStringList(); }

void MainWindow::remember_source(const QString& path) {
    QSettings st;
    QStringList r = st.value("recent/sources").toStringList();
    const QString p = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    r.removeAll(p);
    r.prepend(p);
    while (r.size() > kRecentMax) r.removeLast();
    st.setValue("recent/sources", r);
}

void MainWindow::fill_recent() {
    if (!recent_menu_) return;
    for (QAction* a : recent_menu_->actions())
        if (a->objectName().startsWith("recent:")) {
            recent_menu_->removeAction(a);
            a->deleteLater();
        }
    QAction* clear = action("recent-clear");
    const QStringList r = recent_sources();
    int i = 0;
    for (const auto& p : r) {
        const QFileInfo fi(p);
        auto* a = new QAction(QStringLiteral("%1  ·  %2").arg(fi.fileName(), fi.absolutePath()), recent_menu_);
        a->setObjectName(QStringLiteral("recent:%1").arg(i++));
        a->setData(p);
        a->setEnabled(fi.exists() && backend_ != nullptr);
        connect(a, &QAction::triggered, this, [this, p] {
            if (QFileInfo(p).isDir()) open_source(p, true);
            else add_files({std::filesystem::path(p.toStdString())});
        });
        recent_menu_->insertAction(clear, a);
    }
    if (clear) clear->setEnabled(!r.isEmpty());
}

void MainWindow::save_settings() const {
    QSettings st;
    st.setValue("window/geometry", saveGeometry());
    st.setValue("window/workspace", workspace_);
    st.setValue("window/tab", tab_);
    st.setValue("window/railLeft", session_.rail_left);
    st.setValue("window/railRight", session_.rail_right);
    st.setValue("window/scopes", session_.scopes_open);
    st.setValue("deliver/container", QString::fromStdString(session_.container));
    if (auto* e = findChild<QLineEdit*>("renderDir")) st.setValue("deliver/renderDir", e->text());
    if (auto* e = findChild<QLineEdit*>("renderName")) st.setValue("deliver/renderName", e->text());
    if (auto* c = findChild<QComboBox*>("renderMode")) st.setValue("deliver/renderMode", c->currentIndex());
    if (auto* sb = findChild<QSpinBox*>("renderStart")) st.setValue("deliver/renderStart", sb->value());
    save_queues();
}

void MainWindow::save_queues() const {
    QStringList q;
    for (const auto& e : queues_->entries())
        q << QString::fromStdString(e.queue.string()) + "\t" + QString::fromStdString(e.title);
    QSettings().setValue("queue/entries", q);
}

void MainWindow::restore_settings() {
    QSettings st;
    if (st.contains("window/geometry")) restoreGeometry(st.value("window/geometry").toByteArray());
    if (st.contains("window/workspace")) set_workspace(st.value("window/workspace").toString());
    if (st.contains("window/tab")) show_tab(st.value("window/tab").toString());
    const bool l = st.value("window/railLeft", session_.rail_left).toBool(),
               r = st.value("window/railRight", session_.rail_right).toBool(),
               sc = st.value("window/scopes", session_.scopes_open).toBool();
    if (l != session_.rail_left) run("rail-left");
    if (r != session_.rail_right) run("rail-right");
    if (sc != session_.scopes_open) run("scopes");
    const QString c = st.value("deliver/container").toString();
    if (c == "aces" || c == "linear") session_.set_container(c.toStdString());
    if (auto* e = findChild<QLineEdit*>("renderDir"); e && st.contains("deliver/renderDir"))
        e->setText(st.value("deliver/renderDir").toString());
    if (auto* e = findChild<QLineEdit*>("renderName"); e && st.contains("deliver/renderName"))
        e->setText(st.value("deliver/renderName").toString());
    if (auto* cb = findChild<QComboBox*>("renderMode"); cb && st.contains("deliver/renderMode"))
        cb->setCurrentIndex(st.value("deliver/renderMode").toInt());
    if (auto* sb = findChild<QSpinBox*>("renderStart"); sb && st.contains("deliver/renderStart"))
        sb->setValue(st.value("deliver/renderStart").toInt());
    {   // the exports queued before: listed again, their state read from their files
        std::vector<QueueEntry> entries;
        for (const QString& line : st.value("queue/entries").toStringList()) {
            const auto parts = line.split('\t');
            if (parts.size() == 2 && QFileInfo(parts[0]).isFile())
                entries.push_back({parts[0].toStdString(), parts[1].toStdString()});
        }
        queues_->set_entries(std::move(entries));
    }
    sync_checks();
    sync_ui();
}

void MainWindow::closeEvent(QCloseEvent* e) {
    save_settings();
    QMainWindow::closeEvent(e);
}


// ---------------------------------------------------------------------------
// The Pro chrome
// ---------------------------------------------------------------------------

namespace {

// A small picture of an SDR frame: its display-encoded codes as they are.
QImage sdr_thumb(const SdrImage& sdr, int height) {
    const auto& b = sdr.buffer();
    if (b.width() <= 0 || b.height() <= 0) return {};
    const int h = std::max(1, std::min(height, b.height()));
    const int w = std::max(1, int(std::lround(double(b.width()) * h / b.height())));
    QImage img(w, h, QImage::Format_RGB888);
    for (int y = 0; y < h; ++y) {
        uchar* row = img.scanLine(y);
        const int sy = std::min(b.height() - 1, int((y + 0.5) * b.height() / h));
        for (int x = 0; x < w; ++x) {
            const int sx = std::min(b.width() - 1, int((x + 0.5) * b.width() / w));
            for (int c = 0; c < 3; ++c)
                row[x * 3 + c] = uchar(std::clamp(int(std::lround(b.at(c, sy, sx) * 255.0f)), 0, 255));
        }
    }
    return img;
}

}  // namespace

void MainWindow::note_frame_in_strip(int index) {
    if (!current_frame_ || index < 0 || index >= scrub_->count()) return;
    const QImage t = sdr_thumb(current_frame_->sdr, 88);
    scrub_->set_thumb(index, t);
    if (index == 0 || first_thumb_.isNull()) first_thumb_ = t;
    if (index == 0 || scrub_->count() == 1) refresh_library();   // the shot row's picture
}

void MainWindow::refresh_library() {
    auto* shots = findChild<QWidget*>("shots");
    if (!shots) return;
    for (auto* c : shots->findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) c->deleteLater();
    auto* v = static_cast<QVBoxLayout*>(shots->layout());
    auto add_row = [&](const QString& name, const QString& sub, const QImage& thumb, bool sel, bool hdr,
                       const QString& open) {
        auto* r = new QWidget(shots);
        r->setProperty("role", "shot");
        r->setProperty("sel", sel);
        r->setAttribute(Qt::WA_StyledBackground, true);
        auto* h = new QHBoxLayout(r);
        h->setContentsMargins(8, 6, 8, 6);
        h->setSpacing(10);
        auto* pic = new QLabel(r);
        pic->setFixedSize(46, 30);
        if (!thumb.isNull())
            pic->setPixmap(QPixmap::fromImage(thumb.scaled(46, 30, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation)
                                                  .copy(0, 0, 46, 30)));
        else pic->setStyleSheet({});
        h->addWidget(pic);
        auto* text = new QWidget(r);
        auto* tv = new QVBoxLayout(text);
        tv->setContentsMargins(0, 0, 0, 0);
        tv->setSpacing(0);
        auto* n = new QLabel(name, text);
        n->setProperty("role", "shot-name");
        auto* s = new QLabel(sub, text);
        s->setProperty("role", "shot-sub");
        tv->addWidget(n);
        tv->addWidget(s);
        h->addWidget(text, 1);
        if (hdr) {
            auto* tag = new QLabel("HDR", r);
            tag->setProperty("role", "hdr-tag");
            h->addWidget(tag);
        }
        if (!open.isEmpty()) {
            r->setCursor(Qt::PointingHandCursor);
            r->setToolTip(open);
            r->installEventFilter(this);
            r->setProperty("open", open);
        }
        v->addWidget(r);
    };
    int rows = 0;
    QString current;
    if (!frames_.empty()) {
        const bool folder = frames_.size() > 1;
        const auto& first = frames_.front();
        current = QString::fromStdString(shot_video_ ? shot_video_->path.string() : (folder ? first.parent_path() : first).string());
        QString res;
        if (current_frame_ && current_frame_->header.source_resolution)
            res = QString::fromStdString(*current_frame_->header.source_resolution).replace('x', QChar(0xd7)) + " · ";
        const bool hdr = measure_ && measure_->measured.metrics.maxcll > 203.0;
        add_row(shot_title(),
                res + QStringLiteral("%1 f").arg(frames_.size()), scrub_->count() ? thumb_of_first() : QImage(), true,
                hdr, {});
        ++rows;
    }
    // Then the shots opened before (Open recent), to reopen with a click.
    for (const auto& p : recent_sources()) {
        if (rows >= 6) break;
        if (QDir::cleanPath(p) == QDir::cleanPath(current)) continue;
        const QFileInfo fi(p);
        if (!fi.exists()) continue;
        add_row(fi.fileName(), fi.isDir() ? QStringLiteral("folder") : fi.suffix().toUpper(), {}, false, false, p);
        ++rows;
    }
    shots->setVisible(rows > 0);
}

QImage MainWindow::thumb_of_first() const {
    return first_thumb_;
}

void MainWindow::update_badges() {
    auto* text = findChild<QLabel*>("hdrBadgeText");
    auto* sub = findChild<QLabel*>("hdrBadgeSub");
    auto* dot = findChild<QLabel*>("hdrDot");
    bool hdr = false;
    QString what = "display not measured";
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const ViewerStatus s = viewer_->status();
        hdr = s.target.path != OutputPath::SdrPqSimulation;
        what = hdr ? QLocale(QLocale::English).toString(qlonglong(std::lround(s.target.peak_nits))) + " nits headroom"
                   : QStringLiteral("simulated at 203 nits");
    }
#endif
    text->setText(hdr ? "HDR" : "SDR");
    sub->setText(what);
    if (dot->property("state").toString() != (hdr ? "hdr" : "sdr")) {
        dot->setProperty("state", hdr ? "hdr" : "sdr");
        dot->style()->unpolish(dot);
        dot->style()->polish(dot);
    }
    auto* cll = findChild<QLabel*>("cllBadge");
    if (measure_) {
        const double maxcll = measure_->measured.metrics.maxcll;
        const bool clipped = maxcll > session_.display_nits() + 0.5;
        cll->setText(QStringLiteral("MaxCLL ") + QLocale(QLocale::English).toString(qlonglong(std::lround(maxcll))) +
                     (clipped ? QStringLiteral(" · clipped on screen") : QString()));
    } else {
        cll->setText(QStringLiteral("MaxCLL —"));
    }
}

bool MainWindow::eventFilter(QObject* o, QEvent* e) {
    if (e->type() == QEvent::MouseButtonRelease) {
        const QString p = o->property("open").toString();
        if (!p.isEmpty()) {
            QTimer::singleShot(0, this, [this, p] {
                if (QFileInfo(p).isDir()) open_source(p, true);
                else add_files({std::filesystem::path(p.toStdString())});
            });
            return true;
        }
    }
    return QMainWindow::eventFilter(o, e);
}

void MainWindow::log(const QString& line) {
    if (log_) log_->appendPlainText(line);
}

}  // namespace rudra::app

// ---------------------------------------------------------------------------
// Phase 4 step 11: movies and their exports
// ---------------------------------------------------------------------------

namespace rudra::app {

QString MainWindow::shot_title() const {
    if (frames_.empty()) return {};
    if (shot_video_) return QString::fromStdString(shot_video_->path.filename().string());
    const auto& first = frames_.front();
    return QString::fromStdString(frames_.size() > 1 ? first.parent_path().filename().string() : first.filename().string());
}

std::string MainWindow::frame_name(std::size_t i) const {
    if (shot_video_ && i < std::size_t(shot_video_->count())) return shot_video_->name_of(int(i));
    return i < frames_.size() ? frames_[i].filename().string() : std::string();
}

VideoQueueOptions MainWindow::queue_options() const {
    VideoQueueOptions o;
    if (backend_choice_) {
        o.runtime = to_string(backend_choice_->runtime);
        o.device = backend_choice_->device;
    }
    return o;
}

bool MainWindow::queue_video_export(const QString& format) {
    auto say = [this](const QString& m) {
        log(m);
        if (auto* status = findChild<QLabel*>("renderStatus")) {
            status->setText(m);
            status->setVisible(!m.isEmpty());
        }
    };
    if (!shot_video_) {
        say("Open a movie to export it to HDR10, HLG or ProRes.");
        return false;
    }
    if (!manifest_) {
        say("Open a model package first.");
        return false;
    }
    const QString dir = findChild<QLineEdit*>("renderDir")->text().trimmed();
    if (dir.isEmpty()) {
        say("Choose a render folder first.");
        return false;
    }
    QString name = findChild<QLineEdit*>("renderName")->text().trimmed();
    if (name.isEmpty()) name = QString::fromStdString(shot_video_->path.stem().string()) + "_" + format;
    const bool prores = format.startsWith("prores");
    const std::filesystem::path output =
        std::filesystem::path(dir.toStdString()) / (name.toStdString() + (prores ? ".mov" : ".mp4"));
    std::error_code ec;
    std::filesystem::path sidecar = output;
    sidecar += ".json";
    if (std::filesystem::exists(output, ec) || std::filesystem::exists(sidecar, ec)) {
        say(QString::fromStdString(output.filename().string()) + " is already there; nothing is replaced. Change the name.");
        return false;
    }
    // What this ffmpeg can do, before anything is written.
    auto caps = probe_ffmpeg();
    if (!caps) {
        say(QString::fromStdString(caps.error().message));
        return false;
    }
    const bool can = caps->zscale && (prores ? caps->prores_ks : caps->libx265);
    if (!can) {
        say(QString::fromStdString(std::string("This FFmpeg build needs ") + (prores ? "prores_ks" : "libx265") +
                                   " and zscale support (" + caps->version + ")"));
        return false;
    }
    auto queue = write_export_queue(shot_video_->path, output, manifest_->root / "manifest.json", format.toStdString());
    if (!queue) {
        say(QString::fromStdString(queue.error().message));
        return false;
    }
    const QString label = format == "hdr10" ? "HDR10" : format == "hlg" ? "HLG" : "ProRes 422 HQ";
    const std::string title = shot_title().toStdString() + " \xe2\x86\x92 " + label.toStdString() + ", " + output.filename().string();
    queues_->run({*queue, title}, queue_options());
    save_queues();
    say("Queued " + label + ": " + QString::fromStdString(output.string()));
    return true;
}

void MainWindow::resume_queue(const std::filesystem::path& queue) {
    for (const auto& e : queues_->entries())
        if (e.queue == queue) {
            queues_->run(e, queue_options(), true);   // a failed job is tried again, an interrupted one restarts
            return;
        }
}

void MainWindow::forget_queue(const std::filesystem::path& queue) {
    queues_->forget(queue);
    save_queues();
}

}  // namespace rudra::app
