#include "main_window.hpp"

#include <QActionGroup>
#include <QApplication>
#include <QDialog>
#include <QFileDialog>
#include <QFileInfo>
#include <QGridLayout>
#include <QKeySequence>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QPointer>
#include <QStatusBar>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>

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

// The first backend this build and machine can run, fastest first.
Result<std::unique_ptr<InferenceBackend>> open_backend(const ModelManifest& m) {
    const std::vector<std::pair<Runtime, Device>> order = {
#if defined(_WIN32)
        {Runtime::LibTorch, Device::Cuda}, {Runtime::OnnxRuntime, Device::DirectML},
#elif defined(__APPLE__)
        {Runtime::OnnxRuntime, Device::CoreML}, {Runtime::LibTorch, Device::Mps},
#else
        {Runtime::LibTorch, Device::Cuda},
#endif
        {Runtime::OnnxRuntime, Device::Cpu}, {Runtime::LibTorch, Device::Cpu},
    };
    Error last = make_error(ErrorCode::Unsupported, "This build has no inference runtime.");
    for (auto [rt, dev] : order) {
        auto b = rt == Runtime::LibTorch ? make_libtorch_backend(m, dev) : make_onnxruntime_backend(m, dev);
        if (b) return b;
        last = b.error();
    }
    return last;
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
    resize(1280, 800);
    auto* body = new QWidget(this);
    auto* layout = new QVBoxLayout(body);
    layout->setContentsMargins(0, 0, 0, 0);
    model_ = new QLabel("No model package open.", body);
    model_->setAlignment(Qt::AlignCenter);
    model_->setProperty("role", "key");
    model_->setContentsMargins(8, 6, 8, 6);
    layout->addWidget(model_);
#ifdef RUDRA_APP_VIEWER
    if (with_viewer) {
        viewer_ = new ViewerWindow();
        auto* container = QWidget::createWindowContainer(viewer_, body);
        container->setObjectName("viewerHost");
        container->setFocusPolicy(Qt::StrongFocus);
        container->setMinimumSize(320, 200);
        layout->addWidget(container, 1);
        viewer_->on_status([this](const ViewerStatus&) { show_status(); });
    } else {
        layout->addStretch(1);
    }
#else
    (void)with_viewer;
    layout->addStretch(1);
#endif
    setCentralWidget(body);
    bind_handlers();
    build_menus();
    sync_checks();
    refresh_enabled();

    QStringList runtimes;
    for (auto r : compiled_runtimes()) runtimes << to_string(r);
    runtimes_ = runtimes.isEmpty() ? "no inference runtime" : runtimes.join(", ");
    statusBar()->showMessage(runtimes_);
}

MainWindow::~MainWindow() {
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
    h["open"] = [this] { open_source(); };
    h["open-folder"] = [this] { open_source({}, true); };
    h["open-package"] = [this] { open_package(); };
    h["quit"] = [] { QApplication::quit(); };
    h["close"] = [this] { close_frames(); };
    h["first"] = [this] { step_to(0); };
    h["prev"] = [this] { step_to(current_ - 1); };
    h["next"] = [this] { step_to(current_ + 1); };
    h["last"] = [this] { step_to(int(frames_.size()) - 1); };
    h["play"] = [this] { toggle_play(); };
    h["mode-all"] = [this] { set_mode(RecoveryMode::All); };
    h["mode-highlights"] = [this] { set_mode(RecoveryMode::Highlights); };
    h["mode-shadows"] = [this] { set_mode(RecoveryMode::Shadows); };
    h["mode-off"] = [this] { set_mode(RecoveryMode::Off); };
    h["preserve"] = [this] {
        composite_.preserve_outside = !composite_.preserve_outside;
#ifdef RUDRA_APP_VIEWER
        if (viewer_) viewer_->set_composite(composite_);
#endif
    };
    // The page's nudgeStrength: 0.1 a press, rounded to hundredths, 0 to 2.
    h["strength-down"] = [this] { nudge_strength(-0.1); };
    h["strength-up"] = [this] { nudge_strength(0.1); };
    h["reset-recon"] = [this] {
        composite_.mode = RecoveryMode::All;
        composite_.strength = 1.0f;
        composite_.preserve_outside = true;
#ifdef RUDRA_APP_VIEWER
        if (viewer_) viewer_->set_composite(composite_);
#endif
        statusBar()->showMessage("reconstruction reset", 2000);
    };
    h["container-aces"] = [this] { container_ = "aces"; };
    h["container-linear"] = [this] { container_ = "linear"; };
    h["shortcuts"] = [this] {
        std::vector<std::pair<QString, QString>> rows;
        for (const auto& r : shortcut_sheet()) rows.emplace_back(qs(r.keys), qs(r.what));
        show_sheet("Keyboard", rows);
    };
    h["about"] = [this] {
        QString device = "—";
        if (backend_) {
            const auto info = backend_->info();
            device = QStringLiteral("%1 on %2").arg(to_string(info.runtime), to_string(info.device));
        }
        QString composite = "CPU";
#ifdef RUDRA_APP_VIEWER
        if (viewer_) composite = QStringLiteral("GPU, QRhi %1 — render/shaders/composite.frag")
                                     .arg(QString::fromStdString(viewer_->status().backend));
#endif
        show_sheet("RUDRA Studio", {{"Checkpoint", manifest_ ? QString::fromStdString(manifest_->name) : "—"},
                                    {"Step", "—"},
                                    {"Device", device},
                                    {"Composite", composite},
                                    {"Storage", "log2_extended, 0.005–1 000 000 nits"},
                                    {"Master", "ACES 2065-1 (AP0), ST 2065-4 chromaticities"},
                                    {"Built by", "FXTD Studios / Radiance Research"}});
    };

    // Waiting on later steps of Phase 3: off, and they say why.
    for (const char* id : {"undo", "redo", "reset-regions"}) pending_[id] = "Undo and Region EV arrive with Phase 3 step 3.";
    for (const char* id : {"rail-left", "rail-right", "scopes"}) pending_[id] = "The rails arrive with Phase 3 step 4.";
    for (const char* id : {"copy-metrics", "copy-scopes", "remeasure", "copy-delivery"})
        pending_[id] = "Measurements arrive with Phase 3 step 8.";
    pending_["master"] = "Master arrives with Phase 3 step 9.";

#ifdef RUDRA_APP_VIEWER
    {
        auto view = [this](auto&& edit) {
            return [this, edit] {
                if (!viewer_) return;
                auto v = viewer_->view();
                edit(v);
                viewer_->set_view(v);
            };
        };
        auto guides = [this](auto&& edit) {
            return [this, edit] {
                if (!viewer_) return;
                auto g = viewer_->guides();
                edit(g);
                viewer_->set_guides(g);
            };
        };
        h["wipe"] = view([](ViewParams& v) { v.wipe = v.wipe >= 0.0 ? -1.0 : 0.5; });
        h["zoom-fit"] = [this] { if (viewer_) viewer_->zoom_fit(); };
        h["zoom-actual"] = [this] { if (viewer_) viewer_->zoom_actual(); };
        h["view-image"] = view([](ViewParams& v) { v.mode = ViewMode::Image; });
        h["view-false-colour"] = view([](ViewParams& v) { v.mode = ViewMode::FalseColour; });
        h["view-difference"] = view([](ViewParams& v) { v.mode = ViewMode::Difference; });
        h["guides-action"] = guides([](GuideOptions& g) { g.action_safe = !g.action_safe; });
        h["guides-title"] = guides([](GuideOptions& g) { g.title_safe = !g.title_safe; });
        h["guides-centre"] = guides([](GuideOptions& g) { g.centre = !g.centre; });
        const std::pair<const char*, double> aspects[] = {{"aspect-none", 0.0}, {"aspect-2.39", 2.39},
                                                          {"aspect-1.85", 1.85}, {"aspect-16:9", 16.0 / 9.0},
                                                          {"aspect-4:3", 4.0 / 3.0}, {"aspect-1:1", 1.0}};
        for (const auto& [id, r] : aspects) h[id] = guides([r = r](GuideOptions& g) { g.aspect = r; });
        const std::pair<const char*, double> peaks[] = {{"peak-203", 203.0}, {"peak-400", 400.0},
                                                        {"peak-1000", 1000.0}, {"peak-4000", 4000.0},
                                                        {"peak-display", 10000.0}};
        for (const auto& [id, n] : peaks) h[id] = view([n = n](ViewParams& v) { v.display_nits = n; });
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
        for (auto k : mapped_keys())
            if (action_for_key(k) == spec.id) {
                const QKeySequence seq = key_sequence(k);
                if (!keys.contains(seq)) keys << seq;
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
            else if (e.substr(0, 1) == ">") stack.push_back(stack.back()->addMenu(qs(e.substr(1))));
            else stack.back()->addAction(actions_.at(std::string(e)));
        }
        connect(menu, &QMenu::aboutToShow, this, [this] {
            sync_checks();
            refresh_enabled();
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
            case EnableRule::CanMaster: on = any && backend_ != nullptr; break;
            case EnableRule::CanUndo:
            case EnableRule::CanRedo:
            case EnableRule::HasMetrics:
            case EnableRule::HasScopes: on = false; break;
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
    set("mode-all", composite_.mode == RecoveryMode::All);
    set("mode-highlights", composite_.mode == RecoveryMode::Highlights);
    set("mode-shadows", composite_.mode == RecoveryMode::Shadows);
    set("mode-off", composite_.mode == RecoveryMode::Off);
    set("preserve", composite_.preserve_outside);
    set("container-aces", container_ == "aces");
    set("container-linear", container_ == "linear");
    set("rail-left", true);
    set("rail-right", true);
    set("scopes", true);
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const auto v = viewer_->view();
        const auto g = viewer_->guides();
        const bool fit = viewer_->viewport().scale <= 0.0;
        set("zoom-fit", fit);
        set("zoom-actual", !fit);
        set("wipe", v.wipe >= 0.0);
        set("view-image", v.mode == ViewMode::Image);
        set("view-false-colour", v.mode == ViewMode::FalseColour);
        set("view-difference", v.mode == ViewMode::Difference);
        set("guides-action", g.action_safe);
        set("guides-title", g.title_safe);
        set("guides-centre", g.centre);
        set("aspect-none", g.aspect <= 0.0);
        set("aspect-2.39", std::abs(g.aspect - 2.39) < 1e-9);
        set("aspect-1.85", std::abs(g.aspect - 1.85) < 1e-9);
        set("aspect-16:9", std::abs(g.aspect - 16.0 / 9.0) < 1e-9);
        set("aspect-4:3", std::abs(g.aspect - 4.0 / 3.0) < 1e-9);
        set("aspect-1:1", std::abs(g.aspect - 1.0) < 1e-9);
        set("peak-203", v.display_nits == 203.0);
        set("peak-400", v.display_nits == 400.0);
        set("peak-1000", v.display_nits == 1000.0);
        set("peak-4000", v.display_nits == 4000.0);
        set("peak-display", v.display_nits >= 10000.0);
        return;
    }
#endif
    set("zoom-fit", true);
}

void MainWindow::show_sheet(const QString& title, const std::vector<std::pair<QString, QString>>& rows) {
    QDialog d(this);
    d.setObjectName("sheet");
    d.setWindowTitle(title);
    auto* grid = new QGridLayout(&d);
    auto* head = new QLabel(title, &d);
    head->setProperty("role", "plabel");
    grid->addWidget(head, 0, 0, 1, 2);
    int r = 1;
    for (const auto& [k, v] : rows) {
        auto* kl = new QLabel(k, &d);
        kl->setProperty("role", "value");
        auto* vl = new QLabel(v, &d);
        grid->addWidget(kl, r, 0);
        grid->addWidget(vl, r, 1);
        ++r;
    }
    auto* close = new QLabel("Esc to close", &d);
    close->setProperty("role", "note");
    grid->addWidget(close, r, 0, 1, 2);
    d.exec();
}

void MainWindow::set_mode(RecoveryMode m) {
    if (composite_.mode == m) return;
    composite_.mode = m;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_composite(composite_);
#endif
}

void MainWindow::nudge_strength(double d) {
    const double s = std::clamp(std::round((double(composite_.strength) + d) * 100.0) / 100.0, 0.0, 2.0);
    composite_.strength = float(s);
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->set_composite(composite_);
#endif
    statusBar()->showMessage(QStringLiteral("strength %1").arg(s, 0, 'f', 2), 1500);
}

void MainWindow::open_package(const QString& preset) {
    const QString dir = preset.isEmpty() ? QFileDialog::getExistingDirectory(this, "Open model package") : preset;
    if (dir.isEmpty()) return;
    auto m = read_manifest(dir.toStdString());
    if (!m) {
        model_->setText(QString::fromStdString(m.error().message + " " + m.error().detail));
        return;
    }
    auto v = verify_package_files(*m);
    if (!v) {
        model_->setText(describe(*m) + "  ·  " + QString::fromStdString(v.error().message));
        return;
    }
    auto b = open_backend(*m);
    if (!b) {
        model_->setText(describe(*m) + "  ·  " + QString::fromStdString(b.error().message));
        return;
    }
    // The engine's worker uses the backend: it goes first.
    play_.stop();
    engine_.reset();
    frames_.clear();
    manifest_ = std::make_unique<ModelManifest>(*m);
    backend_ = std::move(*b);
    const auto info = backend_->info();
    model_->setText(describe(*m) + QStringLiteral("  ·  %1 on %2").arg(to_string(info.runtime), to_string(info.device)));
    refresh_enabled();
}

void MainWindow::open_source(const QString& preset, bool folder) {
#if defined(RUDRA_APP_VIEWER) && defined(RUDRA_HAVE_STILL_DECODE)
    if (!backend_) {
        QMessageBox::information(this, "RUDRA", "Open a model package first.");
        return;
    }
    QString path = preset;
    if (path.isEmpty())
        path = folder ? QFileDialog::getExistingDirectory(this, "Open a folder of SDR frames")
                      : QFileDialog::getOpenFileName(this, "Open an SDR still", {},
                                                     "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)");
    if (path.isEmpty()) return;
    std::vector<std::filesystem::path> frames;
    if (QFileInfo(path).isDir()) {
        auto seq = open_sequence(path.toStdString());
        if (!seq) {
            statusBar()->showMessage(QString::fromStdString(seq.error().message));
            return;
        }
        frames = seq->frames;
    } else {
        frames = {std::filesystem::path(path.toStdString())};
    }
    start_engine(std::move(frames));
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
    frames_.clear();
    current_ = 0;
    frame_info_.clear();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->clear_frame();
#endif
    refresh_enabled();
    show_status();
}

void MainWindow::start_engine(std::vector<std::filesystem::path> frames) {
    play_.stop();
    engine_.reset();   // joins the old worker before the new one starts
    frames_ = std::move(frames);
    auto* backend = backend_.get();
    const ModelConstants model{manifest_->log_scale, manifest_->max_hdr, manifest_->corpus_ev};
    auto files = frames_;
    engine_ = std::make_unique<FrameEngine>(
        [files](int i) -> Result<SdrImage> {
#ifdef RUDRA_HAVE_STILL_DECODE
            auto d = decode_sdr_file(files[std::size_t(i)]);
            if (!d) return d.error();
            return std::move(d->rgb);
#else
            // A build without media's still decode (no OpenCV) opens no
            // frames; the viewer and the rest of the shell still work.
            (void)files;
            (void)i;
            return make_error(ErrorCode::Unsupported, "This build has no still decode (RUDRA_WITH_OPENCV=OFF).");
#endif
        },
        [backend](const SdrImage& sdr) {
            // One untiled pass, as the Studio's preview.
            return infer_frame(*backend, sdr, TileConfig{0, 0});
        });
    QPointer<MainWindow> self(this);
    engine_->on_ready([self, model](const ReadyFrame& f) {
        QMetaObject::invokeMethod(qApp, [self, f, model] {
            if (self) self->frame_ready(f, model);
        });
    });
    engine_->set_sequence(int(frames_.size()));
    refresh_enabled();
    step_to(0);
}

void MainWindow::step_to(int i) {
    if (!engine_ || frames_.empty()) return;
    const int n = int(frames_.size());
    current_ = ((i % n) + n) % n;
    waiting_ = true;
    engine_->show(current_);
}

void MainWindow::toggle_play() {
    if (play_.isActive()) {
        play_.stop();
        return;
    }
    // 24 fps, the Studio's default; a frame that is not ready is held, not skipped.
    play_.setInterval(1000 / 24);
    QObject::connect(&play_, &QTimer::timeout, this, [this] {
        if (!waiting_) step_to(current_ + 1);
    }, Qt::UniqueConnection);
    play_.start();
}

void MainWindow::frame_ready(const ReadyFrame& f, const ModelConstants& model) {
    if (f.index != current_) return;   // the engine already dropped stale ones; this is the UI's own check
    waiting_ = false;
    const QString name = QString::fromStdString(frames_[std::size_t(f.index)].filename().string());
    if (f.error) {
        statusBar()->showMessage(name + ": " + QString::fromStdString(f.error->message));
        return;
    }
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        viewer_->set_frame({*f.sdr, f.fields->fields, f.fields->scalars, model});
        viewer_->set_composite(composite_);
    }
#else
    (void)model;
#endif
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
    QString view;
#ifdef RUDRA_APP_VIEWER
    if (viewer_) {
        const ViewerStatus s = viewer_->status();
        const bool hdr = s.target.path != OutputPath::SdrPqSimulation;
        view = QString::fromStdString(s.backend) + " · " + QString::fromStdString(s.swapchain) +
               (hdr ? QStringLiteral(" · peak %1 nits").arg(std::lround(s.target.peak_nits)) : QString()) +
               QStringLiteral(" · %1% · ").arg(s.zoom_percent);
    }
#endif
    statusBar()->showMessage(view + runtimes_ + (frame_info_.isEmpty() ? QString() : "  ·  " + frame_info_));
}

}  // namespace rudra::app
