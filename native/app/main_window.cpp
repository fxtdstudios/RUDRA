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
#include <QFrame>
#include <QKeyEvent>
#include <QLayout>
#include <QLocale>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSignalBlocker>
#include <QSlider>
#include <QStackedWidget>
#include <QStyle>
#include <QVBoxLayout>

#include "region_editor.hpp"
#include "scope_widgets.hpp"
#include "widgets.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <thread>

#include "rudra/core/baseline.hpp"
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
    resize(1600, 1000);
    session_.on_change([this](std::uint32_t what) { session_changed(what); });
    build_ui(with_viewer);
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
    // The page's grading state and its undo: engine/session.
    for (const auto& spec : action_specs())
        if (Session::owns(spec.id)) h[std::string(spec.id)] = [this, id = std::string(spec.id)] { session_.run(id); };
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

    h["view-image"] = [this] { session_.set_view_layer(0); };
    h["view-false-colour"] = [this] { session_.set_view_layer(1); };
    h["view-difference"] = [this] { session_.set_view_layer(2); };

    // Waiting on later steps of Phase 3: off, and they say why.
    for (const char* id : {"copy-metrics", "copy-scopes", "copy-delivery"})
        pending_[id] = "Copying arrives with Phase 3 step 11.";
    h["remeasure"] = [this] {
        measure_now();
        log("re-measured");
    };
    pending_["master"] = "Master arrives with Phase 3 step 9.";

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
        for (const auto& [id, r] : aspects) h[id] = guides([r = r](GuideOptions& g) { g.aspect = r; });
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
            else if (e.substr(0, 1) == ">") stack.push_back(stack.back()->addMenu(qs(e.substr(1))));
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
            case EnableRule::CanMaster: on = any && backend_ != nullptr; break;
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
    auto m = read_manifest(dir.toStdString());
    if (!m) {
        log(QString::fromStdString(m.error().message + " " + m.error().detail));
        return;
    }
    auto v = verify_package_files(*m);
    if (!v) {
        log(describe(*m) + "  ·  " + QString::fromStdString(v.error().message));
        return;
    }
    auto b = open_backend(*m);
    if (!b) {
        log(describe(*m) + "  ·  " + QString::fromStdString(b.error().message));
        return;
    }
    // The engine's worker uses the backend: it goes first.
    play_.stop();
    engine_.reset();
    frames_.clear();
    manifest_ = std::make_unique<ModelManifest>(*m);
    backend_ = std::move(*b);
    const auto info = backend_->info();
    ckpt_->setText(QString::fromStdString(m->name));
    device_->setText(QStringLiteral("%1 on %2").arg(to_string(info.runtime), to_string(info.device)));
    lamp_->setProperty("state", "on");
    lamp_->style()->unpolish(lamp_);
    lamp_->style()->polish(lamp_);
    log("model " + describe(*m));
    log("device " + device_->text());
    sync_ui();
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
            log(QString::fromStdString(seq.error().message));
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
    current_frame_.reset();
    measure_.reset();
    ++stats_gen_;
    clear_scopes();
    clip_bar_->hide();
    probe_pixel(std::nullopt);
    update_pipe();
#ifdef RUDRA_APP_VIEWER
    if (viewer_) viewer_->clear_frame();
#endif
    refresh_enabled();
    sync_ui();
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
    sync_ui();
    step_to(0);
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
        log(name + ": " + QString::fromStdString(f.error->message));
        return;
    }
    FrameHeader header;
    const QString res = QStringLiteral("%1x%2").arg(f.sdr->width()).arg(f.sdr->height());
    header.source_resolution = header.resolution = res.toStdString();
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
    panels_->setCurrentIndex(tab_ == "grade" ? 1 : tab_ == "deliver" ? 2 : 0);
    for (auto* n : notes_) n->setVisible(workspace_ != "simple");
    // Frames (drawFrames)
    const int n = int(frames_.size());
    shot_count_->setText(QString::number(n));
    frames_empty_->setVisible(n == 0);
    if (auto* dz = findChild<QWidget*>("dropzone")) dz->setVisible(n == 0);   // shell.js: tucked once frames open
    tc_->setText(timecode(n ? current_ : 0, 24.0));
    scrub_->set_position(n > 1 ? double(current_) / double(n - 1) : 0.0);
    for (auto* b : {btn_prev_, btn_play_, btn_next_}) b->setEnabled(n > 1);
    const bool ready = n > 0 && backend_ != nullptr;
    btn_master_->setEnabled(ready && pending_reason("master").isEmpty());
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
    std::thread([self, frame, params, gen] {
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
    }).detach();
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
    update_pipe();
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

void MainWindow::log(const QString& line) {
    if (log_) log_->appendPlainText(line);
}

}  // namespace rudra::app
