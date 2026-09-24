// RUDRA: the Qt application shell.
//
// Phase 2 steps 9 and 10: the viewer (render/viewer_window.hpp, a QWindow
// with its own swapchain, ADR-010) sits in the middle of the window. File
// opens a model package, then a still or a folder of frames; frames reach the
// viewer through the engine's FrameEngine (engine/frame_engine.hpp): decode
// and inference on its own thread, generations so a late result never lands
// on the wrong frame, read-ahead and a frame cache, as the Studio's page does.
// Comma and full stop step, Home and End jump, Space plays. The rails and the
// Pro-direction layout are Phase 3; its look (theme.cpp, from ui/theme.css)
// is step 1.

#include <QActionGroup>
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QFile>
#include <QFileInfo>
#include <QMessageBox>
#include <QPointer>
#include <QStatusBar>
#include <QVBoxLayout>

#include <algorithm>
#include <filesystem>
#include <memory>

#include <QKeyEvent>
#include <QTimer>

#include "theme.hpp"

#include "rudra/core/model_manifest.hpp"
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

namespace {

QString describe(const rudra::ModelManifest& m) {
    return QStringLiteral("%1  ·  contract %2  ·  residual gate %3  ·  shadow gate %4  ·  curve %5  ·  corpus EV %6")
        .arg(QString::fromStdString(m.name), QString::fromStdString(m.contract),
             m.has_residual_gate ? "yes" : "no", m.has_shadow_gate ? "yes" : "no", m.has_curve ? "yes" : "no")
        .arg(m.corpus_ev, 0, 'f', 1);
}

// The first backend this build and machine can run, fastest first.
rudra::Result<std::unique_ptr<rudra::InferenceBackend>> open_backend(const rudra::ModelManifest& m) {
    using rudra::Device;
    using rudra::Runtime;
    const std::vector<std::pair<Runtime, Device>> order = {
#if defined(_WIN32)
        {Runtime::OnnxRuntime, Device::DirectML}, {Runtime::LibTorch, Device::Cuda},
#elif defined(__APPLE__)
        {Runtime::OnnxRuntime, Device::CoreML}, {Runtime::LibTorch, Device::Mps},
#else
        {Runtime::LibTorch, Device::Cuda},
#endif
        {Runtime::OnnxRuntime, Device::Cpu}, {Runtime::LibTorch, Device::Cpu},
    };
    rudra::Error last = rudra::make_error(rudra::ErrorCode::Unsupported, "This build has no inference runtime.");
    for (auto [rt, dev] : order) {
        auto b = rt == Runtime::LibTorch ? rudra::make_libtorch_backend(m, dev) : rudra::make_onnxruntime_backend(m, dev);
        if (b) return b;
        last = b.error();
    }
    return last;
}

class MainWindow : public QMainWindow {
public:
    MainWindow() {
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
        viewer_ = new rudra::ViewerWindow();
        auto* container = QWidget::createWindowContainer(viewer_, body);
        container->setObjectName("viewerHost");
        container->setFocusPolicy(Qt::StrongFocus);
        container->setMinimumSize(320, 200);
        layout->addWidget(container, 1);
        viewer_->on_status([this](const rudra::ViewerStatus& s) { show_status(s); });
#else
        layout->addStretch(1);
#endif
        setCentralWidget(body);
        build_menus();

        QStringList runtimes;
        for (auto r : rudra::compiled_runtimes()) runtimes << rudra::to_string(r);
        runtimes_ = runtimes.isEmpty() ? "no inference runtime" : runtimes.join(", ");
        statusBar()->showMessage(runtimes_);
    }

    void open_package(const QString& preset = {}) {
        const QString dir = preset.isEmpty() ? QFileDialog::getExistingDirectory(this, "Open model package") : preset;
        if (dir.isEmpty()) return;
        auto m = rudra::read_manifest(dir.toStdString());
        if (!m) {
            model_->setText(QString::fromStdString(m.error().message + " " + m.error().detail));
            return;
        }
        auto v = rudra::verify_package_files(*m);
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
#ifdef RUDRA_APP_VIEWER
        play_.stop();
#endif
        engine_.reset();
        frames_.clear();
        manifest_ = std::make_unique<rudra::ModelManifest>(*m);
        backend_ = std::move(*b);
        const auto info = backend_->info();
        model_->setText(describe(*m) + QStringLiteral("  ·  %1 on %2").arg(rudra::to_string(info.runtime),
                                                                            rudra::to_string(info.device)));
    }

    // A still, or a folder of frames: both are a sequence to the engine.
    void open_source(const QString& preset = {}, bool folder = false) {
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
            auto seq = rudra::open_sequence(path.toStdString());
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

protected:
    void keyPressEvent(QKeyEvent* e) override {
#ifdef RUDRA_APP_VIEWER
        if (engine_ && frames_.size() > 1) {
            const int n = int(frames_.size());
            switch (e->key()) {
                case Qt::Key_Comma: step_to(current_ - 1); return;
                case Qt::Key_Period: step_to(current_ + 1); return;
                case Qt::Key_Home: step_to(0); return;
                case Qt::Key_End: step_to(n - 1); return;
                case Qt::Key_Space: toggle_play(); return;
                default: break;
            }
        }
#endif
        QMainWindow::keyPressEvent(e);
    }

private:
    void build_menus() {
        auto* file = menuBar()->addMenu("&File");
        file->addAction("Open model package…", QKeySequence("Ctrl+Shift+O"), this, [this] { open_package(); });
        file->addAction("Open still…", QKeySequence::Open, this, [this] { open_source(); });
        file->addAction("Open folder of frames…", QKeySequence("Ctrl+Alt+O"), this, [this] { open_source({}, true); });
        file->addSeparator();
        file->addAction("Quit", QKeySequence::Quit, qApp, &QApplication::quit);
#ifdef RUDRA_APP_VIEWER
        auto* view = menuBar()->addMenu("&View");
        auto* layers = new QActionGroup(this);
        const std::pair<const char*, rudra::ViewMode> modes[] = {{"Image", rudra::ViewMode::Image},
                                                                 {"False colour", rudra::ViewMode::FalseColour},
                                                                 {"Difference", rudra::ViewMode::Difference}};
        for (const auto& [label, mode] : modes) {
            auto* a = view->addAction(label, this, [this, mode = mode] {
                auto v = viewer_->view();
                v.mode = mode;
                viewer_->set_view(v);
            });
            a->setCheckable(true);
            a->setChecked(mode == rudra::ViewMode::Image);
            layers->addAction(a);
        }
        view->addSeparator();
        view->addAction("Wipe", QKeySequence("W"), this, [this] {
            auto v = viewer_->view();
            v.wipe = v.wipe >= 0.0 ? -1.0 : 0.5;
            viewer_->set_view(v);
        });
        view->addAction("Fit", QKeySequence("Ctrl+0"), this, [this] { viewer_->zoom_fit(); });
        view->addAction("Actual pixels", QKeySequence("Ctrl+1"), this, [this] { viewer_->zoom_actual(); });
        auto* guides = view->addMenu("Guides");
        auto toggle = [this](bool rudra::GuideOptions::*flag) {
            return [this, flag] (bool on) {
                auto g = viewer_->guides();
                g.*flag = on;
                viewer_->set_guides(g);
            };
        };
        auto* action_safe = guides->addAction("Action safe (90 %)", QKeySequence("G"), this,
                                              toggle(&rudra::GuideOptions::action_safe));
        auto* title_safe = guides->addAction("Title safe (80 %)", QKeySequence("Shift+G"), this,
                                             toggle(&rudra::GuideOptions::title_safe));
        auto* centre = guides->addAction("Centre cross", this, toggle(&rudra::GuideOptions::centre));
        for (auto* a : {action_safe, title_safe, centre}) a->setCheckable(true);
        guides->addSeparator();
        auto* aspects = new QActionGroup(this);
        const std::pair<const char*, double> ratios[] = {{"No aspect mask", 0.0}, {"2.39", 2.39}, {"1.85", 1.85},
                                                          {"16:9", 16.0 / 9.0}, {"4:3", 4.0 / 3.0}, {"1:1", 1.0}};
        for (const auto& [label, r] : ratios) {
            auto* a = guides->addAction(label, this, [this, r = r] {
                auto g = viewer_->guides();
                g.aspect = r;
                viewer_->set_guides(g);
            });
            a->setCheckable(true);
            a->setChecked(r == 0.0);
            aspects->addAction(a);
        }
        auto* peak = view->addMenu("View peak");
        for (double nits : {203.0, 400.0, 1000.0, 4000.0, 10000.0}) {
            peak->addAction(nits >= 10000.0 ? QString("The display's own") : QString("%1 nits").arg(nits), this,
                            [this, nits] {
                                auto v = viewer_->view();
                                v.display_nits = nits;
                                viewer_->set_view(v);
                            });
        }
        auto* grade = menuBar()->addMenu("&Reconstruction");
        const std::pair<const char*, rudra::RecoveryMode> recs[] = {{"All", rudra::RecoveryMode::All},
                                                                    {"Highlights", rudra::RecoveryMode::Highlights},
                                                                    {"Shadows", rudra::RecoveryMode::Shadows},
                                                                    {"Off", rudra::RecoveryMode::Off}};
        auto* rec_group = new QActionGroup(this);
        int key = 1;
        for (const auto& [label, mode] : recs) {
            auto* a = grade->addAction(label, QKeySequence(QString::number(key++)), this, [this, mode = mode] {
                composite_.mode = mode;
                viewer_->set_composite(composite_);
            });
            a->setCheckable(true);
            a->setChecked(mode == rudra::RecoveryMode::All);
            rec_group->addAction(a);
        }
        grade->addSeparator();
        grade->addAction("Strength down", QKeySequence("["), this, [this] { nudge_strength(-0.05f); });
        grade->addAction("Strength up", QKeySequence("]"), this, [this] { nudge_strength(0.05f); });
        auto* preserve = grade->addAction("Preserve outside the masks", QKeySequence("P"), this, [this](bool on) {
            composite_.preserve_outside = on;
            viewer_->set_composite(composite_);
        });
        preserve->setCheckable(true);
        preserve->setChecked(true);
#endif
    }

#ifdef RUDRA_APP_VIEWER
    void nudge_strength(float d) {
        composite_.strength = std::clamp(composite_.strength + d, 0.0f, 2.0f);
        viewer_->set_composite(composite_);
        statusBar()->showMessage(QStringLiteral("strength %1").arg(double(composite_.strength), 0, 'f', 2), 1500);
    }

    void start_engine(std::vector<std::filesystem::path> frames) {
        play_.stop();
        engine_.reset();   // joins the old worker before the new one starts
        frames_ = std::move(frames);
        auto* backend = backend_.get();
        const rudra::ModelConstants model{manifest_->log_scale, manifest_->max_hdr, manifest_->corpus_ev};
        auto files = frames_;
        engine_ = std::make_unique<rudra::FrameEngine>(
            [files](int i) -> rudra::Result<rudra::SdrImage> {
#ifdef RUDRA_HAVE_STILL_DECODE
                auto d = rudra::decode_sdr_file(files[std::size_t(i)]);
                if (!d) return d.error();
                return std::move(d->rgb);
#else
                // A build without media's still decode (no OpenCV) opens no
                // frames; the viewer and the rest of the shell still work.
                (void)files;
                (void)i;
                return rudra::make_error(rudra::ErrorCode::Unsupported,
                                         "This build has no still decode (RUDRA_WITH_OPENCV=OFF).");
#endif
            },
            [backend](const rudra::SdrImage& sdr) {
                // One untiled pass, as the Studio's preview.
                return rudra::infer_frame(*backend, sdr, rudra::TileConfig{0, 0});
            });
        QPointer<MainWindow> self(this);
        engine_->on_ready([self, model](const rudra::ReadyFrame& f) {
            QMetaObject::invokeMethod(qApp, [self, f, model] {
                if (self) self->frame_ready(f, model);
            });
        });
        engine_->set_sequence(int(frames_.size()));
        step_to(0);
    }

    void step_to(int i) {
        if (!engine_ || frames_.empty()) return;
        const int n = int(frames_.size());
        current_ = ((i % n) + n) % n;
        waiting_ = true;
        engine_->show(current_);
    }

    void toggle_play() {
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

    void frame_ready(const rudra::ReadyFrame& f, const rudra::ModelConstants& model) {
        if (f.index != current_) return;   // the engine already dropped stale ones; this is the UI's own check
        waiting_ = false;
        const QString name = QString::fromStdString(frames_[std::size_t(f.index)].filename().string());
        if (f.error) {
            statusBar()->showMessage(name + ": " + QString::fromStdString(f.error->message));
            return;
        }
        viewer_->set_frame({*f.sdr, f.fields->fields, f.fields->scalars, model});
        viewer_->set_composite(composite_);
        const auto st = engine_->stats();
        frame_info_ = QStringLiteral("%1  %2/%3  ·  %4").arg(name).arg(f.index + 1).arg(frames_.size())
                          .arg(f.from_cache ? QStringLiteral("cached")
                                            : QStringLiteral("decode %1 ms, infer %2 ms").arg(f.decode_ms, 0, 'f', 0)
                                                  .arg(f.infer_ms, 0, 'f', 0));
        frame_info_ += QStringLiteral("  ·  %1 in cache").arg(st.cached);
        show_status(viewer_->status());
    }

    void show_status(const rudra::ViewerStatus& s) {
        const bool hdr = s.target.path != rudra::OutputPath::SdrPqSimulation;
        const QString path = QString::fromStdString(s.swapchain) +
                             (hdr ? QStringLiteral(" · peak %1 nits").arg(std::lround(s.target.peak_nits)) : QString());
        statusBar()->showMessage(QStringLiteral("%1 · %2 · %3% · %4%5").arg(QString::fromStdString(s.backend), path)
                                     .arg(s.zoom_percent)
                                     .arg(runtimes_, frame_info_.isEmpty() ? QString() : "  ·  " + frame_info_));
    }

    rudra::ViewerWindow* viewer_ = nullptr;
    rudra::CompositeParams composite_;
    QTimer play_;
    int current_ = 0;
    bool waiting_ = false;
    QString frame_info_;
#endif
    QLabel* model_ = nullptr;
    QString runtimes_;
    std::unique_ptr<rudra::ModelManifest> manifest_;
    std::unique_ptr<rudra::InferenceBackend> backend_;   // used only from the engine's worker
    std::vector<std::filesystem::path> frames_;
    std::unique_ptr<rudra::FrameEngine> engine_;         // declared after the backend: destroyed first
};

}  // namespace

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
    MainWindow w;
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
