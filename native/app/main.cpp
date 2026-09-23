// RUDRA: the Qt application shell.
//
// Phase 2 step 9: the viewer (render/viewer_window.hpp, a QWindow with its
// own swapchain, ADR-010) sits in the middle of the window. File opens a
// model package and a still; the still is decoded, inferred off the UI
// thread, and shown through the same composite and display passes the
// parity tools hold to the Studio. The rails and the Pro-direction layout
// are Phase 3; the engine's generations and cancellation are step 10.

#include <QActionGroup>
#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QMessageBox>
#include <QPointer>
#include <QStatusBar>
#include <QVBoxLayout>

#include <memory>
#include <thread>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"
#include "rudra/infer/tiler.hpp"

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
        // The surround is neutral grey: R = G = B, no tint (ui/theme.css).
        setStyleSheet("QMainWindow, QWidget { background: #141414; color: #c9c9c9; }"
                      "QLabel { font-size: 13px; } QStatusBar { color: #8a8a8a; }");
        auto* body = new QWidget(this);
        auto* layout = new QVBoxLayout(body);
        layout->setContentsMargins(0, 0, 0, 0);
        model_ = new QLabel("No model package open.", body);
        model_->setAlignment(Qt::AlignCenter);
        model_->setContentsMargins(8, 6, 8, 6);
        layout->addWidget(model_);
#ifdef RUDRA_APP_VIEWER
        viewer_ = new rudra::ViewerWindow();
        auto* container = QWidget::createWindowContainer(viewer_, body);
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
        manifest_ = std::make_unique<rudra::ModelManifest>(*m);
        backend_ = std::move(*b);
        const auto info = backend_->info();
        model_->setText(describe(*m) + QStringLiteral("  ·  %1 on %2").arg(rudra::to_string(info.runtime),
                                                                            rudra::to_string(info.device)));
    }

    void open_image(const QString& preset = {}) {
#if defined(RUDRA_APP_VIEWER) && defined(RUDRA_HAVE_STILL_DECODE)
        if (!backend_) {
            QMessageBox::information(this, "RUDRA", "Open a model package first.");
            return;
        }
        const QString path = preset.isEmpty()
            ? QFileDialog::getOpenFileName(this, "Open an SDR still", {}, "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
            : preset;
        if (path.isEmpty()) return;
        statusBar()->showMessage("Reconstructing " + path + " …");
        // Off the UI thread: one untiled forward pass, as the Studio's preview.
        auto* backend = backend_.get();
        const rudra::ModelConstants model{manifest_->log_scale, manifest_->max_hdr, manifest_->corpus_ev};
        QPointer<MainWindow> self(this);
        std::thread([self, backend, model, file = path.toStdString()] {
            auto decoded = rudra::decode_sdr_file(file);
            rudra::Result<rudra::FrameResult> fr = decoded ? rudra::infer_frame(*backend, decoded->rgb, rudra::TileConfig{0, 0})
                                                           : rudra::Result<rudra::FrameResult>(decoded.error());
            QMetaObject::invokeMethod(qApp, [self, decoded = std::move(decoded), fr = std::move(fr), model]() mutable {
                if (!self) return;
                if (!fr) {
                    self->statusBar()->showMessage(QString::fromStdString(fr.error().message));
                    return;
                }
                self->viewer_->set_frame({std::move(decoded->rgb), std::move(fr->fields), fr->scalars, model});
                self->viewer_->set_composite(self->composite_);
            });
        }).detach();
#else
        (void)preset;
        QMessageBox::information(this, "RUDRA", "This build has no viewer or no still decoder (Qt 6.6 with Shader "
                                                "Tools, and RUDRA_WITH_OPENCV).");
#endif
    }

private:
    void build_menus() {
        auto* file = menuBar()->addMenu("&File");
        file->addAction("Open model package…", QKeySequence("Ctrl+Shift+O"), this, [this] { open_package(); });
        file->addAction("Open still…", QKeySequence::Open, this, [this] { open_image(); });
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

    void show_status(const rudra::ViewerStatus& s) {
        const bool hdr = s.target.path != rudra::OutputPath::SdrPqSimulation;
        const QString path = QString::fromStdString(s.swapchain) +
                             (hdr ? QStringLiteral(" · peak %1 nits").arg(std::lround(s.target.peak_nits)) : QString());
        statusBar()->showMessage(QStringLiteral("%1 · %2 · %3% · %4").arg(QString::fromStdString(s.backend), path)
                                     .arg(s.zoom_percent)
                                     .arg(runtimes_));
    }

    rudra::ViewerWindow* viewer_ = nullptr;
    rudra::CompositeParams composite_;
#endif
    QLabel* model_ = nullptr;
    QString runtimes_;
    std::unique_ptr<rudra::ModelManifest> manifest_;
    std::unique_ptr<rudra::InferenceBackend> backend_;
};

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("RUDRA");
    QApplication::setOrganizationName("FXTD Studios");
    MainWindow w;
    if (argc > 1) w.open_package(QString::fromLocal8Bit(argv[1]));
    if (argc > 2) w.open_image(QString::fromLocal8Bit(argv[2]));
    w.show();
    return QApplication::exec();
}
