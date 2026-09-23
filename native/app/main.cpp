// RUDRA: the Qt application shell.
//
// Phase 0 scope: a window that opens a model package, verifies it and shows
// which inference runtimes this build carries, so the whole layer stack
// (platform -> core -> infer -> app) is linked and exercised. The viewer
// (render/, QRhi), the rails and the Pro-direction layout arrive in Phases 2-3.

#include <QApplication>
#include <QFileDialog>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QStatusBar>
#include <QVBoxLayout>

#include "rudra/core/model_manifest.hpp"
#include "rudra/infer/backend.hpp"

namespace {

QString describe(const rudra::ModelManifest& m) {
    return QStringLiteral("%1  ·  contract %2\nresidual gate %3  ·  shadow gate %4  ·  curve %5\ncorpus EV %6")
        .arg(QString::fromStdString(m.name), QString::fromStdString(m.contract),
             m.has_residual_gate ? "yes" : "no", m.has_shadow_gate ? "yes" : "no", m.has_curve ? "yes" : "no")
        .arg(m.corpus_ev, 0, 'f', 1);
}

class MainWindow : public QMainWindow {
public:
    MainWindow() {
        setWindowTitle("RUDRA");
        resize(960, 600);
        auto* body = new QWidget(this);
        auto* layout = new QVBoxLayout(body);
        model_ = new QLabel("No model package open.", body);
        model_->setAlignment(Qt::AlignCenter);
        layout->addWidget(model_);
        setCentralWidget(body);
        // The surround is neutral grey: R = G = B, no tint (ui/theme.css).
        setStyleSheet("QMainWindow, QWidget { background: #141414; color: #c9c9c9; }"
                      "QLabel { font-size: 15px; }");

        auto* file = menuBar()->addMenu("&File");
        file->addAction("Open model package…", QKeySequence::Open, this, [this] { open(); });
        file->addAction("Quit", QKeySequence::Quit, qApp, &QApplication::quit);

        QStringList runtimes;
        for (auto r : rudra::compiled_runtimes()) runtimes << rudra::to_string(r);
        statusBar()->showMessage(runtimes.isEmpty() ? "No inference runtime in this build"
                                                    : "Runtimes: " + runtimes.join(", "));
    }

    void open(const QString& preset = {}) {
        const QString dir = preset.isEmpty() ? QFileDialog::getExistingDirectory(this, "Open model package") : preset;
        if (dir.isEmpty()) return;
        auto m = rudra::read_manifest(dir.toStdString());
        if (!m) { model_->setText(QString::fromStdString(m.error().message + "\n" + m.error().detail)); return; }
        auto v = rudra::verify_package_files(*m);
        model_->setText(describe(*m) + (v ? "\nfiles verified" : "\n" + QString::fromStdString(v.error().message)));
    }

private:
    QLabel* model_ = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    QApplication::setApplicationName("RUDRA");
    QApplication::setOrganizationName("FXTD Studios");
    MainWindow w;
    if (argc > 1) w.open(QString::fromLocal8Bit(argv[1]));
    w.show();
    return QApplication::exec();
}
