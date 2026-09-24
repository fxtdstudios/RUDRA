#pragma once
// The queue window (Phase 4, step 11): every video export with its state as
// the runner writes it (waiting, the phase and frames, complete, failed with
// its reason, interrupted), a progress bar, and Stop, Resume and Remove. It
// reads the state files, so what it shows is what `rudra batch status` shows.

#include <QDialog>
#include <QTimer>

#include <filesystem>
#include <map>

class QLabel;
class QProgressBar;
class QPushButton;
class QVBoxLayout;

namespace rudra::app {

class MainWindow;

class QueueWindow : public QDialog {
public:
    explicit QueueWindow(MainWindow* w);
    void refresh();
    // The words a row shows for a job's state (the tests read them).
    static QString describe(const struct QueueJobState& s, bool waiting, bool current);

private:
    struct Row {
        QWidget* box = nullptr;
        QLabel *title = nullptr, *status = nullptr;
        QProgressBar* bar = nullptr;
        QPushButton *stop = nullptr, *resume = nullptr, *remove = nullptr;
    };
    Row& row_for(const std::filesystem::path& queue, const QString& title);
    MainWindow* w_;
    QVBoxLayout* rows_ = nullptr;
    QLabel* empty_ = nullptr;
    std::map<std::filesystem::path, Row> rowmap_;
    QTimer tick_;
};

}  // namespace rudra::app
