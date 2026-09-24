#include "queue_window.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

#include "main_window.hpp"
#include "video_queues.hpp"

namespace rudra::app {
namespace fs = std::filesystem;

namespace {
QString phase_word(const std::string& p) {
    if (p == "inference") return "Reconstructing";
    if (p == "encoding") return "Encoding";
    if (p == "quality_check") return "Checking";
    if (p == "starting") return "Starting";
    return QString::fromStdString(p);
}
}  // namespace

QString QueueWindow::describe(const QueueJobState& s, bool waiting, bool current) {
    if (s.status == "complete") return "Complete, QC passed";
    if (s.status == "failed") return "Failed: " + QString::fromStdString(s.error);
    if (s.status == "interrupted") return "Stopped; Resume starts this clip again";
    if (current) {
        if (s.status == "running" && s.frames_total > 0)
            return QStringLiteral("%1, frame %2 of %3").arg(phase_word(s.phase)).arg(s.frames_done).arg(s.frames_total);
        return phase_word(s.phase.empty() ? "starting" : s.phase);
    }
    if (waiting) return "Waiting";
    if (s.status == "running") return "Not running; Resume continues it";   // left running by a crash
    return "Not started";
}

QueueWindow::QueueWindow(MainWindow* w) : QDialog(w), w_(w) {
    setObjectName("queueWindow");
    setWindowTitle("Queue");
    setModal(false);
    resize(620, 360);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(22, 18, 22, 16);
    v->setSpacing(10);
    auto* title = new QLabel("Queue", this);
    title->setProperty("role", "title");
    v->addWidget(title);
    auto* note = new QLabel("Each export is checked before it is published and never replaces a file. A stopped "
                            "export resumes here or with rudra-native batch run.",
                            this);
    note->setProperty("role", "note");
    note->setWordWrap(true);
    v->addWidget(note);
    auto* area = new QScrollArea(this);
    area->setWidgetResizable(true);
    area->setFrameShape(QFrame::NoFrame);
    auto* list = new QWidget(area);
    rows_ = new QVBoxLayout(list);
    rows_->setContentsMargins(0, 0, 0, 0);
    rows_->setSpacing(8);
    empty_ = new QLabel("Nothing queued. Export a movie to HDR10, HLG or ProRes from the export sheet.", list);
    empty_->setObjectName("queueEmpty");
    empty_->setProperty("role", "note");
    rows_->addWidget(empty_);
    rows_->addStretch(1);
    area->setWidget(list);
    v->addWidget(area, 1);
    auto* close = new QPushButton("Close", this);
    close->setObjectName("queueClose");
    connect(close, &QPushButton::clicked, this, &QDialog::close);
    auto* foot = new QHBoxLayout;
    foot->addStretch(1);
    foot->addWidget(close);
    v->addLayout(foot);
    tick_.setInterval(250);
    connect(&tick_, &QTimer::timeout, this, [this] { refresh(); });
    tick_.start();
    refresh();
}

QueueWindow::Row& QueueWindow::row_for(const fs::path& queue, const QString& title) {
    auto it = rowmap_.find(queue);
    if (it != rowmap_.end()) return it->second;
    Row r;
    r.box = new QWidget(this);
    r.box->setProperty("role", "card");
    r.box->setAttribute(Qt::WA_StyledBackground, true);
    r.box->setObjectName("queueRow");
    r.box->setProperty("queue", QString::fromStdString(queue.string()));
    auto* v = new QVBoxLayout(r.box);
    v->setContentsMargins(14, 10, 14, 10);
    v->setSpacing(6);
    auto* top = new QHBoxLayout;
    r.title = new QLabel(title, r.box);
    r.title->setObjectName("queueTitle");
    top->addWidget(r.title, 1);
    r.stop = new QPushButton("Stop", r.box);
    r.stop->setObjectName("queueStop");
    r.resume = new QPushButton("Resume", r.box);
    r.resume->setObjectName("queueResume");
    r.remove = new QPushButton("Remove", r.box);
    r.remove->setObjectName("queueRemove");
    top->addWidget(r.stop);
    top->addWidget(r.resume);
    top->addWidget(r.remove);
    v->addLayout(top);
    r.bar = new QProgressBar(r.box);
    r.bar->setObjectName("queueBar");
    r.bar->setTextVisible(false);
    r.bar->setFixedHeight(4);
    v->addWidget(r.bar);
    r.status = new QLabel(r.box);
    r.status->setObjectName("queueStatus");
    r.status->setProperty("role", "note");
    r.status->setWordWrap(true);
    v->addWidget(r.status);
    rows_->insertWidget(rows_->count() - 1, r.box);
    connect(r.stop, &QPushButton::clicked, this, [this] { w_->video_queues().stop(); });
    connect(r.resume, &QPushButton::clicked, this, [this, queue] { w_->resume_queue(queue); });
    connect(r.remove, &QPushButton::clicked, this, [this, queue] {
        w_->forget_queue(queue);
        auto it2 = rowmap_.find(queue);
        if (it2 != rowmap_.end()) {
            it2->second.box->deleteLater();
            rowmap_.erase(it2);
        }
        refresh();
    });
    return rowmap_.emplace(queue, r).first->second;
}

void QueueWindow::refresh() {
    auto& q = w_->video_queues();
    const auto entries = q.entries();
    empty_->setVisible(entries.empty());
    const auto current = q.current();
    for (const auto& e : entries) {
        Row& r = row_for(e.queue, QString::fromStdString(e.title));
        const QueueJobState s = read_queue_state(e.queue);
        const bool is_current = current && *current == e.queue;
        const bool waiting = q.waiting(e.queue);
        r.status->setText(describe(s, waiting, is_current));
        r.status->setToolTip(QString::fromStdString(e.queue.string()));
        const int pct = s.status == "complete" ? 100
                        : s.frames_total > 0   ? int(100.0 * s.frames_done / s.frames_total * (s.phase == "inference" ? 0.9 : 1.0))
                                               : 0;
        r.bar->setValue(std::clamp(pct, 0, 100));
        r.stop->setVisible(is_current);
        r.resume->setVisible(!is_current && !waiting && s.status != "complete");
        r.remove->setEnabled(!is_current);
    }
}

void MainWindow::open_queue_window() {
    if (!queue_window_) queue_window_ = new QueueWindow(this);
    static_cast<QueueWindow*>(queue_window_.data())->refresh();
    queue_window_->show();
    queue_window_->raise();
}

}  // namespace rudra::app
