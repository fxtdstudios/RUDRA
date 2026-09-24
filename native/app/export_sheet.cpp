#include "export_sheet.hpp"

#include <QComboBox>
#include <QDir>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMouseEvent>
#include <QPushButton>
#include <QSpinBox>
#include <QStyle>
#include <QVBoxLayout>

#include <functional>

#include "main_window.hpp"

namespace rudra::app {
namespace {

QLabel* label(const QString& text, const QString& id, const QString& role, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    if (!id.isEmpty()) l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    return l;
}

// A format tile: a click picks it.
class Tile : public QWidget {
public:
    Tile(const QString& id, const QString& name, const QString& sub, bool offered, QWidget* parent)
        : QWidget(parent) {
        setObjectName(id);
        setProperty("role", "tile");
        setAttribute(Qt::WA_StyledBackground, true);
        setFixedHeight(84);
        setEnabled(offered);
        setCursor(offered ? Qt::PointingHandCursor : Qt::ArrowCursor);
        if (!offered) setToolTip("Arrives with video delivery; rudra deliver writes it from the command line today.");
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(12, 10, 12, 10);
        v->setSpacing(3);
        v->addStretch(1);
        v->addWidget(label(name, {}, "tile-name", this));
        auto* s = label(sub, {}, "tile-sub", this);
        s->setWordWrap(true);
        v->addWidget(s);
    }
    std::function<void()> picked;
    void set_on(bool on) {
        setProperty("on", on);
        style()->unpolish(this);
        style()->polish(this);
    }

protected:
    void mousePressEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton && picked) picked();
    }
};

}  // namespace

ExportSheet::ExportSheet(MainWindow* w) : QDialog(w), w_(w) {
    setObjectName("exportSheet");
    setWindowTitle("Export");
    setModal(true);
    setFixedWidth(720);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(26, 22, 26, 16);
    v->setSpacing(14);

    auto* head = new QWidget(this);
    auto* hv = new QVBoxLayout(head);
    hv->setContentsMargins(0, 0, 0, 0);
    hv->setSpacing(2);
    title_ = label("Export", "exportTitle", "title", head);
    sub_ = label("", "exportSub", "note", head);
    hv->addWidget(title_);
    hv->addWidget(sub_);
    v->addWidget(head);

    auto* tiles = new QWidget(this);
    auto* tg = new QGridLayout(tiles);
    tg->setContentsMargins(0, 0, 0, 0);
    tg->setSpacing(8);
    const struct { const char* id; const char* name; const char* sub; bool offered; } formats[] = {
        {"exr-aces", "EXR", "ACES 2065-1, half", true},
        {"exr-linear", "EXR linear", "Rec.2020 linear, half", true},
        {"hdr10", "HDR10", "HEVC 10-bit, streaming", false},
        {"hlg", "HLG", "HEVC 10-bit, broadcast", false},
        {"prores", "422 HQ", "ProRes, editorial", false}};
    int col = 0;
    for (const auto& f : formats) {
        auto* t = new Tile(f.id, f.name, f.sub, f.offered, tiles);
        const QString id = f.id;
        t->picked = [this, id] { pick(id); };
        tg->addWidget(t, 0, col++);
        tg->setColumnStretch(col - 1, 1);
    }
    v->addWidget(tiles);

    auto* card = new QWidget(this);
    card->setProperty("role", "card");
    card->setAttribute(Qt::WA_StyledBackground, true);
    auto* cv = new QVBoxLayout(card);
    cv->setContentsMargins(14, 2, 14, 2);
    cv->setSpacing(0);
    auto row = [&](const QString& key, QLabel* value) {
        auto* r = new QWidget(card);
        r->setProperty("role", "field");
        auto* h = new QHBoxLayout(r);
        h->setContentsMargins(0, 10, 0, 10);
        h->addWidget(label(key, {}, "key", r));
        h->addStretch(1);
        value->setProperty("role", "value");
        h->addWidget(value);
        cv->addWidget(r);
    };
    signal_ = label("", "exportSignal", {}, card);
    frames_ = label("", "exportFrames", {}, card);
    dest_ = label("", "exportDest", {}, card);
    row("Signal", signal_);
    row("Frames", frames_);
    row("Save to", dest_);
    row("White", label("diffuse 1.0 = 203 nits", {}, {}, card));
    v->addWidget(card);

    auto* checks = new QWidget(this);
    auto* chv = new QVBoxLayout(checks);
    chv->setContentsMargins(0, 0, 0, 0);
    chv->setSpacing(8);
    chv->addWidget(label("Checked before the file is published", {}, "note", checks));
    auto* chips = new QWidget(checks);
    auto* chh = new QHBoxLayout(chips);
    chh->setContentsMargins(0, 0, 0, 0);
    chh->setSpacing(6);
    for (const char* c : {"The render plan", "Every name", "Nothing replaced", "Staged, then linked"})
        chh->addWidget(label(c, {}, "chip", chips));
    chh->addStretch(1);
    chv->addWidget(chips);
    auto* side = label("A sidecar with the model, the settings and the frame's measurements is written beside each "
                       "file. Edit the folder and the name on the Deliver tab.",
                       {}, "note", checks);
    side->setWordWrap(true);
    chv->addWidget(side);
    v->addWidget(checks);

    auto* foot = new QWidget(this);
    auto* fh = new QHBoxLayout(foot);
    fh->setContentsMargins(0, 8, 0, 0);
    fh->setSpacing(10);
    fh->addStretch(1);
    auto* cancel = new QPushButton("Cancel", foot);
    cancel->setObjectName("exportCancel");
    auto* go = new QPushButton("Export", foot);
    go->setObjectName("exportGo");
    go->setProperty("role", "primary");
    connect(cancel, &QPushButton::clicked, this, &QDialog::reject);
    connect(go, &QPushButton::clicked, this, [this] { export_now(); });
    fh->addWidget(cancel);
    fh->addWidget(go);
    v->addWidget(foot);
    refresh();
}

void ExportSheet::pick(const QString& id) {
    if (id != "exr-aces" && id != "exr-linear") return;
    picked_ = id;
    // The container is the session's: the same switch as Deliver > container.
    w_->run(id == "exr-aces" ? "container-aces" : "container-linear");
    refresh();
}

void ExportSheet::refresh() {
    picked_ = w_->container() == "linear" ? "exr-linear" : "exr-aces";
    for (const char* id : {"exr-aces", "exr-linear", "hdr10", "hlg", "prores"})
        if (auto* t = findChild<QWidget*>(id)) {
            t->setProperty("on", picked_ == id);
            t->style()->unpolish(t);
            t->style()->polish(t);
        }
    const auto& frames = w_->frames();
    const QString shot = frames.empty() ? QStringLiteral("no shot")
                         : frames.size() > 1 ? QString::fromStdString(frames.front().parent_path().filename().string())
                                             : QString::fromStdString(frames.front().filename().string());
    title_->setText("Export " + shot);
    sub_->setText(QStringLiteral("%1 frame%2 loaded").arg(frames.size()).arg(frames.size() == 1 ? "" : "s"));
    signal_->setText(picked_ == "exr-aces" ? "Scene linear · AP0 · half" : "Scene linear · Rec.2020 · half");
    auto* mode = w_->findChild<QComboBox*>("renderMode");
    const bool sequence = mode && mode->currentData().toString() == "sequence";
    frames_->setText(sequence ? QStringLiteral("All %1, as a sequence").arg(frames.size()) : QStringLiteral("The frame on screen"));
    const QString dir = w_->findChild<QLineEdit*>("renderDir")->text().trimmed();
    const QString name = w_->findChild<QLineEdit*>("renderName")->text().trimmed();
    dest_->setText(dir.isEmpty() ? QStringLiteral("choose a folder on the Deliver tab")
                                 : QDir::toNativeSeparators(dir) + QDir::separator() + name +
                                       (sequence ? QStringLiteral(".######.exr") : QStringLiteral(".exr")));
    if (auto* go = findChild<QPushButton*>("exportGo")) go->setEnabled(!dir.isEmpty() && !frames.empty());
}

void ExportSheet::export_now() {
    accept();
    w_->run("master");
}

void MainWindow::open_export_sheet() {
    if (!export_sheet_) export_sheet_ = new ExportSheet(this);
    static_cast<ExportSheet*>(export_sheet_.data())->refresh();
    export_sheet_->show();
    export_sheet_->raise();
}

}  // namespace rudra::app
