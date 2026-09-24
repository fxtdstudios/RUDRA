#include "model_dialogs.hpp"

#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPixmap>
#include <QPointer>
#include <QStyle>
#include <QPushButton>
#include <QSettings>
#include <QTreeWidget>
#include <QVBoxLayout>

#include "main_window.hpp"
#include "rudra/core/hdr_card.hpp"

#ifdef RUDRA_APP_VIEWER
#include "rudra/render/viewer_window.hpp"
#endif

namespace rudra::app {
namespace {

QLabel* label(const QString& text, const QString& id, const QString& role, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    l->setObjectName(id);
    if (!role.isEmpty()) l->setProperty("role", role);
    l->setWordWrap(true);
    return l;
}

QPushButton* button(const QString& text, const QString& id, QWidget* parent) {
    auto* b = new QPushButton(text, parent);
    b->setObjectName(id);
    return b;
}

QString q(const std::string& s) { return QString::fromStdString(s); }

}  // namespace

void fill_backend_combo(QComboBox* c, std::optional<BackendChoice> selected) {
    c->clear();
    c->addItem("Automatic (fastest that opens)", QString());
    for (const auto& b : backend_choices()) c->addItem(q(b.label()), q(b.key()));
    if (selected) {
        const int i = c->findData(q(selected->key()));
        if (i >= 0) c->setCurrentIndex(i);
    }
}

std::optional<BackendChoice> backend_from_combo(const QComboBox* c) {
    return BackendChoice::from_key(c->currentData().toString().toStdString());
}

// ---------------------------------------------------------------------------

ModelManager::ModelManager(MainWindow* w) : QDialog(w), w_(w) {
    setObjectName("modelManager");
    setWindowTitle("Model packages");
    resize(760, 460);
    auto* v = new QVBoxLayout(this);
    v->addWidget(label("Model packages", "ckptHead", "plabel", this));
    list_ = new QTreeWidget(this);
    list_->setObjectName("ckptList");
    list_->setHeaderLabels({"Model", "Package", "Source", "State"});
    list_->setRootIsDecorated(false);
    list_->header()->setStretchLastSection(true);
    v->addWidget(list_, 1);
    note_ = label("", "ckptNote", "note", this);
    v->addWidget(note_);
    auto* row = new QHBoxLayout;
    row->addWidget(label("Run on", "ckptBackendLabel", "value", this));
    backend_ = new QComboBox(this);
    backend_->setObjectName("ckptBackend");
    fill_backend_combo(backend_, w_->model_backend());
    row->addWidget(backend_, 1);
    auto* add = button("Add folder…", "ckptAdd", this);
    auto* rescan = button("Rescan", "ckptRescan", this);
    use_ = button("Use", "ckptUse", this);
    use_->setProperty("role", "primary");
    row->addWidget(add);
    row->addWidget(rescan);
    row->addWidget(use_);
    v->addLayout(row);
    status_ = label("", "ckptStatus", "note", this);
    v->addWidget(status_);

    connect(list_, &QTreeWidget::currentItemChanged, this, [this](QTreeWidgetItem* it) {
        if (it) select(std::size_t(it->data(0, Qt::UserRole).toULongLong()));
    });
    connect(add, &QPushButton::clicked, this, [this] {
        const QString d = QFileDialog::getExistingDirectory(this, "Add a folder of model packages");
        if (d.isEmpty()) return;
        w_->add_model_root(d.toStdString());
        refresh();
    });
    connect(rescan, &QPushButton::clicked, this, [this] {
        w_->rescan_models();
        refresh();
    });
    connect(use_, &QPushButton::clicked, this, [this] { use_selected(); });
    refresh();
}

void ModelManager::refresh() {
    const Catalog& c = w_->catalog();
    std::error_code ec;
    const auto in_use = w_->model_package().empty() ? std::filesystem::path()
                                                   : std::filesystem::weakly_canonical(w_->model_package(), ec);
    list_->clear();
    QTreeWidgetItem* current = nullptr;
    for (std::size_t i = 0; i < c.entries.size(); ++i) {
        const auto& e = c.entries[i];
        QString state;
        std::error_code e2;
        const bool used = !in_use.empty() && std::filesystem::weakly_canonical(e.package, e2) == in_use;
        if (!e.manifest) state = "unreadable: " + q(e.problem);
        else if (used) state = "in use";
        else if (c.chosen && *c.chosen == i) state = "opens on a bare start";
        else if (e.is_default) state = "default";
        auto* it = new QTreeWidgetItem({q(e.label()), q(e.package.filename().string()),
                                        e.manifest ? q(e.manifest->source_file) : QString(), state});
        it->setData(0, Qt::UserRole, qulonglong(i));
        it->setToolTip(1, q(e.package.string()));
        if (!e.manifest) it->setFlags(it->flags() & ~Qt::ItemIsEnabled);
        list_->addTopLevelItem(it);
        if (used || (!current && c.pick() && *c.pick() == i)) current = it;
    }
    for (int col = 0; col < 3; ++col) list_->resizeColumnToContents(col);
    if (current) list_->setCurrentItem(current);
    if (c.entries.empty()) {
        QStringList roots;
        for (const auto& r : c.roots) roots << q(r.string());
        note_->setText("No model package in " + (roots.isEmpty() ? QString("any folder") : roots.join(", ")) +
                       ". Add a folder that holds one (tools/export_model.py writes them).");
    }
    use_->setEnabled(list_->currentItem() != nullptr && !w_->loading_model());
}

void ModelManager::select(std::size_t entry) {
    const Catalog& c = w_->catalog();
    if (entry >= c.entries.size()) return;
    const auto& e = c.entries[entry];
    QString t;
    if (!e.note.empty()) t += q(e.note) + "\n";
    if (e.manifest) {
        const auto& m = *e.manifest;
        t += QStringLiteral("%1  ·  contract %2  ·  residual gate %3  ·  shadow gate %4  ·  curve %5  "
                            "·  corpus EV %6")
                 .arg(q(m.name), q(m.contract), m.has_residual_gate ? "yes" : "no", m.has_shadow_gate ? "yes" : "no",
                      m.has_curve ? "yes" : "no")
                 .arg(m.corpus_ev, 0, 'f', 1);
        if (!m.exported.empty()) t += "  ·  exported " + q(m.exported);
    }
    note_->setText(t);
    for (int i = 0; i < list_->topLevelItemCount(); ++i)
        if (list_->topLevelItem(i)->data(0, Qt::UserRole).toULongLong() == entry &&
            list_->currentItem() != list_->topLevelItem(i))
            list_->setCurrentItem(list_->topLevelItem(i));
}

void ModelManager::use_selected() {
    auto* it = list_->currentItem();
    if (!it) return;
    const std::size_t i = std::size_t(it->data(0, Qt::UserRole).toULongLong());
    const Catalog& c = w_->catalog();
    if (i >= c.entries.size() || !c.entries[i].manifest) return;
    status_->setText("Loading and checking " + q(c.entries[i].label()) + "…");
    use_->setEnabled(false);
    QPointer<ModelManager> self(this);
    w_->use_model(c.entries[i].package, backend_from_combo(backend_), [self](bool ok, const QString& why) {
        if (!self) return;
        self->refresh();
        self->status_->setText(
            ok ? "In use: " + q(self->w_->model_package().filename().string()) + " on " +
                     q(self->w_->model_backend() ? self->w_->model_backend()->label() : std::string())
               : why);
    });
}

// ---------------------------------------------------------------------------

namespace {

// One "Getting ready" step of the welcome: a state dot, a name, what it found.
QWidget* step(QWidget* parent, const QString& dot_id, const QString& name, QVBoxLayout*& body) {
    auto* r = new QWidget(parent);
    r->setProperty("role", "step");
    r->setAttribute(Qt::WA_StyledBackground, true);
    auto* h = new QHBoxLayout(r);
    h->setContentsMargins(0, 12, 0, 12);
    h->setSpacing(14);
    auto* dot = label({}, dot_id, "step-dot", r);
    dot->setFixedSize(26, 26);
    dot->setAlignment(Qt::AlignCenter);
    h->addWidget(dot, 0, Qt::AlignTop);
    auto* col = new QWidget(r);
    body = new QVBoxLayout(col);
    body->setContentsMargins(0, 0, 0, 0);
    body->setSpacing(4);
    auto* n = label(name, {}, "step-name", col);
    n->setWordWrap(false);
    body->addWidget(n);
    h->addWidget(col, 1);
    return r;
}

void set_dot(QWidget* root, const char* id, const char* state) {
    if (auto* d = root->findChild<QLabel*>(id)) {
        d->setProperty("state", state);
        d->setText(QString(state) == "done" ? QStringLiteral("✓") : QString(state) == "fail" ? QStringLiteral("!") : QString());
        d->style()->unpolish(d);
        d->style()->polish(d);
    }
}

}  // namespace

FirstRun::FirstRun(MainWindow* w, bool with_card) : QDialog(w), w_(w) {
    setObjectName("firstRun");
    setWindowTitle("Welcome to RUDRA");
    resize(980, 640);
    auto* h = new QHBoxLayout(this);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);

    // The left: the app, its name and what it does.
    auto* side = new QWidget(this);
    side->setObjectName("welcomeSide");
    side->setAttribute(Qt::WA_StyledBackground, true);
    side->setFixedWidth(360);
    auto* sv = new QVBoxLayout(side);
    sv->setContentsMargins(24, 24, 24, 24);
    sv->setSpacing(18);
    sv->addStretch(1);
    auto* icon = new QWidget(side);
    icon->setObjectName("welcomeIcon");
    icon->setAttribute(Qt::WA_StyledBackground, true);
    icon->setFixedSize(128, 128);
    auto* iv = new QVBoxLayout(icon);
    auto* mark = new QLabel(icon);
    mark->setPixmap(QPixmap(":/assets/rudra-mark.png").scaled(84, 84, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    mark->setAlignment(Qt::AlignCenter);
    iv->addWidget(mark);
    sv->addWidget(icon, 0, Qt::AlignHCenter);
    auto* t = label("Welcome to RUDRA", "welcomeTitle", {}, side);
    t->setAlignment(Qt::AlignCenter);
    t->setWordWrap(false);
    sv->addWidget(t);
    auto* sub = label("SDR footage back to scene-linear HDR, and a record of where it did it.", "welcomeSub", {}, side);
    sub->setAlignment(Qt::AlignCenter);
    sv->addWidget(sub);
    sv->addStretch(1);
    auto* ver = label("FXTD Studios", "welcomeVersion", {}, side);
    ver->setAlignment(Qt::AlignCenter);
    sv->addWidget(ver);
    h->addWidget(side);

    // The right: getting ready.
    auto* right = new QWidget(this);
    auto* v = new QVBoxLayout(right);
    v->setContentsMargins(36, 34, 36, 24);
    v->setSpacing(2);
    v->addWidget(label("Getting ready", "readyTitle", {}, right));
    v->addSpacing(8);

    QVBoxLayout* body = nullptr;
    v->addWidget(step(right, "frDotDisplay", "Display", body));
    display_ = label("", "frDisplay", "step-sub", right);
    display_detail_ = label("", "frDisplayDetail", "step-sub", right);
    body->addWidget(display_);
    body->addWidget(display_detail_);
#ifdef RUDRA_APP_VIEWER
    // The HDR card through the viewer's own passes, on this display, at its
    // own peak: the patches say on the glass what the words say.
    if (with_card) {
        auto* win = new ViewerWindow();
        win->set_input_enabled(false);
        const NetworkLinearImage card = hdr_card();
        win->set_composited(card, card);
        ViewParams vp;
        vp.display_nits = 10000.0;   // the ceiling is the display's own peak
        win->set_view(vp);
        card_ = QWidget::createWindowContainer(win, right);
        card_->setObjectName("frCard");
        card_->setFixedHeight(170);
        body->addWidget(card_);
        QPointer<FirstRun> self(this);
        win->on_status([self](const ViewerStatus& s) {
            if (self) self->set_display(s.target, s.swapchain, s.peak_from);
        });
    }
#endif
    (void)with_card;
    if (!card_) {
        display_->setText("Display not checked");
        display_detail_->setText("This window has no viewer (a build without Qt 6.6 Shader Tools, or the tests): "
                                 "the HDR card cannot be shown.");
    }

    w_->rescan_models();
    const Catalog& c = w_->catalog();
    const auto pick = c.pick();
    v->addWidget(step(right, "frDotBackend", "Acceleration", body));
    auto* row = new QWidget(right);
    auto* rh = new QHBoxLayout(row);
    rh->setContentsMargins(0, 2, 0, 0);
    rh->setSpacing(8);
    backend_ = new QComboBox(row);
    backend_->setObjectName("frBackend");
    fill_backend_combo(backend_, w_->model_backend());
    auto* test = button("Load and test", "frTest", row);
    test->setEnabled(pick.has_value() || !w_->model_package().empty());
    rh->addWidget(backend_, 1);
    rh->addWidget(test);
    body->addWidget(row);

    v->addWidget(step(right, "frDotModel", "Model", body));
    // The package in use if there is one (opened on the command line, say), else the catalog's pick.
    const bool in_use = !w_->model_package().empty();
    model_ = label(in_use ? q(w_->model_package().filename().string()) + "  ·  in use"
                   : pick ? q(c.entries[*pick].label()) + "  ·  " + q(c.entries[*pick].package.filename().string())
                          : QString("No model package found. Add one in File > Model packages… later."),
                   "frModel", "step-sub", right);
    body->addWidget(model_);
    model_status_ = label(pick ? "Verified against its hash, then its golden frames run once on the backend you choose; "
                                 "a model that drifts is refused."
                               : QString(),
                          "frModelStatus", "step-sub", right);
    body->addWidget(model_status_);
    v->addStretch(1);
    auto* lic = label("The model weights are licensed for non-commercial use (checkpoints/LICENSE).", "licenceNote", {},
                      right);
    v->addWidget(lic);
    v->addSpacing(14);

    auto* end = new QHBoxLayout;
    end->setSpacing(10);
    end->addWidget(label("You can start on the CPU and switch later.", {}, "step-sub", right));
    end->addStretch(1);
    auto* cpu = button("Start on CPU", "frStartCpu", right);
    auto* start_btn = button("Continue", "frStart", right);
    start_btn->setProperty("role", "primary");
    end->addWidget(cpu);
    end->addWidget(start_btn);
    v->addLayout(end);
    h->addWidget(right, 1);

    set_dot(this, "frDotDisplay", card_ ? "busy" : "");
    set_dot(this, "frDotBackend", "");
    set_dot(this, "frDotModel", in_use ? "done" : pick ? "" : "fail");
    connect(test, &QPushButton::clicked, this, [this] { test_model(); });
    connect(start_btn, &QPushButton::clicked, this, [this] { start(); });
    connect(cpu, &QPushButton::clicked, this, [this] {
        // The first CPU backend this build has.
        for (int i = 0; i < backend_->count(); ++i)
            if (backend_->itemData(i).toString().endsWith("/cpu")) {
                backend_->setCurrentIndex(i);
                break;
            }
        start();
    });
}

void FirstRun::set_display(const DisplayTarget& target, const std::string& swapchain, const std::string& peak_from) {
    const DisplayReport r = display_report(target, swapchain, peak_from);
    display_->setText(q(r.headline));
    display_detail_->setText(q(r.detail));
    display_->setProperty("state", r.hdr ? "on" : "off");
    set_dot(this, "frDotDisplay", "done");
}

void FirstRun::test_model() {
    const Catalog& c = w_->catalog();
    const auto pick = c.pick();
    const std::filesystem::path package = !w_->model_package().empty() ? w_->model_package()
                                          : pick ? c.entries[*pick].package : std::filesystem::path();
    if (package.empty()) return;
    model_status_->setText("Loading and checking…");
    set_dot(this, "frDotBackend", "busy");
    set_dot(this, "frDotModel", "busy");
    QPointer<FirstRun> self(this);
    w_->use_model(package, backend_from_combo(backend_), [self](bool ok, const QString& why) {
        if (!self) return;
        set_dot(self, "frDotBackend", ok ? "done" : "fail");
        set_dot(self, "frDotModel", ok ? "done" : "fail");
        self->model_status_->setText(
            ok ? "Ready: " + q(self->w_->model_package().filename().string()) + " on " +
                     q(self->w_->model_backend() ? self->w_->model_backend()->label() : std::string())
               : why);
    });
}

void FirstRun::start() {
    QSettings().setValue("firstRun/done", true);
    // No model yet: the one the catalog picks, on the automatic backend.
    if (w_->model_package().empty() && !w_->loading_model())
        if (const auto p = w_->catalog().pick()) w_->use_model(w_->catalog().entries[*p].package, backend_from_combo(backend_));
    accept();
}

// ---------------------------------------------------------------------------

void MainWindow::open_model_manager() {
    rescan_models();
    if (!manager_) manager_ = new ModelManager(this);
    else static_cast<ModelManager*>(manager_.data())->refresh();
    manager_->show();
    manager_->raise();
}

void MainWindow::open_first_run() {
    if (!first_run_) first_run_ = new FirstRun(this, viewer_ != nullptr);
    first_run_->show();
    first_run_->raise();
}

}  // namespace rudra::app
