#include "model_dialogs.hpp"

#include <QComboBox>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPointer>
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

FirstRun::FirstRun(MainWindow* w, bool with_card) : QDialog(w), w_(w) {
    setObjectName("firstRun");
    setWindowTitle("Welcome to RUDRA");
    resize(820, 640);
    auto* v = new QVBoxLayout(this);
    v->addWidget(label("The display", "frDisplayHead", "plabel", this));
#ifdef RUDRA_APP_VIEWER
    // The HDR card through the viewer's own passes, on this display, at its
    // own peak: the patches say on the glass what the numbers below say.
    if (with_card) {
        auto* win = new ViewerWindow();
        win->set_input_enabled(false);
        const NetworkLinearImage card = hdr_card();
        win->set_composited(card, card);
        ViewParams vp;
        vp.display_nits = 10000.0;   // the ceiling is the display's own peak
        win->set_view(vp);
        card_ = QWidget::createWindowContainer(win, this);
        card_->setObjectName("frCard");
        card_->setMinimumHeight(300);
        v->addWidget(card_, 1);
        QPointer<FirstRun> self(this);
        win->on_status([self](const ViewerStatus& s) {
            if (self) self->set_display(s.target, s.swapchain, s.peak_from);
        });
    }
#endif
    (void)with_card;
    display_ = label("", "frDisplay", "value", this);
    display_detail_ = label("", "frDisplayDetail", "note", this);
    v->addWidget(display_);
    v->addWidget(display_detail_);
    if (!card_) {
        display_->setText("Display not checked");
        display_detail_->setText("This window has no viewer (a build without Qt 6.6 Shader Tools, or the tests): "
                                 "the HDR card cannot be shown.");
    }

    v->addWidget(label("The model", "frModelHead", "plabel", this));
    w_->rescan_models();
    const Catalog& c = w_->catalog();
    const auto pick = c.pick();
    model_ = label(pick ? q(c.entries[*pick].label()) + "  ·  " + q(c.entries[*pick].package.string())
                        : QString("No model package found. Add one in File > Model packages… later."),
                   "frModel", "value", this);
    v->addWidget(model_);
    auto* row = new QHBoxLayout;
    backend_ = new QComboBox(this);
    backend_->setObjectName("frBackend");
    fill_backend_combo(backend_, w_->model_backend());
    auto* test = button("Load and test", "frTest", this);
    test->setEnabled(pick.has_value());
    row->addWidget(backend_, 1);
    row->addWidget(test);
    v->addLayout(row);
    model_status_ = label(pick ? "The package's golden frames run once on the backend you choose; a model that drifts "
                                 "is refused."
                               : QString(),
                          "frModelStatus", "note", this);
    v->addWidget(model_status_);
    auto* start_btn = button("Start", "frStart", this);
    start_btn->setProperty("role", "primary");
    auto* end = new QHBoxLayout;
    end->addStretch(1);
    end->addWidget(start_btn);
    v->addLayout(end);
    connect(test, &QPushButton::clicked, this, [this] { test_model(); });
    connect(start_btn, &QPushButton::clicked, this, [this] { start(); });
}

void FirstRun::set_display(const DisplayTarget& target, const std::string& swapchain, const std::string& peak_from) {
    const DisplayReport r = display_report(target, swapchain, peak_from);
    display_->setText(q(r.headline));
    display_detail_->setText(q(r.detail));
    display_->setProperty("state", r.hdr ? "on" : "off");
}

void FirstRun::test_model() {
    const Catalog& c = w_->catalog();
    const auto pick = c.pick();
    if (!pick) return;
    model_status_->setText("Loading and checking…");
    QPointer<FirstRun> self(this);
    w_->use_model(c.entries[*pick].package, backend_from_combo(backend_), [self](bool ok, const QString& why) {
        if (!self) return;
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
