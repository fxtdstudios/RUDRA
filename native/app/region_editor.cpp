#include "region_editor.hpp"

#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QStyle>
#include <QVBoxLayout>

#include "rudra/core/js_format.hpp"

namespace rudra::app {
namespace {

void repolish(QWidget* w) {
    w->style()->unpolish(w);
    w->style()->polish(w);
    w->update();
}

// The page's .sw: an 11 px square, accent-dim when the row is selected.
class Swatch : public QWidget {
public:
    explicit Swatch(QWidget* parent) : QWidget(parent) { setFixedSize(11, 11); }
    bool on = false;

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setPen(QPen(palette().color(on ? QPalette::Link : QPalette::Midlight), 1));
        p.setBrush(on ? palette().color(QPalette::Highlight) : Qt::transparent);
        p.drawRoundedRect(QRectF(0.5, 0.5, 10, 10), 2, 2);
    }
};

}  // namespace

RegionEditor::RegionEditor(Session& session, QWidget* parent) : QWidget(parent), session_(session) {
    setObjectName("regions");
    setProperty("role", "regions");
    setAttribute(Qt::WA_StyledBackground, true);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(4, 0, 4, 8);
    v->setSpacing(0);
    sync();
}

void RegionEditor::build(int n) {
    auto* v = static_cast<QVBoxLayout*>(layout());
    for (auto& r : rows_) delete r.row;
    rows_.clear();
    for (int i = 0; i < n; ++i) {
        Row r;
        r.row = new QWidget(this);
        r.row->setProperty("role", "region");
        r.row->setAttribute(Qt::WA_StyledBackground, true);
        r.row->setCursor(Qt::PointingHandCursor);
        auto* h = new QHBoxLayout(r.row);
        h->setContentsMargins(8, 5, 8, 5);
        h->setSpacing(8);
        h->addWidget(new Swatch(r.row));
        r.q = new QLabel(r.row);
        r.q->setProperty("role", "region-q");
        h->addWidget(r.q, 1);
        r.ev = new QLabel(r.row);
        r.ev->setProperty("role", "region-ev");
        r.ev->setCursor(Qt::SizeHorCursor);
        h->addWidget(r.ev);
        auto* u = new QLabel("EV", r.row);
        u->setProperty("role", "region-u");
        u->setFixedWidth(20);
        h->addWidget(u);
        r.row->installEventFilter(this);
        r.ev->installEventFilter(this);
        v->addWidget(r.row);
        rows_.push_back(r);
    }
}

void RegionEditor::sync() {
    const auto& regions = session_.grade.regions;
    if (int(rows_.size()) != int(regions.size())) build(int(regions.size()));
    for (std::size_t i = 0; i < regions.size(); ++i) {
        const auto& g = regions[i];
        auto& r = rows_[i];
        r.q->setText(QString::fromStdString(nits_label(g.low_nits) + " – " + nits_label(g.high_nits) + " nits"));
        r.ev->setText(QString::fromStdString(js_signed(g.ev, 2)));
        const bool live = std::abs(g.ev) > 1e-9, sel = int(i) == session_.region_sel;
        if (r.ev->property("live").toBool() != live) {
            r.ev->setProperty("live", live);
            repolish(r.ev);
        }
        if (r.row->property("sel").toBool() != sel) {
            r.row->setProperty("sel", sel);
            repolish(r.row);
            if (auto* sw = r.row->findChild<QWidget*>()) {
                static_cast<Swatch*>(sw)->on = sel;
                sw->update();
            }
        }
    }
}

bool RegionEditor::eventFilter(QObject* o, QEvent* e) {
    int i = -1;
    bool on_value = false;
    for (std::size_t k = 0; k < rows_.size(); ++k) {
        if (o == rows_[k].ev) {
            i = int(k);
            on_value = true;
        } else if (o == rows_[k].row) {
            i = int(k);
        }
    }
    if (i < 0) return QWidget::eventFilter(o, e);
    auto* me = static_cast<QMouseEvent*>(e);
    switch (e->type()) {
        case QEvent::MouseButtonPress:
            if (me->button() != Qt::LeftButton) break;
            if (on_value) {
                // pointerdown on .ev: the row is selected and the drag begins.
                dragging_ = i;
                // x in the value label's own coordinates: it does not move while
                // it is dragged, and they keep a high-DPI mouse's fractions.
                session_.region_press(i, me->position().x());
            } else {
                session_.select_region(i);
            }
            return true;
        case QEvent::MouseMove:
            if (dragging_ >= 0) {
                session_.region_move(me->position().x(), me->modifiers() & Qt::ShiftModifier);
                return true;
            }
            break;
        case QEvent::MouseButtonRelease:
            if (dragging_ >= 0) {
                dragging_ = -1;
                session_.region_release();
                return true;
            }
            break;
        case QEvent::MouseButtonDblClick:
            // dblclick on .ev: zero it (the second press of the pair is this).
            if (on_value && me->button() == Qt::LeftButton) {
                session_.region_zero(i);
                return true;
            }
            break;
        default: break;
    }
    return QWidget::eventFilter(o, e);
}

}  // namespace rudra::app
