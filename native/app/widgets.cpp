#include "widgets.hpp"

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QStyle>
#include <QStyleOption>

#include <algorithm>

namespace rudra::app {
namespace {

// Re-polish after a dynamic property changes, so the sheet's [on="true"] applies.
void repolish(QWidget* w) {
    w->style()->unpolish(w);
    w->style()->polish(w);
    w->update();
}

}  // namespace

Seg::Seg(const QString& id, const std::vector<std::pair<QString, QString>>& buttons, QWidget* parent, bool exclusive)
    : QWidget(parent) {
    setObjectName(id);
    setProperty("role", "seg");
    auto* row = new QHBoxLayout(this);
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(0);
    auto* group = new QButtonGroup(this);
    group->setExclusive(exclusive);
    for (const auto& [key, text] : buttons) {
        auto* b = new QPushButton(text, this);
        b->setProperty("role", "seg-button");
        b->setCheckable(true);
        b->setFocusPolicy(Qt::NoFocus);
        if (key.startsWith('#')) b->setObjectName(key.mid(1));   // a button with its own id (wipeBtn)
        group->addButton(b);
        row->addWidget(b, 1);
        buttons_.push_back(b);
        keys_.push_back(key);
        connect(b, &QPushButton::clicked, this, [this, key] {
            if (clicked) clicked(key);
        });
    }
}

QPushButton* Seg::button(const QString& key) const {
    for (std::size_t i = 0; i < keys_.size(); ++i)
        if (keys_[i] == key) return buttons_[i];
    return nullptr;
}

void Seg::set_on(const QString& key) {
    for (std::size_t i = 0; i < keys_.size(); ++i) {
        auto* b = buttons_[i];
        const bool on = keys_[i] == key;
        auto* g = b->group();
        if (g && g->exclusive() && !on && b->isChecked()) {
            g->setExclusive(false);
            b->setChecked(false);
            g->setExclusive(true);
        } else {
            b->setChecked(on);
        }
    }
}

QString Seg::on() const {
    for (std::size_t i = 0; i < keys_.size(); ++i)
        if (buttons_[i]->isChecked()) return keys_[i];
    return {};
}

CheckRow::CheckRow(const QString& id, const QString& label, const QString& hint_id, const QString& hint,
                   QWidget* parent)
    : QWidget(parent) {
    setObjectName(id);
    setProperty("role", "check");
    setCursor(Qt::PointingHandCursor);
    auto* row = new QHBoxLayout(this);
    row->setContentsMargins(18, 3, 0, 3);   // the box is painted in the first 18 px
    row->setSpacing(6);
    label_ = new QLabel(label, this);
    label_->setProperty("role", "check-lab");
    hint_ = new QLabel(hint, this);
    hint_->setObjectName(hint_id);
    hint_->setProperty("role", "check-hint");
    row->addWidget(label_);
    row->addStretch(1);
    row->addWidget(hint_);
}

void CheckRow::set_on(bool on) {
    on_ = on;
    setProperty("on", on);
    repolish(this);
}

void CheckRow::set_hint(const QString& h) { hint_->setText(h); }

void CheckRow::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && clicked) clicked();
}

void CheckRow::paintEvent(QPaintEvent*) {
    QStyleOption opt;
    opt.initFrom(this);
    QPainter p(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
    const QRectF box(2.5, height() / 2.0 - 5.5, 11, 11);
    p.setRenderHint(QPainter::Antialiasing);
    // .check.on .box: accent-dim with the accent line; off: the panel with line-2.
    p.setPen(QPen(palette().color(on_ ? QPalette::Link : QPalette::Midlight), 1));
    p.setBrush(on_ ? palette().color(QPalette::Highlight) : palette().color(QPalette::Base));
    p.drawRoundedRect(box, 1, 1);
    if (on_) {
        QPainterPath tick;   // the page's check mark: M1 3 L2.8 4.8 L6 1.2 in a 7 x 6 box
        const QPointF o = box.center() - QPointF(3.5, 3.0);
        tick.moveTo(o + QPointF(1, 3));
        tick.lineTo(o + QPointF(2.8, 4.8));
        tick.lineTo(o + QPointF(6, 1.2));
        p.setPen(QPen(palette().color(QPalette::HighlightedText), 1.2));
        p.setBrush(Qt::NoBrush);
        p.drawPath(tick);
    }
}

QWidget* panel_label(const QString& text, const QString& note, const QString& note_id, QWidget* parent) {
    auto* w = new QWidget(parent);
    w->setProperty("role", "plabel");
    w->setAttribute(Qt::WA_StyledBackground, true);
    w->setFixedHeight(24);
    auto* row = new QHBoxLayout(w);
    row->setContentsMargins(9, 0, 9, 0);
    row->setSpacing(6);
    auto* t = new QLabel(text, w);
    t->setProperty("role", "plabel-text");
    row->addWidget(t);
    row->addStretch(1);
    if (!note.isNull()) {
        auto* n = new QLabel(note, w);
        n->setProperty("role", "note");
        if (!note_id.isEmpty()) n->setObjectName(note_id);
        row->addWidget(n);
    }
    return w;
}

ScrubBar::ScrubBar(QWidget* parent) : QWidget(parent) {
    setObjectName("scrub");
    setMinimumHeight(14);
    setCursor(Qt::PointingHandCursor);
}

void ScrubBar::set_position(double f) {
    pos_ = std::clamp(f, 0.0, 1.0);
    update();
}

void ScrubBar::paintEvent(QPaintEvent*) {
    QPainter p(this);
    const double y = height() / 2.0;
    p.fillRect(QRectF(0, y - 1.5, width(), 3), palette().color(QPalette::Mid));
    const double x = pos_ * (width() - 2);
    p.fillRect(QRectF(x, 1, 2, height() - 2), palette().color(QPalette::Link));
}

void ScrubBar::mousePressEvent(QMouseEvent* e) {
    if (seek && width() > 0) seek(std::clamp(e->position().x() / width(), 0.0, 1.0));
}

void ScrubBar::mouseMoveEvent(QMouseEvent* e) {
    if ((e->buttons() & Qt::LeftButton) && seek && width() > 0)
        seek(std::clamp(e->position().x() / width(), 0.0, 1.0));
}

IconButton::IconButton(const QString& id, Glyph g, const QString& title, QWidget* parent)
    : QToolButton(parent), glyph_(g) {
    setObjectName(id);
    setToolTip(title);
    setAccessibleName(title);
    setFocusPolicy(Qt::NoFocus);
    setCursor(Qt::PointingHandCursor);
}

void IconButton::set_glyph(Glyph g) {
    glyph_ = g;
    update();
}

void IconButton::set_on(bool on) {
    on_ = on;
    setProperty("on", on);
    repolish(this);
}

void IconButton::paintEvent(QPaintEvent* e) {
    QToolButton::paintEvent(e);   // the sheet's background and border
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    // .ibtn: ink-3, the accent when on (with its 2 px bar), the ink on hover.
    const bool rail = glyph_ == Glyph::Media || glyph_ == Glyph::Scopes || glyph_ == Glyph::Inspector ||
                      glyph_ == Glyph::Help;
    QColor ink = palette().color(QPalette::PlaceholderText);
    if (!rail) ink = palette().color(QPalette::WindowText);
    if (on_) ink = palette().color(QPalette::Link);
    else if (underMouse()) ink = palette().color(QPalette::Text);
    if (!isEnabled()) ink = palette().color(QPalette::Disabled, QPalette::WindowText);
    if (rail && on_) p.fillRect(QRectF(0, 0, 2, height()), palette().color(QPalette::Link));
    p.setPen(QPen(ink, 1.3, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);
    auto box = [&](double w, double h) {   // centre a w x h drawing
        return QPointF((width() - w) / 2.0, (height() - h) / 2.0);
    };
    switch (glyph_) {
        case Glyph::Media: {   // 24-unit box scaled to 16 px
            p.translate(box(16, 16));
            p.scale(16.0 / 24.0, 16.0 / 24.0);
            p.drawRoundedRect(QRectF(3, 5, 18, 14), 2, 2);
            QPainterPath m;
            m.moveTo(3, 15); m.lineTo(8, 11); m.lineTo(12, 14); m.lineTo(15, 12); m.lineTo(21, 16);
            p.drawPath(m);
            break;
        }
        case Glyph::Scopes: {
            p.translate(box(16, 16));
            p.scale(16.0 / 24.0, 16.0 / 24.0);
            QPainterPath m;
            m.moveTo(3, 17); m.lineTo(7, 11); m.lineTo(11, 15); m.lineTo(15, 7); m.lineTo(21, 17);
            p.drawPath(m);
            break;
        }
        case Glyph::Inspector: {
            p.translate(box(16, 16));
            p.scale(16.0 / 24.0, 16.0 / 24.0);
            p.drawLine(QPointF(4, 6), QPointF(20, 6));
            p.drawLine(QPointF(4, 12), QPointF(14, 12));
            p.drawLine(QPointF(4, 18), QPointF(17, 18));
            break;
        }
        case Glyph::Help: {
            p.translate(box(16, 16));
            p.scale(16.0 / 24.0, 16.0 / 24.0);
            p.drawEllipse(QPointF(12, 12), 8, 8);
            QPainterPath q;
            q.moveTo(9.6, 9.4);
            q.arcTo(QRectF(9.6, 7.0, 5.0, 5.0), 180, -250);
            q.lineTo(13, 13);
            p.drawPath(q);
            p.setBrush(ink);
            p.drawEllipse(QPointF(12, 16.6), 0.6, 0.6);
            break;
        }
        case Glyph::Prev:
        case Glyph::Next:
        case Glyph::Play: {   // the page's 9 x 10 triangles
            p.translate(box(9, 10));
            QPainterPath t;
            if (glyph_ == Glyph::Prev) {
                t.moveTo(8, 1); t.lineTo(1, 5); t.lineTo(8, 9); t.closeSubpath();
            } else {
                t.moveTo(1, 1); t.lineTo(8, 5); t.lineTo(1, 9); t.closeSubpath();
            }
            p.setPen(QPen(ink, 1));
            if (glyph_ == Glyph::Play) p.setBrush(ink);
            p.drawPath(t);
            break;
        }
        case Glyph::Pause: {
            p.translate(box(9, 10));
            p.setPen(Qt::NoPen);
            p.setBrush(ink);
            p.drawRect(QRectF(1, 1, 2.5, 8));
            p.drawRect(QRectF(5.5, 1, 2.5, 8));
            break;
        }
    }
}

}  // namespace rudra::app
