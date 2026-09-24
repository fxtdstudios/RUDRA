#include "widgets.hpp"

#include "theme.hpp"

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
    for (const auto& entry : buttons) {
        // Not a structured binding: the click lambda captures the key, which
        // Apple clang 15 cannot do with a binding (C++20 allows it, clang 16+).
        const QString key = entry.first;
        const auto& text = entry.second;
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
    row->setContentsMargins(0, 4, 50, 4);   // the switch is painted in the last 50 px
    row->setSpacing(0);
    auto* text = new QWidget(this);
    auto* col = new QVBoxLayout(text);
    col->setContentsMargins(0, 0, 0, 0);
    col->setSpacing(1);
    label_ = new QLabel(label, text);
    label_->setProperty("role", "check-lab");
    hint_ = new QLabel(hint, text);
    hint_->setObjectName(hint_id);
    hint_->setProperty("role", "check-hint");
    hint_->setWordWrap(true);
    col->addWidget(label_);
    col->addWidget(hint_);
    row->addWidget(text, 1);
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
    p.setRenderHint(QPainter::Antialiasing);
    // A 38 x 22 switch: the accent when on, the raised grey when off; a white knob.
    const QRectF track(width() - 40.0, height() / 2.0 - 11.0, 38.0, 22.0);
    p.setPen(Qt::NoPen);
    p.setBrush(on_ ? palette().color(QPalette::Link) : palette().color(QPalette::Light));
    p.drawRoundedRect(track, 11, 11);
    const double kx = on_ ? track.right() - 20.0 : track.left() + 2.0;
    p.setBrush(palette().color(QPalette::BrightText));
    p.drawEllipse(QRectF(kx, track.top() + 2.0, 18.0, 18.0));
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
    setMinimumHeight(58);
    setCursor(Qt::PointingHandCursor);
}

void ScrubBar::set_position(double f) {
    pos_ = std::clamp(f, 0.0, 1.0);
    update();
}

void ScrubBar::set_count(int n) {
    count_ = std::max(0, n);
    thumbs_.assign(std::size_t(count_), QImage());
    lanes_.assign(std::size_t(count_), Lane::None);
    update();
}

void ScrubBar::set_thumb(int frame, const QImage& small) {
    if (frame < 0 || frame >= count_) return;
    thumbs_[std::size_t(frame)] = small;
    update();
}

void ScrubBar::set_lane(int frame, Lane l) {
    if (frame < 0 || frame >= count_) return;
    lanes_[std::size_t(frame)] = l;
    update();
}

ScrubBar::Lane ScrubBar::lane(int frame) const {
    return frame >= 0 && frame < count_ ? lanes_[std::size_t(frame)] : Lane::None;
}

void ScrubBar::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const double strip_h = std::max(10.0, height() - 14.0), lane_y = strip_h + 8.0;
    const QRectF strip(0, 0, width(), strip_h);
    // The strip: a rounded well, cells of the nearest frame that has a thumbnail.
    QPainterPath well;
    well.addRoundedRect(strip, 6, 6);
    p.fillPath(well, palette().color(QPalette::Dark));
    if (count_ > 0) {
        p.save();
        p.setClipPath(well);
        const int cells = std::max(1, std::min(count_, int(width() / 64.0) + 1));
        const double cw = double(width()) / cells;
        for (int c = 0; c < cells; ++c) {
            const int f = std::min(count_ - 1, int((c + 0.5) * count_ / cells));
            const QImage* img = nullptr;   // this cell's frame, else the nearest with a picture
            for (int d = 0; d < count_ && !img; ++d) {
                for (int g : {f - d, f + d})
                    if (g >= 0 && g < count_ && !thumbs_[std::size_t(g)].isNull()) {
                        img = &thumbs_[std::size_t(g)];
                        break;
                    }
                if (d > count_ / cells) break;   // no further than a cell away
            }
            const QRectF cell(c * cw, 0, cw - (c + 1 < cells ? 2.0 : 0.0), strip_h);
            if (img) {
                // cover: scale to fill, centred
                const double s = std::max(cell.width() / img->width(), cell.height() / img->height());
                const QSizeF sz(img->width() * s, img->height() * s);
                p.drawImage(QRectF(cell.center() - QPointF(sz.width() / 2, sz.height() / 2), sz), *img);
            } else {
                p.fillRect(cell, palette().color(QPalette::Button));
            }
        }
        p.restore();
    }
    // The clipping lane.
    const QRectF lane(0, lane_y, width(), 6);
    p.setPen(Qt::NoPen);
    p.setBrush(palette().color(QPalette::Button));
    p.drawRoundedRect(lane, 3, 3);
    if (count_ > 0) {
        static const QColor lane_gold = theme_colour("gold"), lane_violet = theme_colour("violet");
        for (int f = 0; f < count_; ++f) {
            const Lane l = lanes_[std::size_t(f)];
            if (l == Lane::None) continue;
            const double x0 = double(f) * width() / count_, x1 = double(f + 1) * width() / count_;
            p.setBrush(l == Lane::Highlights ? lane_gold : lane_violet);
            p.drawRect(QRectF(x0, lane.top(), std::max(1.5, x1 - x0), lane.height()));
        }
    }
    // The playhead.
    const double x = count_ > 1 || count_ == 0 ? pos_ * (width() - 2) : 0.0;
    p.fillRect(QRectF(x, 0, 2, strip_h), palette().color(QPalette::Link));
}

void ScrubBar::mousePressEvent(QMouseEvent* e) {
    if (seek && width() > 0) seek(std::clamp(e->position().x() / width(), 0.0, 1.0));
}

void ScrubBar::mouseMoveEvent(QMouseEvent* e) {
    if ((e->buttons() & Qt::LeftButton) && seek && width() > 0)
        seek(std::clamp(e->position().x() / width(), 0.0, 1.0));
}

ClipBar::ClipBar(QWidget* parent) : QWidget(parent) {
    setObjectName("clipBar");
    setAttribute(Qt::WA_StyledBackground, true);
    setFixedHeight(5);
    i_ = new QWidget(this);
    i_->setObjectName("clipLost");
    i_->setAttribute(Qt::WA_StyledBackground, true);
    u_ = new QWidget(this);
    u_->setObjectName("clipActed");
    u_->setAttribute(Qt::WA_StyledBackground, true);
    place();
}

void ClipBar::set(double clipped_pct, double mask_pct) {
    // paintClipBar's CSS: i width min(100, clipped) %, u from there,
    // max(0, min(100 - clipped, mask)) % wide.
    i_w_ = std::min(100.0, clipped_pct);
    u_l_ = i_w_;
    u_w_ = std::max(0.0, std::min(100.0 - clipped_pct, mask_pct));
    place();
}

void ClipBar::resizeEvent(QResizeEvent*) { place(); }

void ClipBar::place() {
    const double w = width();
    i_->setGeometry(0, 0, int(std::lround(w * i_w_ / 100.0)), height());
    u_->setGeometry(int(std::lround(w * u_l_ / 100.0)), 0, int(std::lround(w * u_w_ / 100.0)), height());
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
    // The toolbar's icons: the secondary ink, the accent when on, the ink on hover.
    const bool transport = glyph_ == Glyph::Prev || glyph_ == Glyph::Play || glyph_ == Glyph::Pause || glyph_ == Glyph::Next;
    QColor ink = transport ? palette().color(QPalette::WindowText) : theme_colour("ink-2");
    if (on_) ink = palette().color(QPalette::Link);
    else if (underMouse() && !transport) ink = palette().color(QPalette::Text);
    // The play button is the white disc of the boards: its glyph is dark.
    if (glyph_ == Glyph::Play || glyph_ == Glyph::Pause) ink = objectName() == "btnPlay" ? theme_colour("panel") : ink;
    if (!isEnabled()) ink = palette().color(QPalette::Disabled, QPalette::WindowText);
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
        case Glyph::Sidebar: {   // the boards' 18 x 14 sidebar glyph
            p.translate(box(18, 14));
            p.drawRoundedRect(QRectF(0.7, 0.7, 16.6, 12.6), 2.5, 2.5);
            p.drawLine(QPointF(6, 1), QPointF(6, 13));
            break;
        }
        case Glyph::Probe: {   // a reticle
            p.translate(box(16, 16));
            p.drawEllipse(QPointF(8, 8), 5.5, 5.5);
            p.drawLine(QPointF(8, 0), QPointF(8, 4));
            p.drawLine(QPointF(8, 12), QPointF(8, 16));
            p.drawLine(QPointF(0, 8), QPointF(4, 8));
            p.drawLine(QPointF(12, 8), QPointF(16, 8));
            break;
        }
        case Glyph::Export: {   // share: an arrow out of a tray
            p.translate(box(16, 16));
            QPainterPath m;
            m.moveTo(8, 10); m.lineTo(8, 1);
            m.moveTo(4.5, 4.5); m.lineTo(8, 1); m.lineTo(11.5, 4.5);
            m.moveTo(2, 8); m.lineTo(2, 14); m.lineTo(14, 14); m.lineTo(14, 8);
            p.drawPath(m);
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
