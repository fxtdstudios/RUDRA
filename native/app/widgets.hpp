#pragma once
// The page's small controls as Qt widgets (Phase 3 step 4). Each carries the
// page's id as its object name and the page's class as its "role", so
// app/theme/studio.qss styles it the way ui/theme.css styles the element.

#include <QPushButton>
#include <QToolButton>
#include <QWidget>

#include <functional>
#include <vector>

class QButtonGroup;
class QLabel;

namespace rudra::app {

// .seg: a row of buttons, one of which may be "on".
class Seg : public QWidget {
public:
    Seg(const QString& id, const std::vector<std::pair<QString, QString>>& buttons, QWidget* parent = nullptr,
        bool exclusive = true);
    QPushButton* button(const QString& key) const;   // by its data-* value
    void set_on(const QString& key);                  // the one that is on ("" for none)
    QString on() const;
    std::function<void(const QString&)> clicked;     // with the button's key
    std::vector<QPushButton*> buttons() const { return buttons_; }

private:
    std::vector<QPushButton*> buttons_;
    std::vector<QString> keys_;
};

// .check: the box, the label and the hint; clicking anywhere toggles.
class CheckRow : public QWidget {
public:
    CheckRow(const QString& id, const QString& label, const QString& hint_id, const QString& hint,
             QWidget* parent = nullptr);
    void set_on(bool on);
    bool on() const { return on_; }
    void set_hint(const QString& h);
    QLabel* label() const { return label_; }
    QLabel* hint() const { return hint_; }
    std::function<void()> clicked;

protected:
    void mousePressEvent(class QMouseEvent* e) override;
    void paintEvent(class QPaintEvent* e) override;

private:
    QLabel* label_ = nullptr;
    QLabel* hint_ = nullptr;
    bool on_ = false;
};

// .plabel: a panel label with an optional note on the right.
QWidget* panel_label(const QString& text, const QString& note = {}, const QString& note_id = {},
                     QWidget* parent = nullptr);

// The transport's scrub bar: a track and a head at 0..1; a press or a drag seeks.
class ScrubBar : public QWidget {
public:
    explicit ScrubBar(QWidget* parent = nullptr);
    void set_position(double f);
    double position() const { return pos_; }
    std::function<void(double)> seek;
    QSize sizeHint() const override { return {200, 14}; }

protected:
    void paintEvent(class QPaintEvent* e) override;
    void mousePressEvent(class QMouseEvent* e) override;
    void mouseMoveEvent(class QMouseEvent* e) override;

private:
    double pos_ = 0.0;
};

// The clip bar under the frame stats (paintClipBar): red for what the SDR
// lost, amber after it for where the network acted, as widths of the bar.
class ClipBar : public QWidget {
public:
    explicit ClipBar(QWidget* parent = nullptr);
    void set(double clipped_pct, double mask_pct);
    QWidget* lost() const { return i_; }
    QWidget* acted() const { return u_; }

protected:
    void resizeEvent(class QResizeEvent* e) override;

private:
    void place();
    QWidget *i_ = nullptr, *u_ = nullptr;
    double i_w_ = 0, u_l_ = 0, u_w_ = 0;
};

// The icon rail's buttons and the transport's, drawn as the page's SVGs are.
class IconButton : public QToolButton {
public:
    enum class Glyph { Media, Scopes, Inspector, Help, Prev, Play, Pause, Next };
    IconButton(const QString& id, Glyph g, const QString& title, QWidget* parent = nullptr);
    void set_glyph(Glyph g);
    void set_on(bool on);
    bool on() const { return on_; }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    Glyph glyph_;
    bool on_ = false;
};

}  // namespace rudra::app
