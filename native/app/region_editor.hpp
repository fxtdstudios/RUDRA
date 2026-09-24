#pragma once
// The Region EV editor (Phase 3 step 6): the page's #regions. One row a band:
// its swatch, its range ("400 – 2k nits"), its value ("+0.37", green when not
// zero) and "EV". A press on a row selects it; a drag on the value scrubs it
// (0.01 EV a pixel, 0.002 with Shift, into engine/session's undo); a double
// click zeroes it. drawRegions and bindRegions in ui/app.js.

#include <QLabel>
#include <QWidget>

#include <vector>

#include "rudra/engine/session.hpp"

namespace rudra::app {

class RegionEditor : public QWidget {
public:
    RegionEditor(Session& session, QWidget* parent = nullptr);
    // Redraw from the session (drawRegions).
    void sync();
    // The value label of a row (the page's .ev), for gestures and tests.
    QLabel* value(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].ev : nullptr; }
    QWidget* row(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].row : nullptr; }
    QLabel* range(int i) const { return i >= 0 && i < int(rows_.size()) ? rows_[std::size_t(i)].q : nullptr; }
    int rows() const { return int(rows_.size()); }

protected:
    bool eventFilter(QObject* o, QEvent* e) override;

private:
    struct Row {
        QWidget* row = nullptr;
        QLabel *q = nullptr, *ev = nullptr;
    };
    void build(int n);
    Session& session_;
    std::vector<Row> rows_;
    int dragging_ = -1;
};

}  // namespace rudra::app
