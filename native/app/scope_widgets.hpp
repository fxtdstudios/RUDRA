#pragma once
// The scope widgets (Phase 3 step 5): the waveform and the histogram paint
// the SVG core/scope_draw builds (the page's drawScopes, element for element),
// scaled as the browser scales an <svg> with its viewBox and xMidYMid meet;
// the vectorscope shows core/scopes' 256 x 256 picture 176 px across in the
// page's ring.

#include <QImage>
#include <QWidget>

#include <optional>

#include "rudra/core/scope_draw.hpp"

namespace rudra::app {

class ScopePlot : public QWidget {
public:
    enum class Kind { Waveform, Histogram };
    ScopePlot(const QString& id, Kind kind, int height, QWidget* parent = nullptr);
    void set_data(const ScopeData& s, std::optional<double> maxcll);
    void clear();
    const SvgDrawing& drawing() const { return drawing_; }
    bool has_data() const { return !drawing_.elements.empty(); }
    void set_text_shown(bool on) { text_ = on; update(); }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    Kind kind_;
    SvgDrawing drawing_;
    bool text_ = true;
};

class VectorscopeView : public QWidget {
public:
    explicit VectorscopeView(QWidget* parent = nullptr);
    // core/scopes vectorscope(): 256 x 256 RGBA8, straight alpha.
    void set_image(const std::vector<std::uint8_t>& rgba);
    void clear();
    QSize sizeHint() const override { return {256, 190}; }

protected:
    void paintEvent(class QPaintEvent* e) override;

private:
    QImage image_;
};

// Paints one drawing into `target` as an <svg> of that box would show it;
// without its <text> elements when `text` is false (the tests).
void paint_svg(class QPainter& p, const SvgDrawing& d, const QRectF& target, bool text = true);

}  // namespace rudra::app
