#include "scope_widgets.hpp"

#include <QFontDatabase>
#include <QPainter>
#include <QPainterPath>
#include <QStyle>
#include <QStyleOption>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace rudra::app {
namespace {

double number(const SvgElement& e, const char* name, double fallback = 0.0) {
    const std::string* v = e.attr(name);
    return v ? std::strtod(v->c_str(), nullptr) : fallback;
}

std::vector<QPointF> points(const std::string& s) {
    std::vector<QPointF> out;
    std::istringstream in(s);
    std::string pair;
    while (in >> pair) {
        const auto comma = pair.find(',');
        if (comma == std::string::npos) continue;
        out.emplace_back(std::strtod(pair.c_str(), nullptr), std::strtod(pair.c_str() + comma + 1, nullptr));
    }
    return out;
}

QColor colour(const std::string* s, double opacity) {
    if (!s) return {};
    const auto c = parse_colour(*s);
    if (!c) return {};
    QColor q(c->r, c->g, c->b);
    q.setAlphaF(float(std::clamp(opacity, 0.0, 1.0)));
    return q;
}

// One element, with the opacity of the groups around it and the fill a group
// hands down (SVG's inheritance, for the attributes the page uses).
void paint(QPainter& p, const SvgElement& e, double group_opacity, const std::string* inherited_fill, bool text) {
    const double opacity = number(e, "opacity", 1.0) * group_opacity;
    const std::string* fill = e.attr("fill") ? e.attr("fill") : inherited_fill;
    if (e.tag == "g") {
        for (const auto& c : e.children) paint(p, c, opacity, fill, text);
        return;
    }
    QPen pen(Qt::NoPen);
    if (const std::string* stroke = e.attr("stroke"); stroke && *stroke != "none") {
        const double w = number(e, "stroke-width", 1.0);
        // A stroke under one device pixel is drawn as the browser draws it
        // (Skia's thin-stroke rule): one pixel wide, its alpha scaled by the
        // width it should have had.
        const double scale = std::sqrt(std::abs(p.transform().determinant()));
        const double device = w * scale;
        QColor c = colour(stroke, opacity);
        double pen_w = w;
        if (device < 1.0 && device > 0.0) {
            c.setAlphaF(float(c.alphaF() * device));
            pen_w = 1.0 / scale;
        }
        pen = QPen(c, pen_w);
        pen.setCapStyle(Qt::FlatCap);
        if (const std::string* dash = e.attr("stroke-dasharray")) {
            // SVG dashes are in user units; Qt's are in pen widths.
            QList<qreal> pattern;
            std::istringstream in(*dash);
            double d;
            while (in >> d) pattern << d / std::max(pen_w, 1e-6);
            if (pattern.size() % 2) pattern << pattern;
            pen.setDashPattern(pattern);
        }
    }
    const QBrush brush = fill && *fill != "none" ? QBrush(colour(fill, opacity)) : QBrush(Qt::NoBrush);
    if (e.tag == "line") {
        p.setPen(pen);
        p.drawLine(QPointF(number(e, "x1"), number(e, "y1")), QPointF(number(e, "x2"), number(e, "y2")));
    } else if (e.tag == "polyline" || e.tag == "polygon") {
        const auto pts = points(*e.attr("points"));
        QPolygonF poly(QList<QPointF>(pts.begin(), pts.end()));
        p.setPen(pen);
        p.setBrush(brush);
        if (e.tag == "polygon") p.drawPolygon(poly, Qt::OddEvenFill);
        else p.drawPolyline(poly);
    } else if (e.tag == "rect") {
        p.setPen(pen);
        p.setBrush(brush);
        p.drawRect(QRectF(number(e, "x"), number(e, "y"), number(e, "width"), number(e, "height")));
    } else if (e.tag == "text" && text) {
        // SVG's default white space: runs of spaces collapse to one.
        const QString t = QString::fromStdString(e.text).simplified();
        // At the size it lands on the screen, fractional and unhinted, as
        // the browser sets small text (a 1 px font scaled up has the wrong
        // advances).
        const double scale = std::sqrt(std::abs(p.transform().determinant()));
        const QPointF at = p.transform().map(QPointF(number(e, "x"), number(e, "y")));
        QFont f("IBM Plex Mono");
        f.setHintingPreference(QFont::PreferNoHinting);
        f.setPointSizeF(number(e, "font-size", 8.0) * scale * 72.0 / 96.0);   // points at 96 dpi
        p.save();
        p.resetTransform();
        p.setFont(f);
        p.setPen(colour(fill, opacity));
        const QFontMetricsF fm(f, p.device());
        double x = at.x();
        if (const std::string* a = e.attr("text-anchor"); a && *a == "middle") x -= fm.horizontalAdvance(t) / 2.0;
        p.drawText(QPointF(x, at.y()), t);
        p.restore();
    }
}

}  // namespace

void paint_svg(QPainter& p, const SvgDrawing& d, const QRectF& target, bool text) {
    std::istringstream in(d.view_box);
    double vx = 0, vy = 0, vw = 1, vh = 1;
    in >> vx >> vy >> vw >> vh;
    // preserveAspectRatio xMidYMid meet: the whole box, centred.
    const double s = std::min(target.width() / vw, target.height() / vh);
    const double ox = target.x() + (target.width() - vw * s) / 2.0 - vx * s;
    const double oy = target.y() + (target.height() - vh * s) / 2.0 - vy * s;
    p.save();
    p.setClipRect(target);
    p.translate(ox, oy);
    p.scale(s, s);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::TextAntialiasing);
    for (const auto& e : d.elements) paint(p, e, 1.0, nullptr, text);
    p.restore();
}

ScopePlot::ScopePlot(const QString& id, Kind kind, int height, QWidget* parent) : QWidget(parent), kind_(kind) {
    setObjectName(id);
    setProperty("role", "plot-area");
    setAttribute(Qt::WA_StyledBackground, true);
    setFixedHeight(height);
}

void ScopePlot::set_data(const ScopeData& s, std::optional<double> maxcll) {
    drawing_ = kind_ == Kind::Waveform ? waveform_svg(s, maxcll) : histogram_svg(s);
    update();
}

void ScopePlot::clear() {
    drawing_ = {};
    update();
}

void ScopePlot::paintEvent(QPaintEvent*) {
    QPainter p(this);
    QStyleOption opt;
    opt.initFrom(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);   // the sheet's background and border
    if (drawing_.elements.empty()) return;
    paint_svg(p, drawing_, QRectF(rect()).adjusted(1, 1, -1, -1), text_);
}

VectorscopeView::VectorscopeView(QWidget* parent) : QWidget(parent) {
    setObjectName("vector");
    setProperty("role", "plot");   // it is .plot.vs: the plot's own background
    setAttribute(Qt::WA_StyledBackground, true);
    // .plot.vs: 8 px above the ring and 6 below, the labels placed in this box.
    setFixedHeight(8 + vector_frame().display + 6);
}

void VectorscopeView::set_image(const std::vector<std::uint8_t>& rgba) {
    if (rgba.size() != std::size_t(kVectorSize) * kVectorSize * 4) return;
    image_ = QImage(rgba.data(), kVectorSize, kVectorSize, QImage::Format_RGBA8888).copy();
    update();
}

void VectorscopeView::clear() {
    image_ = {};
    update();
}

void VectorscopeView::paintEvent(QPaintEvent*) {
    const auto& f = vector_frame();
    QPainter p(this);
    QStyleOption opt;
    opt.initFrom(this);
    style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    const double d = f.display;
    // Centred and snapped to whole pixels, as the browser lays the canvas out
    // (the page's 47.5 px lands on 48).
    const QRectF box(std::floor((width() - d) / 2.0 + 0.5), 8.0, d, d);
    auto rgb = [](const Rgb8& c) { return QColor(c.r, c.g, c.b); };
    // The canvas: a disc of the background, the picture over it.
    QPainterPath disc;
    disc.addEllipse(box);
    p.fillPath(disc, rgb(f.background));
    if (!image_.isNull()) {
        p.save();
        p.setClipPath(disc);
        p.drawImage(box, image_);
        p.restore();
    }
    // The rings: 1 px borders, the inner one inset 27 % of the inside.
    p.setBrush(Qt::NoBrush);
    p.setPen(QPen(rgb(f.ring), 1));
    p.drawEllipse(box.adjusted(0.5, 0.5, -0.5, -0.5));
    const double inset = f.inner_inset * (d - 2);
    p.setPen(QPen(rgb(f.inner_ring), 1));
    p.drawEllipse(box.adjusted(1 + inset + 0.5, 1 + inset + 0.5, -1 - inset - 0.5, -1 - inset - 0.5));
    // The hue labels, centred on their places.
    QFont font("IBM Plex Mono");
    font.setPixelSize(int(std::lround(f.label_px)));
    p.setFont(font);
    p.setPen(rgb(f.label));
    const QFontMetricsF fm(font);
    for (const auto& l : f.labels) {
        const QString t = QString::fromLatin1(l.text);
        const QPointF c(l.left * width(), l.top * height());   // left and top of .plot.vs
        p.drawText(QPointF(c.x() - fm.horizontalAdvance(t) / 2.0, c.y() + (fm.ascent() - fm.descent()) / 2.0), t);
    }
}

}  // namespace rudra::app
