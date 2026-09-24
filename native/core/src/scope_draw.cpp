#include "rudra/core/scope_draw.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "rudra/core/js_format.hpp"

namespace rudra {
namespace {

constexpr double kDW = 203.0;   // DIFFUSE_WHITE
constexpr const char* kShadow = "#4d8fd6";
constexpr const char* kMid = "#9fb0c0";
constexpr const char* kHi = "#e8b07a";
constexpr const char* kHiBright = "#f8d8b8";
constexpr const char* kMono = "IBM Plex Mono, monospace";

std::string f1(double v) { return js_to_fixed(v, 1); }
std::string num(double v) { return js_number(v); }

double nits_to_y(double n) {
    const double span = std::log10(kScopeHi / kScopeLo);
    const double v = std::min(std::max(n, kScopeLo), kScopeHi);
    return kWaveH - (std::log10(v / kScopeLo) / span) * kWaveH;
}

SvgElement el(std::string tag, std::vector<std::pair<std::string, std::string>> attrs, std::string text = {}) {
    SvgElement e;
    e.tag = std::move(tag);
    e.attrs = std::move(attrs);
    e.text = std::move(text);
    return e;
}

std::string join(const std::vector<std::string>& v) {
    std::string o;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) o += ' ';
        o += v[i];
    }
    return o;
}

}  // namespace

const std::string* SvgElement::attr(const std::string& name) const {
    for (const auto& [k, v] : attrs)
        if (k == name) return &v;
    return nullptr;
}

SvgDrawing waveform_svg(const ScopeData& s, std::optional<double> maxcll) {
    SvgDrawing d;
    d.view_box = "0 -9 " + num(kWaveW + 30) + " " + num(kWaveH + 18);
    auto& out = d.elements;
    const std::size_t n = s.mid.size();
    const double step = double(kWaveW) / double(n);

    // gridlines first, so the trace sits on top of them
    const std::pair<double, const char*> grid[] = {{0.05, "0.05"}, {1, "1"}, {10, "10"},
                                                   {100, "100"},  {1000, "1k"}, {4000, "4k"}};
    for (const auto& [nits, lab] : grid) {
        const double y = nits_to_y(nits);
        out.push_back(el("line", {{"x1", "0"}, {"y1", f1(y)}, {"x2", num(kWaveW)}, {"y2", f1(y)},
                                  {"stroke", "#1e2c3a"}, {"stroke-width", "0.6"}}));
        out.push_back(el("text", {{"x", num(kWaveW + 3)}, {"y", f1(y + 3)}, {"font-family", kMono},
                                  {"font-size", "8"}, {"fill", "#5b6b7a"}},
                         lab));
    }
    const double dw_y = nits_to_y(kDW);
    out.push_back(el("line", {{"x1", "0"}, {"y1", f1(dw_y)}, {"x2", num(kWaveW)}, {"y2", f1(dw_y)},
                              {"stroke", "#6f7f8f"}, {"stroke-width", "0.8"}, {"stroke-dasharray", "3 3"},
                              {"opacity", "0.8"}}));
    out.push_back(el("text", {{"x", "3"}, {"y", f1(dw_y - 4)}, {"font-family", kMono}, {"font-size", "8"},
                              {"fill", "#8fa2b4"}},
                     "203 diffuse"));

    // the envelope as filled bands
    std::vector<std::string> top, bot, q3, q1, spine;
    for (std::size_t i = 0; i < n; ++i) {
        const std::string x = f1(double(i) * step + 0.5);
        top.push_back(x + "," + f1((1 - s.hi[i]) * kWaveH));
        bot.push_back(x + "," + f1((1 - s.lo[i]) * kWaveH));
        q3.push_back(x + "," + f1((1 - s.q3[i]) * kWaveH));
        q1.push_back(x + "," + f1((1 - s.q1[i]) * kWaveH));
        spine.push_back(x + "," + f1((1 - s.mid[i]) * kWaveH));
    }
    std::reverse(bot.begin(), bot.end());
    std::reverse(q1.begin(), q1.end());
    out.push_back(el("polygon", {{"points", join(top) + " " + join(bot)}, {"fill", kMid}, {"opacity", "0.13"}}));
    out.push_back(el("polygon", {{"points", join(q3) + " " + join(q1)}, {"fill", kMid}, {"opacity", "0.30"}}));
    out.push_back(el("polyline", {{"points", join(spine)}, {"fill", "none"}, {"stroke", kHiBright},
                                  {"stroke-width", "1.1"}, {"opacity", "0.92"}}));

    // how much of the frame is sitting on the ceiling
    const double clip_y = nits_to_y(kScopeHi);
    int clipped = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (s.hi[i] >= 0.999) ++clipped;
    if (clipped) {
        out.push_back(el("rect", {{"x", "0"}, {"y", "0"}, {"width", num(kWaveW)}, {"height", f1(clip_y + 3)},
                                  {"fill", kHi}, {"opacity", "0.10"}}));
        out.push_back(el("line", {{"x1", "0"}, {"y1", f1(clip_y)}, {"x2", num(kWaveW)}, {"y2", f1(clip_y)},
                                  {"stroke", kHi}, {"stroke-width", "1"}}));
        out.push_back(el("text", {{"x", "3"}, {"y", f1(clip_y + 9)}, {"font-family", kMono}, {"font-size", "8"},
                                  {"fill", kHi}},
                         "at ceiling  " + f1(double(clipped) * 100.0 / double(n)) + "% of columns"));
    }

    const double span = std::log10(kScopeHi / kScopeLo);
    if (maxcll && std::isfinite(*maxcll)) {
        const double y =
            kWaveH - (std::log10(std::min(std::max(*maxcll, kScopeLo), kScopeHi) / kScopeLo) / span) * kWaveH;
        out.push_back(el("line", {{"x1", "0"}, {"y1", f1(y)}, {"x2", num(kWaveW)}, {"y2", f1(y)},
                                  {"stroke", "#cfcfcf"}, {"stroke-width", "1"}, {"stroke-dasharray", "2 3"},
                                  {"opacity", "0.55"}}));
        out.push_back(el("text", {{"x", "4"}, {"y", f1(y - 4)}, {"font-family", kMono}, {"font-size", "8.5"},
                                  {"fill", "#a8a8a8"}},
                         "MaxCLL " + num(js_round(*maxcll)) + " nits"));
    } else {
        const double y = kWaveH - (std::log10(kDW / kScopeLo) / span) * kWaveH;
        out.push_back(el("line", {{"x1", "0"}, {"y1", f1(y)}, {"x2", num(kWaveW)}, {"y2", f1(y)},
                                  {"stroke", "#777"}, {"stroke-width", "1"}, {"stroke-dasharray", "4 3"},
                                  {"opacity", "0.75"}}));
        out.push_back(el("text", {{"x", "4"}, {"y", f1(y - 4)}, {"font-family", kMono}, {"font-size", "8.5"},
                                  {"fill", "#888"}},
                         "Diffuse white 203"));
    }
    return d;
}

SvgDrawing histogram_svg(const ScopeData& s) {
    SvgDrawing d;
    d.view_box = "0 0 310 116";
    const std::size_t bins = s.histogram.size();
    const double bw = double(kHistW) / double(bins);
    const double lo2 = std::log2(kScopeLo), span2 = std::log2(kScopeHi / kScopeLo);
    SvgElement g = el("g", {{"fill", "#b4b4b4"}, {"opacity", "0.78"}});
    for (std::size_t j = 0; j < bins; ++j) {
        const double v = s.histogram[j];
        if (v <= 0.004) continue;
        const double stops = (lo2 + (double(j) + 0.5) / double(bins) * span2) - std::log2(kDW);
        const char* c = stops < -4 ? kShadow : (stops > 2 ? kHi : kMid);
        const double h = std::max(1.0, v * kHistH);
        g.children.push_back(el("rect", {{"x", js_to_fixed(double(j) * bw, 2)}, {"y", f1(kHistH - h)},
                                         {"width", js_to_fixed(bw * 0.78, 2)}, {"height", f1(h)}, {"fill", c},
                                         {"opacity", "0.88"}}));
    }
    d.elements.push_back(std::move(g));
    const double dw = (std::log2(kDW / kScopeLo) / std::log2(kScopeHi / kScopeLo)) * kHistW;
    d.elements.push_back(el("line", {{"x1", "0"}, {"y1", num(kHistH)}, {"x2", num(kHistW)}, {"y2", num(kHistH)},
                                     {"stroke", "#2c2c2c"}}));
    d.elements.push_back(el("line", {{"x1", f1(dw)}, {"y1", "0"}, {"x2", f1(dw)}, {"y2", num(kHistH)},
                                     {"stroke", "#4a4a4a"}, {"stroke-dasharray", "2 3"}}));
    d.elements.push_back(el("text", {{"x", f1(dw + 3)}, {"y", "10"}, {"font-family", kMono}, {"font-size", "8.5"},
                                     {"fill", "#777"}},
                            "DW"));
    const std::pair<double, const char*> marks[] = {{0.05, "0.05"}, {1, "1"}, {10, "10"},
                                                    {203, "203"},  {1000, "1k"}, {4000, "4k"}};
    for (const auto& [nits, lab] : marks) {
        const double mx = (std::log2(nits / kScopeLo) / span2) * kHistW;
        d.elements.push_back(el("text", {{"x", js_to_fixed(mx, 0)}, {"y", "110"}, {"font-family", kMono},
                                         {"font-size", "8"}, {"fill", "#5b6b7a"}, {"text-anchor", "middle"}},
                                lab));
    }
    return d;
}

std::optional<Rgb8> parse_colour(const std::string& s) {
    if (s.size() != 4 && s.size() != 7) return std::nullopt;
    if (s[0] != '#') return std::nullopt;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    int v[6];
    const bool shortform = s.size() == 4;
    for (int i = 0; i < 6; ++i) {
        v[i] = hex(s[std::size_t(shortform ? 1 + i / 2 : 1 + i)]);
        if (v[i] < 0) return std::nullopt;
    }
    return Rgb8{std::uint8_t(v[0] * 16 + v[1]), std::uint8_t(v[2] * 16 + v[3]), std::uint8_t(v[4] * 16 + v[5])};
}

const VectorFrame& vector_frame() {
    static const VectorFrame f = [] {
        VectorFrame v;
        v.labels = {{"R", 0.50, 0.07}, {"MG", 0.88, 0.285}, {"B", 0.88, 0.715},
                    {"CY", 0.50, 0.93}, {"G", 0.12, 0.715}, {"YL", 0.12, 0.285}};
        return v;
    }();
    return f;
}

}  // namespace rudra
