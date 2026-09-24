#include "rudra/engine/session.hpp"

#include "rudra/engine/actions.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string>

namespace rudra {
namespace {

constexpr double kPageDiffuseWhite = 203.0;   // DIFFUSE_WHITE in app.js
constexpr std::size_t kUndoDepth = 60;   // pushUndo keeps the last 60

// Math.round: the nearest integer, halves up (toward +infinity).
double js_round(double x) { return std::floor(x + 0.5); }

// Math.pow(2, e) for the peak slider's steps. The slider moves in halves, and
// 2^(n/2) is a power of two times sqrt(2) for odd n: exact, where a library
// pow may differ from V8's in the last place.
double pow2(double e) {
    const double twice = e * 2.0;
    if (twice == std::floor(twice) && std::abs(twice) < 2048.0) {
        const long long n = static_cast<long long>(twice);
        const long long whole = n >= 0 ? n / 2 : -((-n + 1) / 2);
        const bool half = (n - 2 * whole) != 0;
        return std::ldexp(half ? std::sqrt(2.0) : 1.0, int(whole));
    }
    return std::pow(2.0, e);
}

void append_string(std::string& out, std::string_view s) {
    out += '"';
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += c;
        }
    }
    out += '"';
}

RecoveryMode recovery(std::string_view m) {
    if (m == "highlights") return RecoveryMode::Highlights;
    if (m == "shadows") return RecoveryMode::Shadows;
    if (m == "off") return RecoveryMode::Off;
    return RecoveryMode::All;
}

}  // namespace

std::vector<RegionState> default_regions() {
    return {{"highlights", 400.0, 2000.0, 0.0}, {"speculars", 2000.0, 8000.0, 0.0}, {"shadows", 0.05, 12.0, 0.0}};
}

std::string js_number(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return "0";   // -0 prints as 0 too
    // The shortest round-trip digits, as d.ddde±x.
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string sci(buf, r.ptr);
    const bool neg = sci[0] == '-';
    if (neg) sci.erase(0, 1);
    const auto e = sci.find('e');
    const int exp = std::atoi(sci.c_str() + e + 1);
    std::string digits;
    for (std::size_t i = 0; i < e; ++i)
        if (sci[i] != '.') digits += sci[i];
    const int k = int(digits.size());
    const int n = exp + 1;   // the decimal point sits after n digits
    std::string out;
    if (k <= n && n <= 21) {
        out = digits + std::string(std::size_t(n - k), '0');
    } else if (0 < n && n <= 21) {
        out = digits.substr(0, std::size_t(n)) + "." + digits.substr(std::size_t(n));
    } else if (-6 < n && n <= 0) {
        out = "0." + std::string(std::size_t(-n), '0') + digits;
    } else {
        out = digits.substr(0, 1);
        if (k > 1) out += "." + digits.substr(1);
        out += (n - 1 >= 0 ? "e+" : "e-") + std::to_string(std::abs(n - 1));
    }
    return neg ? "-" + out : out;
}

Session::Session() { grade.regions = default_regions(); }

double Session::display_nits() const { return kPageDiffuseWhite * pow2(peak_ev); }

std::string Session::params_json() const {
    std::string o = "{\"strength\":" + js_number(grade.strength) + ",\"display_nits\":" + js_number(display_nits()) +
                    ",\"recovery_mode\":";
    append_string(o, grade.mode);
    o += ",\"preserve_outside\":";
    o += grade.preserve ? "true" : "false";
    o += ",\"regions\":[";
    for (std::size_t i = 0; i < grade.regions.size(); ++i) {
        const auto& r = grade.regions[i];
        if (i) o += ',';
        o += "{\"label\":";
        append_string(o, r.label);
        o += ",\"low_nits\":" + js_number(r.low_nits) + ",\"high_nits\":" + js_number(r.high_nits) +
             ",\"ev\":" + js_number(r.ev) + "}";
    }
    o += "],\"region_softness_stops\":1,\"tile_size\":0,\"tile_overlap\":64,\"max_side\":1600}";
    return o;
}

CompositeParams Session::composite_params() const {
    CompositeParams p;
    p.mode = recovery(grade.mode);
    p.strength = float(grade.strength);
    p.preserve_outside = grade.preserve;
    for (const auto& r : grade.regions) p.regions.push_back({r.low_nits, r.high_nits, r.ev});
    p.region_softness_stops = 1.0;
    return p;
}

void Session::notify(std::uint32_t what) const {
    if (changed_) changed_(what);
}

void Session::push_undo() {
    undo_.push_back(grade);
    while (undo_.size() > kUndoDepth) undo_.erase(undo_.begin());
    redo_.clear();
}

void Session::restore(const GradeSnapshot& s) {
    grade = s;
    notify(Grade);
}

void Session::undo() {
    if (undo_.empty()) return;   // "nothing to undo"
    redo_.push_back(grade);
    GradeSnapshot s = undo_.back();
    undo_.pop_back();
    restore(s);
}

void Session::redo() {
    if (redo_.empty()) return;
    undo_.push_back(grade);
    GradeSnapshot s = redo_.back();
    redo_.pop_back();
    restore(s);
}

void Session::set_mode(std::string_view mode) {
    if (grade.mode == mode) return;
    push_undo();
    grade.mode = std::string(mode);
    notify(Grade);
}

void Session::nudge_strength(double delta) {
    push_undo();
    grade.strength = std::max(0.0, std::min(2.0, js_round((grade.strength + delta) * 100.0) / 100.0));
    notify(Grade);
}

void Session::strength_press() { push_undo(); }

void Session::strength_input(double value) {
    grade.strength = value;
    notify(Grade);
}

void Session::peak_input(double ev) {
    peak_ev = ev;
    notify(Peak);
}

void Session::toggle_preserve() {
    push_undo();
    grade.preserve = !grade.preserve;
    notify(Grade);
}

void Session::reset_recon() {
    push_undo();
    grade.mode = "all";
    grade.strength = 1.0;
    grade.preserve = true;
    notify(Grade);
}

void Session::reset_regions() {
    push_undo();
    grade.regions = default_regions();
    notify(Grade);
}

void Session::set_container(std::string_view kind) {
    container = std::string(kind);
    notify(Delivery);
}

void Session::toggle_wipe() {
    wipe = wipe ? std::nullopt : std::optional<double>(0.5);
    flip_held = false;
    notify(Wipe | Flip);
}

void Session::set_wipe(double x) {
    wipe = std::max(0.0, std::min(1.0, x));
    notify(Wipe);
}

void Session::set_show(std::string_view source) {
    show = std::string(source);
    wipe.reset();
    notify(View | Wipe);
}

void Session::set_view_layer(int layer) {
    view_layer = layer;
    notify(View);
}

void Session::toggle_anchor() {
    anchor = !anchor;
    notify(Delivery);
}

void Session::toggle_carry_chroma() {
    carry_chroma = !carry_chroma;
    notify(Delivery);
}

void Session::region_press(int index, double x) {
    if (index < 0 || index >= int(grade.regions.size())) return;
    region_sel = index;
    drag_ = index;
    drag_x0_ = x;
    drag_ev0_ = grade.regions[std::size_t(index)].ev;
    drag_moved_ = false;
    push_undo();
}

void Session::region_move(double x, bool shift) {
    if (drag_ < 0) return;
    const double step = shift ? 0.002 : 0.01;
    const double value = std::max(-4.0, std::min(4.0, drag_ev0_ + (x - drag_x0_) * step));
    auto& r = grade.regions[std::size_t(drag_)];
    if (std::abs(value - r.ev) < 1e-6) return;
    drag_moved_ = true;
    r.ev = js_round(value * 100.0) / 100.0;
    notify(Grade);
}

void Session::region_release() {
    // A press that never moved leaves no undo step behind.
    if (drag_ >= 0 && !drag_moved_ && !undo_.empty()) undo_.pop_back();
    drag_ = -1;
}

void Session::region_zero(int index) {
    if (index < 0 || index >= int(grade.regions.size())) return;
    push_undo();
    grade.regions[std::size_t(index)].ev = 0.0;
    notify(Grade);
}

bool Session::owns(std::string_view a) {
    static const std::string_view mine[] = {"mode-all", "mode-highlights", "mode-shadows", "mode-off", "preserve",
                                            "strength-down", "strength-up", "reset-recon", "reset-regions", "undo",
                                            "redo", "container-aces", "container-linear", "wipe",
                                            "rail-left", "rail-right", "scopes"};
    return std::find(std::begin(mine), std::end(mine), a) != std::end(mine);
}

bool Session::run(std::string_view a) {
    if (a == "mode-all") set_mode("all");
    else if (a == "mode-highlights") set_mode("highlights");
    else if (a == "mode-shadows") set_mode("shadows");
    else if (a == "mode-off") set_mode("off");
    else if (a == "preserve") toggle_preserve();
    else if (a == "strength-down") nudge_strength(-0.1);
    else if (a == "strength-up") nudge_strength(0.1);
    else if (a == "reset-recon") reset_recon();
    else if (a == "reset-regions") reset_regions();
    else if (a == "undo") undo();
    else if (a == "redo") redo();
    else if (a == "container-aces") set_container("aces");
    else if (a == "container-linear") set_container("linear");
    else if (a == "wipe") toggle_wipe();
    else if (a == "rail-left") {
        rail_left = !rail_left;
        notify(Window);
    } else if (a == "rail-right") {
        rail_right = !rail_right;
        notify(Window);
    } else if (a == "scopes") {
        scopes_open = !scopes_open;
        notify(Window);
    } else return false;
    return true;
}

std::string Session::key_down(std::string_view k, bool shift, bool modified, bool repeat) {
    if (modified) return {};
    if (k == "b" || k == "B") {
        if (!repeat) {
            flip_held = true;
            notify(Flip);
        }
        return {};
    }
    if (k == "w" || k == "W") {
        toggle_wipe();
        return "wipe";
    }
    if (wipe && (k == "ArrowLeft" || k == "ArrowRight")) {
        const double step = shift ? 0.01 : 0.05;
        wipe = std::max(0.0, std::min(1.0, *wipe + (k == "ArrowRight" ? step : -step)));
        notify(Wipe);
        return {};
    }
    if (k == "Escape") {
        if (wipe) {
            wipe.reset();
            notify(Wipe);
            return {};
        }
        return "escape";   // the app closes a sheet or a menu
    }
    // The page's map, Space and 1 to 4: engine/actions has them.
    std::string act(action_for_key(k));
    if (!act.empty()) run(act);
    return act;
}

void Session::key_up(std::string_view k) {
    if ((k == "b" || k == "B") && flip_held) {
        flip_held = false;
        notify(Flip);
    }
}

}  // namespace rudra
