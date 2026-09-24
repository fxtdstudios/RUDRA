#pragma once
// The session (Phase 3 step 3): the Studio page's grading state in C++, with
// its undo and redo, and the handlers that change it -- a port of `state`,
// params(), snapshot(), pushUndo(), undo(), redo() and the handlers of the
// mode buttons, the strength and peak sliders, Preserve, the Region EV drag
// and double click, the reset actions, the wipe keys and the keydown map in
// ui/app.js. After the same gestures it gives the same params() JSON, byte
// for byte, and the same undo and redo depths (tests/golden/session, from
// the page itself in headless Chromium, tools/emit_session_golden.py).
//
// No Qt: the app's widgets call these and draw from the state.

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rudra/core/composite.hpp"

namespace rudra {

struct RegionState {
    std::string label;
    double low_nits = 0.0, high_nits = 0.0, ev = 0.0;
};

// The part of the state undo takes back (snapshot()).
struct GradeSnapshot {
    std::string mode = "all";
    double strength = 1.0;
    bool preserve = true;
    std::vector<RegionState> regions;
};

// The page's defaultRegions(): highlights 400 to 2 000, speculars 2 000 to
// 8 000, shadows 0.05 to 12 nits, all at 0 EV.
std::vector<RegionState> default_regions();

// A number as JavaScript prints it (Number.prototype.toString): the shortest
// digits that read back to the same double, fixed between 1e-7 and 1e21.
std::string js_number(double v);

class Session {
public:
    Session();

    // ---- the state (the page's names) -------------------------------------
    GradeSnapshot grade;              // mode, strength, preserve, regions
    double peak_ev = 0.0;             // the view peak slider: 203 * 2^peak_ev nits
    bool anchor = true, carry_chroma = true;
    std::string container = "aces";   // "aces" or "linear"
    std::optional<double> wipe;       // off, or 0..1 across the plate
    bool flip_held = false;
    int region_sel = -1;

    double display_nits() const;
    // JSON.stringify(params()), the settings a render and a master take.
    std::string params_json() const;
    // The composite the viewer draws with these settings.
    CompositeParams composite_params() const;

    std::size_t undo_depth() const { return undo_.size(); }
    std::size_t redo_depth() const { return redo_.size(); }

    // Called after every change, with what changed.
    enum Change : std::uint32_t { Grade = 1, Peak = 2, Wipe = 4, Delivery = 8, Flip = 16 };
    void on_change(std::function<void(std::uint32_t)> cb) { changed_ = std::move(cb); }

    // ---- actions (engine/actions ids) --------------------------------------
    // Runs one of the session's actions (the modes, preserve, strength,
    // resets, undo, redo, containers, wipe); false for any other id.
    bool run(std::string_view action);
    static bool owns(std::string_view action);

    // ---- the page's handlers ----------------------------------------------
    void set_mode(std::string_view mode);        // setMode: no-op when unchanged
    void nudge_strength(double delta);           // nudgeStrength
    void strength_press();                       // the slider's pointerdown: pushUndo
    void strength_input(double value);           // its input event
    void peak_input(double peak_ev);             // the peak slider: not undone
    void toggle_preserve();                      // the Preserve check
    void reset_recon();
    void reset_regions();
    void undo();
    void redo();
    void set_container(std::string_view kind);
    void toggle_wipe();                          // ACTIONS.wipe
    void set_wipe(double x);                     // a drag on the plate, clamped 0..1

    // A Region EV value: press, move, release (the pointer's x in pixels),
    // and the double click that zeroes it.
    void region_press(int index, double x);
    void region_move(double x, bool shift);
    void region_release();
    void region_zero(int index);

    // The window's keydown and keyup (KeyboardEvent.key). Returns the action
    // it ran, or the one the app must run ("open", "play", "prev" ... which
    // are not the session's), or empty. Modified keys do nothing.
    std::string key_down(std::string_view key, bool shift, bool modified = false, bool repeat = false);
    void key_up(std::string_view key);

private:
    void push_undo();
    void restore(const GradeSnapshot& s);
    void notify(std::uint32_t what) const;

    std::vector<GradeSnapshot> undo_, redo_;
    std::function<void(std::uint32_t)> changed_;
    // The Region EV drag.
    int drag_ = -1;
    double drag_x0_ = 0.0, drag_ev0_ = 0.0;
    bool drag_moved_ = false;
};

}  // namespace rudra
