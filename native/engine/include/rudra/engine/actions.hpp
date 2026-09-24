#pragma once
// The app's commands (Phase 3 step 2): the browser Studio's 33 actions, with
// its menus, labels, key hints, check states, keys and shortcut sheet, and the
// few the native app adds (opening a model package, quitting, the viewer's
// modes, guides and view peak, which the page keeps on its toolbar).
//
// The Studio part is a port of ui/index.html's menubar and of ACTIONS, the
// keydown handler, SHORTCUTS and refreshMenu() in ui/app.js, and equals them
// (tests/golden/actions, tools/emit_actions_golden.py). No Qt here: the
// session (step 3) runs these ids, and the app builds its QActions from them.

#include <string_view>
#include <vector>

namespace rudra {

enum class ActionOrigin { Studio, Native };

// When the page enables a menu item (refreshMenu); Always when it has no rule.
enum class EnableRule {
    Always,
    AnyFrames,      // any
    ManyFrames,     // many: more than one frame
    CanMaster,      // any && state.live && !state.busy
    CanUndo,        // state.undo.length > 0
    CanRedo,        // state.redo.length > 0
    HasMetrics,     // !!state.metrics
    HasScopes,      // !!state.scopeData
};

struct ActionSpec {
    std::string_view id;
    std::string_view label;      // the menu text, as the page writes it
    std::string_view hint;       // the key shown in the menu (<i>): "O", "Home", "," ...; empty for none
    std::string_view check;      // the page's data-check: "mode:all", "preserve", "railLeft" ...; empty for none
    ActionOrigin origin = ActionOrigin::Studio;
    EnableRule enable = EnableRule::Always;
    std::string_view native_key; // a native-only shortcut (Qt key-sequence text), for Native actions
};

// Every action once: the Studio's in menubar order, then ACTIONS.wipe, then
// the native ones.
const std::vector<ActionSpec>& action_specs();
const ActionSpec* find_action(std::string_view id);

// The menubar. An entry is an action id, "-" a separator, ">Title" opens a
// submenu and "<" closes it (native menus only). The Studio's menus come in
// the page's order with its entries; native entries are marked by their
// action's origin, and a menu whose title the page does not have is native.
struct MenuSpec {
    std::string_view title;
    std::vector<std::string_view> entries;
};
const std::vector<MenuSpec>& menu_specs();

// The page's keydown: the action a plain key runs ("o", "O", ",", " ", "1",
// "Home" ... as KeyboardEvent.key), or empty. Modified keys (Ctrl, Alt, Meta)
// run nothing; B is held to flip, the arrows nudge a wipe and Escape leaves
// it, which the viewer handles (key_nudge_step).
std::string_view action_for_key(std::string_view key);
// The keys that have an action, in the golden's order (sorted).
std::vector<std::string_view> mapped_keys();
// The wipe nudge: 0.05 of the width per arrow press, 0.01 with Shift.
double key_nudge_step(bool shift);

// The Keyboard sheet (Help > Keyboard shortcuts), row for row.
struct ShortcutRow {
    std::string_view keys, what;
};
const std::vector<ShortcutRow>& shortcut_sheet();

}  // namespace rudra
