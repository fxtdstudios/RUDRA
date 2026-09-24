#include "rudra/engine/actions.hpp"

#include <algorithm>
#include <map>
#include <string>

namespace rudra {
namespace {

using O = ActionOrigin;
using E = EnableRule;

const std::vector<ActionSpec> kActions = {
    // File
    {"open", "Open frames…", "O", "", O::Studio, E::Always, ""},
    {"close", "Close all frames", "", "", O::Studio, E::AnyFrames, ""},
    {"master", "Master EXR", "M", "", O::Studio, E::CanMaster, ""},
    // Edit
    {"undo", "Undo", "Z", "", O::Studio, E::CanUndo, ""},
    {"redo", "Redo", "Y", "", O::Studio, E::CanRedo, ""},
    {"reset-recon", "Reset reconstruction", "", "", O::Studio, E::Always, ""},
    {"reset-regions", "Reset region EV", "", "", O::Studio, E::Always, ""},
    // Clip
    {"first", "First frame", "Home", "", O::Studio, E::ManyFrames, ""},
    {"prev", "Previous frame", ",", "", O::Studio, E::ManyFrames, ""},
    {"next", "Next frame", ".", "", O::Studio, E::ManyFrames, ""},
    {"last", "Last frame", "End", "", O::Studio, E::ManyFrames, ""},
    {"play", "Play / pause", "Space", "", O::Studio, E::ManyFrames, ""},
    // Reconstruct
    {"mode-all", "All", "", "mode:all", O::Studio, E::Always, ""},
    {"mode-highlights", "Highlights", "", "mode:highlights", O::Studio, E::Always, ""},
    {"mode-shadows", "Shadows", "", "mode:shadows", O::Studio, E::Always, ""},
    {"mode-off", "Off", "", "mode:off", O::Studio, E::Always, ""},
    {"preserve", "Preserve outside masks", "", "preserve", O::Studio, E::Always, ""},
    {"strength-down", "Weaker residual", "[", "", O::Studio, E::Always, ""},
    {"strength-up", "Stronger residual", "]", "", O::Studio, E::Always, ""},
    // Measure
    {"copy-metrics", "Copy measurements as JSON", "", "", O::Studio, E::HasMetrics, ""},
    {"copy-scopes", "Copy scope data as JSON", "", "", O::Studio, E::HasScopes, ""},
    {"remeasure", "Re-measure this frame", "", "", O::Studio, E::AnyFrames, ""},
    // Deliver
    {"container-aces", "Container: ACES 2065-1", "", "container:aces", O::Studio, E::Always, ""},
    {"container-linear", "Container: linear Rec.2020", "", "container:linear", O::Studio, E::Always, ""},
    {"copy-delivery", "Copy delivery metadata", "", "", O::Studio, E::AnyFrames, ""},
    // Window
    {"rail-left", "Frames rail", "", "railLeft", O::Studio, E::Always, ""},
    {"rail-right", "Reconstruction rail", "", "railRight", O::Studio, E::Always, ""},
    {"scopes", "Scopes", "", "scopesOpen", O::Studio, E::Always, ""},
    {"zoom-fit", "Fit to window", "", "zoom:fit", O::Studio, E::Always, ""},
    {"zoom-actual", "Actual pixels", "", "zoom:actual", O::Studio, E::Always, ""},
    // Help
    {"shortcuts", "Keyboard shortcuts", "?", "", O::Studio, E::Always, ""},
    {"about", "About RUDRA Studio", "", "", O::Studio, E::Always, ""},
    // Bound by the page's bind() rather than listed in ACTIONS: W.
    {"wipe", "Wipe", "W", "wipe", O::Studio, E::Always, ""},

    // Native only. The page opens a package with the server and keeps its
    // view modes and guides on the viewer toolbar; the app also puts them in
    // a menu, on keys the page leaves free. The view peak is the session's
    // peak slider, as on the page.
    {"open-package", "Open model package…", "", "", O::Native, E::Always, "Ctrl+Shift+O"},
    {"open-folder", "Open folder of frames…", "", "", O::Native, E::Always, "Ctrl+Alt+O"},
    {"quit", "Quit", "", "", O::Native, E::Always, "Ctrl+Q"},
    {"view-image", "Image", "", "view:0", O::Native, E::Always, ""},
    {"view-false-colour", "False colour", "", "view:1", O::Native, E::Always, ""},
    {"view-difference", "Difference", "", "view:2", O::Native, E::Always, ""},
    {"guides-action", "Action safe (90 %)", "", "guides:action", O::Native, E::Always, "G"},
    {"guides-title", "Title safe (80 %)", "", "guides:title", O::Native, E::Always, "Shift+G"},
    {"guides-centre", "Centre cross", "", "guides:centre", O::Native, E::Always, ""},
    {"aspect-none", "No aspect mask", "", "aspect:0", O::Native, E::Always, ""},
    {"aspect-2.39", "2.39", "", "aspect:2.39", O::Native, E::Always, ""},
    {"aspect-1.85", "1.85", "", "aspect:1.85", O::Native, E::Always, ""},
    {"aspect-16:9", "16:9", "", "aspect:16:9", O::Native, E::Always, ""},
    {"aspect-4:3", "4:3", "", "aspect:4:3", O::Native, E::Always, ""},
    {"aspect-1:1", "1:1", "", "aspect:1:1", O::Native, E::Always, ""},
};

const std::vector<MenuSpec> kMenus = {
    {"File", {"open", "open-folder", "open-package", "close", "-", "master", "-", "quit"}},
    {"Edit", {"undo", "redo", "-", "reset-recon", "reset-regions"}},
    {"Clip", {"first", "prev", "next", "last", "-", "play"}},
    {"Reconstruct",
     {"mode-all", "mode-highlights", "mode-shadows", "mode-off", "-", "preserve", "-", "strength-down", "strength-up"}},
    {"Measure", {"copy-metrics", "copy-scopes", "-", "remeasure"}},
    {"Deliver", {"master", "-", "container-aces", "container-linear", "-", "copy-delivery"}},
    {"View",
     {"view-image", "view-false-colour", "view-difference", "-", "wipe", "-", ">Guides", "guides-action",
      "guides-title", "guides-centre", "-", "aspect-none", "aspect-2.39", "aspect-1.85", "aspect-16:9", "aspect-4:3",
      "aspect-1:1", "<"}},
    {"Window", {"rail-left", "rail-right", "scopes", "-", "zoom-fit", "zoom-actual"}},
    {"Help", {"shortcuts", "about"}},
};

// The keydown handler: the map, then Space, 1 to 4 and W, which it tests on
// their own.
const std::map<std::string, std::string_view, std::less<>> kKeys = {
    {" ", "play"},        {",", "prev"},        {".", "next"},          {"1", "mode-all"},
    {"2", "mode-highlights"}, {"3", "mode-shadows"}, {"4", "mode-off"},  {"?", "shortcuts"},
    {"End", "last"},      {"Home", "first"},    {"M", "master"},         {"O", "open"},
    {"P", "preserve"},    {"W", "wipe"},        {"Y", "redo"},           {"Z", "undo"},
    {"[", "strength-down"}, {"]", "strength-up"}, {"m", "master"},       {"o", "open"},
    {"p", "preserve"},    {"w", "wipe"},        {"y", "redo"},           {"z", "undo"},
};

const std::vector<ShortcutRow> kSheet = {
    {"B (hold)", "Flip to the inverse-ACES baseline"},
    {"W", "Wipe: baseline left, RUDRA right. Drag the image to move it."},
    {"← →", "Nudge the wipe (Shift for fine)"},
    {"O", "Open frames"},
    {"M", "Master EXR"},
    {", / .", "Previous / next frame"},
    {"Home / End", "First / last frame"},
    {"Space", "Play / pause"},
    {"[ / ]", "Weaker / stronger residual"},
    {"1 2 3 4", "Recovery: all, highlights, shadows, off"},
    {"P", "Preserve outside masks"},
    {"Z / Y", "Undo / redo"},
    {"?", "This list"},
};

}  // namespace

const std::vector<ActionSpec>& action_specs() { return kActions; }

const ActionSpec* find_action(std::string_view id) {
    const auto it = std::find_if(kActions.begin(), kActions.end(), [&](const ActionSpec& a) { return a.id == id; });
    return it == kActions.end() ? nullptr : &*it;
}

const std::vector<MenuSpec>& menu_specs() { return kMenus; }

std::string_view action_for_key(std::string_view key) {
    const auto it = kKeys.find(key);
    return it == kKeys.end() ? std::string_view{} : it->second;
}

std::vector<std::string_view> mapped_keys() {
    std::vector<std::string_view> k;
    for (const auto& [key, act] : kKeys) k.push_back(key);
    return k;
}

double key_nudge_step(bool shift) { return shift ? 0.01 : 0.05; }

const std::vector<ShortcutRow>& shortcut_sheet() { return kSheet; }

}  // namespace rudra
