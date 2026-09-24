"""Golden actions for the native app (Phase 3 step 2): the Studio page's menus,
actions, keys and shortcut sheet, read from ui/index.html and ui/app.js.

The page is the oracle, read as it is: the menubar from the markup (every
menu, item, separator, key hint and check state, in order), and from app.js
the ACTIONS table (with ACTIONS.wipe, which bind() adds), the keydown map and
the keys the handler takes before it, the SHORTCUTS sheet, and the
refreshMenu() enable rules. The native action table (engine/actions.cpp)
must equal it; the app's menubar is then checked against the same file.

    python tools/emit_actions_golden.py        # writes native/tests/golden/actions/
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "native" / "tests" / "golden" / "actions"


class MenuParser(HTMLParser):
    """The .menubar: <div class="menu"> with a .mtitle button and a .drop of
    <button data-act> items, <hr> separators and <i> key hints."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.menus: list[dict] = []
        self.depth = 0
        self.menu_depth = None
        self.in_title = False
        self.item: dict | None = None
        self.in_hint = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "div":
            self.depth += 1
            if "menu" in cls and self.menu_depth is None:
                self.menu_depth = self.depth
                self.menus.append({"title": "", "items": []})
        if self.menu_depth is None:
            return
        if tag == "button" and "mtitle" in cls:
            self.in_title = True
        elif tag == "button" and "data-act" in a:
            self.item = {"act": a["data-act"], "label": "", "hint": None, "check": a.get("data-check")}
        elif tag == "hr":
            self.menus[-1]["items"].append({"separator": True})
        elif tag == "i" and self.item is not None:
            self.in_hint = True
            self.item["hint"] = ""

    def handle_endtag(self, tag):
        if tag == "div":
            if self.menu_depth == self.depth:
                self.menu_depth = None
            self.depth -= 1
        elif tag == "button":
            if self.in_title:
                self.in_title = False
            elif self.item is not None:
                self.item["label"] = self.item["label"].strip()
                self.menus[-1]["items"].append(self.item)
                self.item = None
        elif tag == "i":
            self.in_hint = False

    def handle_data(self, data):
        if self.menu_depth is None:
            return
        if self.in_title:
            self.menus[-1]["title"] += data.strip()
        elif self.item is not None:
            if self.in_hint:
                self.item["hint"] += data.strip()
            else:
                self.item["label"] += data


def block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i + len(start))
    return src[i + len(start):j]


def main() -> int:
    html = (REPO / "ui" / "index.html").read_text(encoding="utf-8")
    js = (REPO / "ui" / "app.js").read_text(encoding="utf-8")

    p = MenuParser()
    p.feed(html)
    menus = p.menus

    actions = re.findall(r'^    "([a-z-]+)": ', block(js, "var ACTIONS = {", "\n  };"), re.M)
    actions += re.findall(r"ACTIONS\.([a-z]+) = function", js)

    # The keydown handler: modifiers pass through, B is held, W wipes, the
    # arrows nudge a wipe, Escape leaves it, then the map, Space and 1 to 4.
    handler = block(js, 'window.addEventListener("keydown", function (e) {', "\n    });")
    keymap = json.loads("{" + block(handler, "var map = {", "};").strip().rstrip(",") + "}")
    modes = json.loads(re.search(r'setMode\((\[[^\]]+\])\[Number\(k\) - 1\]\)', handler).group(1))
    for n, mode in enumerate(modes, 1):
        keymap[str(n)] = "mode-" + mode
    if 'if (k === " ") { e.preventDefault(); togglePlay();' in handler:
        keymap[" "] = "play"
    if 'if (k === "w" || k === "W") { ACTIONS.wipe(); return; }' in handler:
        keymap["w"] = keymap["W"] = "wipe"
    nudge = re.search(r"var step = e\.shiftKey \? ([0-9.]+) : ([0-9.]+);", handler)
    special = {
        "modifiers_pass": "if (e.metaKey || e.ctrlKey || e.altKey) { return; }" in handler,
        "hold_b_flips": 'if (k === "b" || k === "B")' in handler and "state.flipHeld = true" in handler,
        "wipe_nudge": {"step": float(nudge.group(2)), "shift_step": float(nudge.group(1))},
        "escape_leaves_wipe": 'if (k === "Escape")' in handler and "state.wipe = null" in handler,
    }

    shortcuts = json.loads(block(js, "var SHORTCUTS = ", ";\n"))

    flags_src = block(js, "var flags = {", "};")
    enable = dict(re.findall(r'"([a-z-]+)":\s*([^,\n]+?)(?:,|\s*$)', flags_src, re.M))

    golden = {
        "source": "ui/index.html (menubar) and ui/app.js (ACTIONS, keydown, SHORTCUTS, refreshMenu)",
        "menus": menus,
        "actions": sorted(set(actions)),
        "keys": dict(sorted(keymap.items())),
        "special_keys": special,
        "shortcuts": shortcuts,
        "enable_when": dict(sorted((k, v.strip()) for k, v in enable.items())),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "actions.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(golden, f, indent=1, ensure_ascii=False)
        f.write("\n")
    n_items = sum(1 for m in menus for i in m["items"] if "act" in i)
    print(f"actions: {len(menus)} menus, {n_items} items, {len(golden['actions'])} actions, "
          f"{len(keymap)} keys, {len(shortcuts)} shortcut rows -> {OUT / 'actions.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
