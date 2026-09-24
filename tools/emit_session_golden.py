"""Golden session scripts for the native app (Phase 3 step 3): the Studio
page's state, params() and undo, driven by its own handlers.

The whole page runs in headless Chromium: ui/index.html with its stylesheets,
compositor.js, app.js and shell.js, served from memory, with the server's
/api/model answering "no model" so nothing reaches the network. app.js is
unmodified except for one line before its closing "}());" that hands the
harness a view of its closure (state, params, snapshot, ACTIONS). Each script
is a list of gestures made with real DOM events -- clicks on the mode buttons
and the Preserve check, the strength and peak sliders (pointerdown, then an
input event, as a drag does), pointer drags and double clicks on the Region
EV values, menu items, and keys on the window -- and after every gesture the
golden records JSON.stringify(params()), the undo and redo depths, and the
wipe. engine/session.cpp must give the same bytes after the same gestures.

    python tools/emit_session_golden.py       # writes native/tests/golden/session/scripts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UI = REPO / "ui"
OUT = REPO / "native" / "tests" / "golden" / "session"

HOOK = ("window.__studio = {state: state, params: params, snapshot: snapshot, ACTIONS: ACTIONS, "
        "displayNits: displayNits, drawScopes: drawScopes, drawVector: drawVector, timecode: timecode};\n")

# Gestures: ["mode", m] clicks the mode button; ["preserve"] clicks the check;
# ["strength", v] and ["peak", v] drag a slider to v; ["region", i, [dx...], shift]
# drags a Region EV value by dx pixels per move; ["region0", i] double-clicks it;
# ["act", id] clicks the menu item; ["key", k, shift] presses a key on the window.
SCRIPTS = {
    "modes_and_strength": [
        ["mode", "highlights"], ["mode", "highlights"], ["key", "3", False], ["key", "1", False],
        ["act", "strength-up"], ["act", "strength-up"], ["key", "]", False], ["strength", "1.35"],
        ["strength", "0.4"], ["preserve"], ["act", "undo"], ["act", "undo"], ["act", "undo"],
        ["act", "redo"], ["mode", "off"], ["act", "redo"], ["act", "reset-recon"], ["key", "z", False],
        ["key", "Y", False], ["key", "4", False], ["key", "p", False], ["key", "P", False],
    ],
    "strength_clamps": [["act", "strength-up"]] * 12 + [["key", "[", False]] * 25 + [["strength", "2"]],
    "peak_is_not_undone": [
        ["peak", "2.5"], ["peak", "-1"], ["peak", "5"], ["peak", "0.5"], ["act", "undo"], ["mode", "shadows"],
        ["peak", "3"], ["act", "undo"],
    ],
    "regions": [
        ["region", 0, [37], False], ["region", 1, [-500], True], ["region", 1, [120, 90, -30], True],
        ["region", 2, [0], False], ["region", 2, [0.3], False], ["region", 0, [1000], False],
        ["region", 0, [-5000], False], ["region0", 0], ["region0", 2], ["act", "reset-regions"],
        ["act", "undo"], ["act", "undo"], ["act", "undo"], ["act", "redo"], ["region", 2, [-3.3], False],
        ["key", "z", False], ["key", "z", False],
    ],
    "undo_holds_sixty": [["act", "strength-down"], ["act", "strength-up"]] * 36 + [["key", "z", False]] * 64 +
                        [["act", "redo"]] * 3,
    "wipe_keys": [
        ["key", "w", False], ["key", "ArrowRight", False], ["key", "ArrowRight", True], ["key", "ArrowLeft", False],
        ["key", "ArrowRight", False]] + [["key", "ArrowRight", False]] * 12 + [
        ["key", "Escape", False], ["key", "W", False], ["key", "ArrowLeft", True], ["key", "w", False],
    ],
}

RUN = r"""
async (ops) => {
  const S = window.__studio;
  const $ = (id) => document.getElementById(id);
  const out = [];
  const record = (op) => out.push({op: op, params: JSON.stringify(S.params()),
                                   undo: S.state.undo.length, redo: S.state.redo.length,
                                   wipe: S.state.wipe});
  const slider = (id, v) => {
    const el = $(id);
    el.dispatchEvent(new PointerEvent("pointerdown", {bubbles: true}));
    el.value = v;
    el.dispatchEvent(new Event("input", {bubbles: true}));
  };
  record(["start"]);
  for (const op of ops) {
    const k = op[0];
    if (k === "mode") {
      document.querySelector('#mode button[data-mode="' + op[1] + '"]').click();
    } else if (k === "preserve") {
      $("preserve").click();
    } else if (k === "strength") {
      slider("strength", op[1]);
    } else if (k === "peak") {
      slider("peak", op[1]);
    } else if (k === "region") {
      const ev = $("regions").querySelectorAll(".ev")[op[1]];
      let x = 100;
      ev.dispatchEvent(new PointerEvent("pointerdown", {bubbles: true, clientX: x, pointerId: 1}));
      for (const dx of op[2]) {
        x += dx;
        $("regions").dispatchEvent(new PointerEvent("pointermove", {bubbles: true, clientX: x, shiftKey: op[3],
                                                                   pointerId: 1}));
      }
      window.dispatchEvent(new PointerEvent("pointerup", {bubbles: true, pointerId: 1}));
    } else if (k === "region0") {
      const ev = $("regions").querySelectorAll(".ev")[op[1]];
      ev.dispatchEvent(new MouseEvent("dblclick", {bubbles: true}));
    } else if (k === "act") {
      document.querySelector('#menubar button[data-act="' + op[1] + '"]').click();
    } else if (k === "key") {
      window.dispatchEvent(new KeyboardEvent("keydown", {key: op[1], shiftKey: op[2], bubbles: true}));
      window.dispatchEvent(new KeyboardEvent("keyup", {key: op[1], shiftKey: op[2], bubbles: true}));
    }
    record(op);
  }
  return out;
}
"""


def serve(route) -> None:
    url = route.request.url
    path = url.split("://", 1)[1].split("/", 1)[1].split("?", 1)[0]
    if path.startswith("api/model"):
        route.fulfill(status=200, content_type="application/json",
                      body=json.dumps({"loaded": False, "reason": "golden harness: no model"}))
        return
    if path.startswith("api/"):
        route.fulfill(status=404, body="")
        return
    f = UI / (path or "index.html")
    if not f.is_file():
        route.fulfill(status=404, body="")
        return
    body = f.read_text(encoding="utf-8") if f.suffix in (".html", ".js", ".css") else f.read_bytes()
    if f.name == "app.js":
        end = body.rindex("}());")
        body = body[:end] + HOOK + body[end:]
    types = {".html": "text/html", ".js": "application/javascript", ".css": "text/css", ".png": "image/png"}
    route.fulfill(status=200, content_type=types.get(f.suffix, "application/octet-stream"),
                  body=body if isinstance(body, bytes) else body.encode("utf-8"))


def main() -> int:
    from playwright.sync_api import sync_playwright

    results = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for name, ops in SCRIPTS.items():
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.route("https://fonts.googleapis.com/**", lambda r: r.abort())
            page.route("https://fonts.gstatic.com/**", lambda r: r.abort())
            page.route("http://studio.test/**", serve)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("http://studio.test/index.html")
            page.wait_for_function("window.__studio !== undefined")
            page.wait_for_function("document.getElementById('regions').children.length === 3")
            results[name] = page.evaluate(RUN, ops)
            if errors:
                print(f"{name}: page errors: {errors}")
                return 1
            page.close()
        browser.close()

    OUT.mkdir(parents=True, exist_ok=True)
    golden = {"oracle": "ui/app.js (state, params, pushUndo, undo, redo, the handlers) in the whole page",
              "scripts": results}
    with open(OUT / "scripts.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(golden, f, indent=1)
        f.write("\n")
    steps = sum(len(v) for v in results.values())
    print(f"session: {len(results)} scripts, {steps} states -> {OUT / 'scripts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
