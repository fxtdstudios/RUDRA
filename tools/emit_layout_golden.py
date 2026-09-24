"""Golden layout of the Studio page for the native main window (Phase 3 step 4).

The whole page runs in headless Chromium at 1600 x 1000, served from memory
as emit_session_golden.py serves it (no model, nothing on the network), and
this records, in five states -- the full workspace, the simple one, both
rails and the scopes hidden from the Window menu, and the Grade and Deliver
tabs -- every element with an id: whether it is shown and its text (boxes only
for the frame, which the page sizes itself: the rest depend on the fonts),
plus the panel labels, toolbar labels, segment buttons, check boxes, fields
and notes in reading order. The native window builds the same named widgets
with the same words, shows and hides them in the same states, and puts the
frame (menubar, icon rail, rails, toolbar, transport, pipeline bar) where the
page puts it.

    python tools/emit_layout_golden.py        # writes native/tests/golden/layout/layout.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.emit_session_golden import serve  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "layout"
W, H = 1600, 1000

# state name -> clicks to reach it from a fresh page (menu items by data-act, or ids).
STATES = {
    "full": [],
    "simple": ["#wsSimple"],
    "rails_hidden": ["act:rail-left", "act:rail-right", "act:scopes"],
    "tab_grade": ["#tabGrade"],
    "tab_deliver": ["#tabDeliver"],
}

SNAP = r"""
() => {
  const txt = (el) => (el.textContent || "").replace(/\s+/g, " ").trim();
  const shown = (el) => {
    if (el.hidden) return false;
    for (let e = el; e && e !== document.body; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === "none" || cs.visibility === "hidden" || e.hidden) return false;
    }
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  };
  const box = (el) => { const r = el.getBoundingClientRect();
    return [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; };
  const ids = {};
  document.querySelectorAll(".app [id]").forEach((el) => {
    if (el.closest("svg") && el.tagName.toLowerCase() !== "svg") return;
    const kids = el.querySelectorAll(":scope > button");
    ids[el.id] = {tag: el.tagName.toLowerCase(), shown: shown(el),
                  text: ["input", "select", "canvas", "svg"].includes(el.tagName.toLowerCase()) ? null : txt(el),
                  buttons: kids.length ? Array.from(kids).map(txt) : null,
                  on: el.classList.contains("on"),
                  disabled: el.disabled === true};
  });
  const list = (sel) => Array.from(document.querySelectorAll(sel)).map((el) => ({text: txt(el), shown: shown(el)}));
  const frame = {};
  for (const [k, sel] of Object.entries({menubar: ".menubar", iconrail: ".iconrail", rail_left: "#railLeft",
                                         centre: ".centre", vtools: ".vtools", viewer: "#viewer",
                                         transport: ".transport", rail_right: "#railRight", pipe: "#pipe"})) {
    const el = document.querySelector(sel);
    frame[k] = {shown: shown(el), box: box(el)};
  }
  return {
    ids: ids, frame: frame,
    panel_labels: list(".plabel > span:first-child, .plabel:not(:has(span))"),
    toolbar_labels: list(".vtools .vlabel"),
    checks: Array.from(document.querySelectorAll(".check")).map((el) => ({id: el.id,
      label: txt(el.querySelector(".lab")), hint: el.querySelector(".hint") ? txt(el.querySelector(".hint")) : null,
      on: el.classList.contains("on")})),
    fields: Array.from(document.querySelectorAll(".field")).map((el) => [txt(el.querySelector(".k")), txt(el.querySelector(".v"))]),
    notes: list(".note-p"),
    sliders: Array.from(document.querySelectorAll(".slider")).map((el) => ({
      label: txt(el.querySelector(".lab")), value: txt(el.querySelector(".val")), unit: txt(el.querySelector(".unit")),
      min: el.querySelector("input").min, max: el.querySelector("input").max, step: el.querySelector("input").step})),
    icon_titles: Array.from(document.querySelectorAll(".iconrail .ibtn")).map((el) => ({id: el.id, title: el.title,
      on: el.classList.contains("on")})),
    titles: Object.fromEntries(Array.from(document.querySelectorAll(".app [id][title]")).map((el) => [el.id, el.title])),
    placeholders: Object.fromEntries(Array.from(document.querySelectorAll(".app input[placeholder]")).map((el) => [el.id, el.placeholder])),
    body_class: document.body.className,
  };
}
"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    states = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for name, clicks in STATES.items():
            page = browser.new_page(viewport={"width": W, "height": H})
            page.route("https://fonts.googleapis.com/**", lambda r: r.abort())
            page.route("https://fonts.gstatic.com/**", lambda r: r.abort())
            page.route("http://studio.test/**", serve)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("http://studio.test/index.html")
            page.wait_for_function("window.__studio !== undefined")
            page.wait_for_function("document.getElementById('regions').children.length === 3")
            for c in clicks:
                if c.startswith("act:"):
                    page.evaluate("(a) => document.querySelector('#menubar button[data-act=\"' + a + '\"]').click()",
                                  c[4:])
                else:
                    page.evaluate("(s) => document.querySelector(s).click()", c)
            page.wait_for_timeout(50)
            states[name] = page.evaluate(SNAP)
            if errors:
                print(f"{name}: page errors: {errors}")
                return 1
            page.close()
        browser.close()
    OUT.mkdir(parents=True, exist_ok=True)
    golden = {"oracle": "ui/index.html, style.css, theme.css, app.js and shell.js, rendered at 1600 x 1000",
              "viewport": [W, H], "states": states}
    with open(OUT / "layout.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(golden, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"layout: {len(states)} states, {len(states['full']['ids'])} ids -> {OUT / 'layout.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
