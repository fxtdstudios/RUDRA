"""Golden viewport geometry from the browser Studio's own layout (Phase 2 step 9).

The native viewer places the picture the way the Studio does: fit, 1:1, zoom
about the cursor, middle-drag pan. In the Studio that is CSS (ui/style.css,
the .viewer / .plate / canvas rules) plus fitScale, applyViewport and
zoomAbout in ui/app.js. This script builds the viewer's markup from
ui/index.html with the real stylesheets in headless Chromium, runs those
three functions unmodified (cut out of app.js by name, as
emit_viewer_golden.py does), and records where the canvas lands
(getBoundingClientRect) after every step of a set of gesture scripts.

    python tools/emit_viewport_golden.py      # writes native/tests/golden/viewport/index.json
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.emit_viewer_golden import cut_function  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "viewport"
UI = REPO / "ui"

# (viewer width, viewer height, frame width, frame height, script)
# Script ops: ["wheel", x, y, dir] (+1 zooms in, x, y in viewer pixels),
# ["pan", dx, dy] (a middle drag), ["actual"] (1:1), ["fit"] (double click).
SCRIPTS = [
    (960, 600, 1920, 1080, [["wheel", 480, 300, 1], ["wheel", 700, 150, 1], ["wheel", 100, 500, -1],
                            ["pan", 37.3, -12.6], ["wheel", 300, 300, 1], ["actual"], ["wheel", 10, 10, 1],
                            ["pan", -200, 90], ["fit"]]),
    (960, 600, 80, 48, [["wheel", 480, 300, 1], ["wheel", 520, 310, 1], ["pan", 5, 5], ["wheel", 0, 0, -1], ["fit"]]),
    (1400, 800, 900, 40, [["wheel", 1000, 400, 1], ["pan", -333.33, 1.05], ["wheel", 50, 790, -1], ["actual"], ["fit"]]),
    (1200, 900, 1080, 1920, [["wheel", 600, 450, -1], ["wheel", 600, 450, -1], ["pan", 0.04, 0.06], ["wheel", 900, 800, 1]]),
    # Clamped at 32x and at 0.05x.
    (800, 600, 400, 300, [["wheel", 400, 300, 1]] * 40 + [["wheel", 400, 300, -1]] * 90),
]

HARNESS = r"""
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var state = {scale: null, panX: 0, panY: 0, zoom: "fit"};
  var frame = null;
  var ctx = {size: function () { return frame; }};
  /*APP*/
  function record() {
    var v = $("viewer").getBoundingClientRect(), c = $("gl").getBoundingClientRect();
    return {scale: state.scale, pan_x: state.panX, pan_y: state.panY, fit: fitScale(),
            label: $("zoomVal").textContent,
            rect: [c.left - v.left, c.top - v.top, c.width, c.height]};
  }
  window.runScript = function (job) {
    var host = $("host");
    host.style.width = job.vw + "px"; host.style.height = job.vh + "px";
    var cv = $("gl"); cv.width = job.fw; cv.height = job.fh;
    frame = {width: job.fw, height: job.fh};
    state.scale = null; state.panX = state.panY = 0;
    applyViewport();
    var v = $("viewer").getBoundingClientRect();
    var out = [record()];
    job.ops.forEach(function (op) {
      if (op[0] === "wheel") {
        zoomAbout(v.left + op[1], v.top + op[2], op[3] > 0 ? 1.12 : 1 / 1.12);
      } else if (op[0] === "pan") {
        // The middle-drag handler: enter scaled mode at the fit scale, then move.
        if (state.scale === null) { state.scale = fitScale(); }
        state.panX = state.panX + op[1]; state.panY = state.panY + op[2];
        applyViewport();
      } else if (op[0] === "actual") {
        state.scale = 1; state.zoom = "actual"; state.panX = state.panY = 0; applyViewport();
      } else if (op[0] === "fit") {
        state.scale = null; applyViewport();
      }
      out.push(record());
    });
    return {viewer: [v.width, v.height, $("viewer").clientWidth, $("viewer").clientHeight], states: out};
  };
}());
"""

PAGE = """<!doctype html><html><head><style>{css}</style>
<style>body{{margin:0}} #host{{display:flex;flex-direction:column}}</style></head><body>
<div id="host"><div class="viewer" id="viewer">
  <div class="plate" id="plate"><canvas id="gl"></canvas></div>
</div></div>
<span id="zoomVal"></span><div id="zoomSeg"></div>
</body></html>"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    app = (UI / "app.js").read_text(encoding="utf-8")
    fns = "\n\n".join(cut_function(app, n) for n in ("fitScale", "applyViewport", "zoomAbout"))
    css = (UI / "theme.css").read_text(encoding="utf-8") + "\n" + (UI / "style.css").read_text(encoding="utf-8")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    cases = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        page.set_content(PAGE.format(css=css))
        page.add_script_tag(content=HARNESS.replace("/*APP*/", fns))
        for vw, vh, fw, fh, ops in SCRIPTS:
            res = page.evaluate("job => window.runScript(job)", {"vw": vw, "vh": vh, "fw": fw, "fh": fh, "ops": ops})
            cases.append({"viewer": [vw, vh], "frame": [fw, fh], "ops": ops, "client": res["viewer"][2:],
                          "states": res["states"]})
            first = res["states"][0]
            print(f"viewer {vw}x{vh} frame {fw}x{fh}: fit {first['fit']:.4f} rect {first['rect']} "
                  f"-> {len(ops)} ops, last {res['states'][-1]['label']}")
        browser.close()
    (OUT / "index.json").write_text(json.dumps({"oracle": "ui/style.css + ui/app.js fitScale, applyViewport, zoomAbout",
                                                "cases": cases}, indent=2), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
