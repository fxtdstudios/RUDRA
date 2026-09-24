"""Golden read-outs for the native probe, frame measurements, clip bar and
pipeline bar (Phase 3 steps 7 and 8).

The page writes them from data: showProbe (the rail's probe panel and the
floating box), showMetrics (the Frame panel's two columns, the mask and time
status, the source line), updatePipe (the colour pipeline bar and its
warning) and paintClipBar. The whole page runs headless (as for the session
golden), each writer runs on the inputs given here, and the text, classes and
widths it left in the DOM are recorded. core/readouts.cpp must write the same.

    python tools/emit_readouts_golden.py      # writes native/tests/golden/readouts/readouts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.emit_session_golden import serve  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "readouts"

PROBES = [
    None,
    {"x": 812, "y": 407, "model": {"nits": 1234.4}, "baseline": {"nits": 203.0},
     "sdr": [255, 250, 10], "hiMask": 0.7349, "shMask": 0.0},
    {"x": 3, "y": 0, "model": {"nits": 0.5}, "baseline": {"nits": 0.52}, "sdr": None, "hiMask": None, "shMask": None},
    {"x": 1919, "y": 1079, "model": {"nits": 12345.6}, "baseline": {"nits": 4000.49},
     "sdr": [254, 12, 0], "hiMask": 1.0, "shMask": 0.125},
    {"x": 100, "y": 200, "model": {"nits": 150.0}, "baseline": {"nits": 180.0},
     "sdr": [180, 181, 179], "hiMask": 0.004, "shMask": 0.995},
    {"x": 0, "y": 5, "model": {"nits": 203.0}, "baseline": {"nits": 203.0}, "sdr": [253, 253, 253],
     "hiMask": 0.0, "shMask": 0.0},
    {"x": 7, "y": 7, "model": {"nits": 0.0001}, "baseline": {"nits": 0.0}, "sdr": [0, 0, 0],
     "hiMask": 0.0, "shMask": 0.5},
]

NAN = "NaN"
METRICS = [
    {"m": {"maxcll": 1235, "maxfall": 87, "peak_nits": 1234.49, "p99_nits": 812.345, "median_nits": 42.125,
           "above_diffuse_white_pct": 12.3456, "above_1000_nits_pct": 0.0125, "headroom_highlight_stops": 2.604,
           "headroom_shadow_stops": -0.1249, "departure_rms_stops": 0.41234, "highlight_mask_pct": 3.456,
           "shadow_mask_pct": 10.0, "compose_ms": 17.25},
     "header": {"source_resolution": "3840x2160", "elapsed_s": 1.234, "resolution": "1920x1080", "tiled": False},
     "clipped": 4.567},
    {"m": {"maxcll": 10000, "maxfall": 2000, "peak_nits": 10000.0, "p99_nits": 4000.0, "median_nits": 0.05,
           "above_diffuse_white_pct": 100.0, "above_1000_nits_pct": 99.9995, "headroom_highlight_stops": NAN,
           "headroom_shadow_stops": NAN, "departure_rms_stops": 0.0, "highlight_mask_pct": 0.0,
           "shadow_mask_pct": 0.0, "compose_ms": 1234.5},
     "header": {"source_resolution": "1920x1080", "elapsed_s": 0.07, "resolution": "1920x1080", "tiled": True},
     "clipped": 0.0},
    {"m": {"maxcll": 3, "maxfall": 1, "peak_nits": 2.25, "p99_nits": 1.005, "median_nits": 0.125,
           "above_diffuse_white_pct": 0.0, "above_1000_nits_pct": 0.0, "headroom_highlight_stops": 0.0,
           "headroom_shadow_stops": 0.0, "departure_rms_stops": 1.0005, "highlight_mask_pct": 99.995,
           "shadow_mask_pct": 0.005, "compose_ms": 0.4},
     "header": None, "clipped": 100.0},
]

PIPES = [
    {"container": "aces", "peakEv": 0, "m": None, "header": None},
    {"container": "linear", "peakEv": 2.5, "m": {"maxcll": 1235}, "header": {"source_bits": 10}},
    {"container": "aces", "peakEv": 0, "m": {"maxcll": 203.2}, "header": {"source_bits": 8}},
    {"container": "aces", "peakEv": 0, "m": {"maxcll": 203.1}, "header": {}},
    {"container": "aces", "peakEv": -1, "m": {"maxcll": 4000}, "header": None},
    {"container": "linear", "peakEv": 5, "m": {"maxcll": NAN}, "header": None},
]

CLIPS = [[0, 0], [4.567, 10.0], [96.5, 10.0], [120, 5], [30, 80]]

RUN = r"""
(json) => {
  const c = JSON.parse(json);
  const S = window.__studio, $ = (id) => document.getElementById(id);
  const fix = (o) => { if (o === null || typeof o !== "object") return o;
    const r = Array.isArray(o) ? [] : {};
    for (const k of Object.keys(o)) r[k] = o[k] === "NaN" ? NaN : fix(o[k]);
    return r; };
  const txt = (id) => $(id).textContent;
  const out = {probes: [], metrics: [], pipes: [], clips: []};
  for (const p of c.probes) {
    S.showProbe(fix(p), 400, 300);
    const rows = Array.from($("probeBox").querySelectorAll(".pr")).map((r) => ({
      k: r.querySelector(".k").textContent, v: r.querySelector(".v").textContent,
      cls: r.querySelector(".v").className.replace(/^v ?/, "")}));
    out.probes.push({xy: txt("probeXY"), nits: txt("probeNits"),
      idle: $("probeNits").parentNode.classList.contains("idle"),
      delta: txt("probeDelta"), delta_class: $("probeDelta").className,
      src: txt("probeSrc"), src_class: $("probeSrc").className, base: txt("probeBase"),
      model: txt("probeModel"), model_class: $("probeModel").className, mask: txt("probeMask"),
      box_hidden: $("probeBox").hidden, box: rows});
  }
  for (const c2 of c.metrics) {
    S.state.header = c2.header; S.state.maskPct = {highlight: 0, shadow: 0, clipped: c2.clipped};
    S.showMetrics(fix(c2.m));
    const rows = (id) => Array.from($(id).querySelectorAll(".ro")).map((r) => ({
      k: r.querySelector(".k").textContent, v: r.querySelector(".v").textContent,
      warn: r.querySelector(".v").classList.contains("warn"), u: r.querySelector(".u").textContent}));
    out.metrics.push({measA: rows("measA"), measB: rows("measB"), statusMask: txt("statusMask"),
                      statusTime: txt("statusTime"), srcInfo: txt("srcInfo")});
  }
  for (const p of c.pipes) {
    S.state.container = p.container; S.state.peakEv = p.peakEv; S.state.header = p.header;
    S.updatePipe(fix(p.m));
    out.pipes.push({pipeIn: txt("pipeIn"), pipeWorking: txt("pipeWorking"), viewTransform: txt("viewTransform"),
                    pipeMaster: txt("pipeMaster"), warn: txt("pipeWarn"), warn_hidden: $("pipeWarn").hidden});
  }
  for (const [a, b] of c.clips) {
    S.paintClipBar(a, b);
    const bar = $("clipBar");
    out.clips.push({hidden: bar.hidden, i_width: bar.querySelector("i").style.width,
                    u_left: bar.querySelector("u").style.left, u_width: bar.querySelector("u").style.width});
  }
  return out;
}
"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.route("https://fonts.googleapis.com/**", lambda r: r.abort())
        page.route("https://fonts.gstatic.com/**", lambda r: r.abort())
        page.route("http://studio.test/**", serve)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://studio.test/index.html")
        page.wait_for_function("window.__studio !== undefined")
        # As JSON text the page parses: Playwright's own argument serialiser
        # hands a Python 0.0 over as -0, which toLocaleString prints "-0.00".
        got = page.evaluate(RUN, json.dumps({"probes": PROBES, "metrics": METRICS, "pipes": PIPES, "clips": CLIPS}))
        browser.close()
        if errors:
            print(f"page errors: {errors}")
            return 1
    golden = {"oracle": "ui/app.js showProbe, showProbePanel, showMetrics, updatePipe, paintClipBar",
              "inputs": {"probes": PROBES, "metrics": METRICS, "pipes": PIPES, "clips": CLIPS}, "page": got}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "readouts.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(golden, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"readouts: {len(PROBES)} probes, {len(METRICS)} frames, {len(PIPES)} pipes, {len(CLIPS)} clip bars"
          f" -> {OUT / 'readouts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
