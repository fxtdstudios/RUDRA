"""Golden clipboard texts for the native Measure and Deliver copies (Phase 3 step 11).

The page's copy-metrics, copy-scopes and copy-delivery write
JSON.stringify(value, null, 2) of state.metrics, state.scopeData and
deliveryRecord() to the clipboard. The whole page runs headless (as for the
session golden), the state each case gives is set on it, the clipboard is
replaced by a recorder, and the page's own action runs; the text it wrote is
recorded, with the log line. core/js_json and engine/copy_texts must write
the same bytes from the same state.

    python tools/emit_copy_golden.py   # writes native/tests/golden/copy/texts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.emit_session_golden import serve  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "copy"
NAN = "NaN"

METRICS = {"maxcll": 1235, "maxfall": 87, "peak_nits": 1234.4912109375, "baseline_peak_nits": 402.25,
           "headroom_stops": 1.617, "headroom_highlight_stops": 2.6040000000000001,
           "headroom_shadow_stops": NAN, "departure_rms_stops": 0.41234,
           "p99_nits": 812.3450000000001, "median_nits": 42.125, "above_diffuse_white_pct": 12.3456,
           "above_1000_nits_pct": 0.0125, "highlight_mask_pct": 3.456, "shadow_mask_pct": 0,
           "compose_ms": 17.25}

SCOPES = {"lo": [0.0009765625, 0.5], "q1": [0.1, 1], "mid": [0.30000000000000004, 1e-7], "q3": [0.75, 123456.5],
          "hi": [1, 1e21], "histogram": [0, 0.25, 1]}

REGIONS_DEFAULT = [{"label": "highlights", "low_nits": 400, "high_nits": 2000, "ev": 0},
                   {"label": "speculars", "low_nits": 2000, "high_nits": 8000, "ev": 0},
                   {"label": "shadows", "low_nits": 0.05, "high_nits": 12, "ev": 0}]
REGIONS_GRADED = [dict(REGIONS_DEFAULT[0], ev=0.5), dict(REGIONS_DEFAULT[1], ev=-1.25),
                  dict(REGIONS_DEFAULT[2], ev=0)]

# (name, action, state to set)
CASES = [
    ("metrics", "copy-metrics", {"metrics": METRICS}),
    ("scopes", "copy-scopes", {"scopeData": SCOPES}),
    ("delivery_empty", "copy-delivery", {}),
    ("delivery_full", "copy-delivery",
     {"header": {"checkpoint": "sdr2hdr_shadow_v1", "step": 90000, "resolution": "1920x1080"},
      "frames": [{"name": "shot_010.000001.png"}], "index": 0, "container": "linear", "mode": "shadows",
      "strength": 1.1, "preserve": False, "regions": REGIONS_GRADED, "metrics": METRICS}),
    ("delivery_ungraded", "copy-delivery",
     {"header": {"checkpoint": "a \"quoted\" name\\", "resolution": "640x480"},
      "frames": [{"name": "café\tframe.png"}], "index": 0, "container": "aces", "mode": "all",
      "strength": 0.30000000000000004, "preserve": True, "regions": REGIONS_DEFAULT}),
]

RUN = r"""
(json) => {
  const c = JSON.parse(json);
  const fix = (o) => { if (o === null || typeof o !== "object") return o;
    const r = Array.isArray(o) ? [] : {};
    for (const k of Object.keys(o)) r[k] = o[k] === "NaN" ? NaN : fix(o[k]);
    return r; };
  const S = window.__studio, st = S.state;
  const set = fix(c.set);
  for (const k of Object.keys(set)) st[k] = set[k];
  let text = null;
  Object.defineProperty(navigator, "clipboard", {configurable: true,
    value: {writeText: (t) => { text = t; return Promise.resolve(); }}});
  S.ACTIONS[c.act]();
  return new Promise((done) => setTimeout(() => {
    const lines = document.getElementById("log").innerText.trim().split("\n");
    done({text: text, log: lines[lines.length - 1]});
  }, 20));
}
"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    out = {"oracle": "ui/app.js copy(): JSON.stringify(value, null, 2) of state.metrics, state.scopeData, "
                     "deliveryRecord()", "cases": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        errors: list[str] = []
        for name, act, state in CASES:
            # A fresh page per case: the state a case sets is all it has.
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.route("http://studio.test/**", serve)
            page.route("https://**", lambda r: r.abort())
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("http://studio.test/index.html")
            page.wait_for_function("window.__studio !== undefined")
            got = page.evaluate(RUN, json.dumps({"act": act, "set": state}))
            out["cases"].append({"name": name, "act": act, "state": state, "text": got["text"], "log": got["log"]})
            page.close()
        browser.close()
        if errors:
            print(f"page errors: {errors}")
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "texts.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"copy: {len(CASES)} cases -> {OUT / 'texts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
