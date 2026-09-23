"""Golden viewer output from the browser Studio itself (Phase 2 step 1).

The native viewer ports ui/compositor.js and the measurement and scope code of
ui/app.js, so those files are the oracle. This script runs them UNMODIFIED in
headless Chromium (Playwright; WebGL2 with float render targets on SwiftShader):

    compositor.js   loaded whole, driven through window.RudraGL
    app.js          the functions the viewer ports -- HALF, adopt, computeStats,
                    buildScopes, drawVector -- cut out of the file by name and
                    evaluated against stubs for the DOM and the page state, so
                    what runs is the Studio's own code, not a transcription

Each frame goes in exactly as POST /api/frame sends it (ui/server.py): fields
as float16 RGBA (log residual, highlight mask), the shadow mask as float16,
the SDR as the 8-bit pixels the page composites from, and the header scalars.
The fields come from the shipped checkpoint on CPU, like the other goldens.

Recorded per frame, under native/tests/golden/viewer/:

    <frame>_body.bin              the /api/frame body, byte for byte
    <frame>_<case>_model.npy      readComposite("model"), RGB float32, image row order
                                  (every case on main, the first on wide)
    <frame>_base.npy              readComposite("baseline")
    <frame>_<view>.npy            the canvas after present(), RGB8, image row order
    <frame>_sample_index.npy      sample()'s source pixel per sample; its values are
                                  checked here to be the composites at that index
    <frame>_vector.npy            drawVector's 256x256 RGBA8 image
    index.json                    headers, cases, peak/mean reductions, probes,
                                  computeStats metrics, buildScopes output

    python tools/emit_viewer_golden.py        # needs: pip install playwright

SwiftShader is deterministic on one machine; across CPUs the float results may
move in the last bit, so the native tests compare with tolerances and CI
re-emits rather than diffs.
"""
from __future__ import annotations

import base64
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.emit_composite_golden import GRADED_BANDS, synthetic_frame  # noqa: E402
from training.infer_sdr2hdr import load_models, predict_fields  # noqa: E402
from ui.server import MAX_FIELD_MAGNITUDE  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "viewer"
CHECKPOINT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"
APP = REPO / "ui" / "app.js"
COMPOSITOR = REPO / "ui" / "compositor.js"

# (name, height, width): a small frame, and a long thin one whose long side
# passes the 768-pixel sample cap so sample() really downsamples.
FRAMES = [("main", 48, 80), ("wide", 40, 900)]

COMPOSITE_CASES = [
    ("all_s100_p", {"mode": "all", "strength": 1.0, "preserve": True}),
    ("all_s060", {"mode": "all", "strength": 0.6, "preserve": False}),
    ("highlights_s130_p", {"mode": "highlights", "strength": 1.3, "preserve": True}),
    ("shadows", {"mode": "shadows", "strength": 1.0, "preserve": False}),
    ("off", {"mode": "off", "strength": 1.0, "preserve": True}),
    ("graded", {"mode": "all", "strength": 1.0, "preserve": True, "regions": GRADED_BANDS}),
]

# Display cases run on the first composite case. Keys are ctx.setParams keys.
VIEW_CASES = [
    ("view_image_203", {"view": 0, "displayNits": 203.0, "wipe": -1, "show": "model"}),
    ("view_image_1000", {"view": 0, "displayNits": 1000.0, "wipe": -1, "show": "model"}),
    ("view_baseline_203", {"view": 0, "displayNits": 203.0, "wipe": -1, "show": "baseline"}),
    ("view_false_colour", {"view": 1, "displayNits": 203.0, "wipe": -1, "show": "model"}),
    ("view_difference", {"view": 2, "displayNits": 203.0, "wipe": -1, "show": "model", "diffGain": 2000.0}),
    ("view_wipe_037", {"view": 0, "displayNits": 406.0, "wipe": 0.37, "show": "model"}),
    ("view_wipe_false_colour", {"view": 1, "displayNits": 203.0, "wipe": 0.61, "show": "model"}),
]

APP_FUNCTIONS = ["adopt", "computeStats", "buildScopes", "drawVector"]
APP_CONSTANTS = ["var PEAK_NITS", "var SCOPE_LO", "var W = 460", "var COLUMNS", "var VEC_SIZE"]


def cut_function(src: str, name: str) -> str:
    """The text of `function name(...) {...}` from app.js, braces matched with
    strings, comments and regex-free code in mind (app.js has no regex literals
    in these functions)."""
    m = re.search(r"^  function " + re.escape(name) + r"\(", src, re.M)
    if not m:
        raise SystemExit(f"app.js has no function {name}")
    i = src.index("{", m.end())
    depth, j, quote = 0, i, None
    while j < len(src):
        c = src[j]
        if quote:
            if c == "\\":
                j += 1
            elif c == quote:
                quote = None
        elif src.startswith("/*", j):
            j = src.index("*/", j) + 1
        elif src.startswith("//", j):
            j = src.index("\n", j)
        elif c in "'\"":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[m.start():j + 1]
        j += 1
    raise SystemExit(f"unbalanced braces in {name}")


def cut_half(src: str) -> str:
    start = src.index("  var HALF = (function () {")
    end = src.index("}());", start) + len("}());")
    return src[start:end]


def app_scope() -> str:
    src = APP.read_text(encoding="utf-8")
    lines = []
    for prefix in APP_CONSTANTS:
        hit = [ln for ln in src.splitlines() if ln.startswith("  " + prefix)]
        if len(hit) != 1:
            raise SystemExit(f"app.js: expected one line starting {prefix!r}, found {len(hit)}")
        lines.append(hit[0])
    parts = [cut_half(src)] + [cut_function(src, n) for n in APP_FUNCTIONS]
    return "\n".join(lines) + "\n" + "\n\n".join(parts)


HARNESS = r"""
(function () {
  var captured = {};
  function element(id) {
    if (id === "vector") {
      return {getContext: function () {
        return {createImageData: function (w, h) {
                  return {width: w, height: h, data: new Uint8ClampedArray(w * h * 4)}; },
                putImageData: function (img) { captured.vector = img.data; }};
      }};
    }
    return {hidden: false, textContent: "", title: "", innerHTML: "",
            classList: {toggle: function () {}}, children: []};
  }
  var $ = element;
  var state = {frames: [], index: -1, mode: "all", strength: 1, peakEv: 0, preserve: true,
               view: 0, wipe: null, regions: null, header: null, metrics: null,
               scopeData: null, maskPct: {highlight: 0, shadow: 0}};
  var ctx = null;
  var hiMask = null, shMask = null;
  function paintClipBar() {}
  function present() {}
  function showMetrics(m) { captured.metrics = m; }
  function drawScopes() {}
  function current() { return null; }
  function updatePipe() {}
  function drawFrames() {}
  function evictCache() {}

  /*APP*/

  function b64(typed) {
    var bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength), out = "";
    for (var i = 0; i < bytes.length; i += 0x8000) {
      out += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(out);
  }
  function unb64(s) {
    var bin = atob(s), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) { out[i] = bin.charCodeAt(i); }
    return out.buffer;
  }
  function canvasPixels(gl, w, h) {
    var px = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px;
  }

  window.runOracle = function (job) {
    var canvas = document.getElementById("gl");
    ctx = window.RudraGL.create(canvas);
    if (!ctx) { return {error: "no WebGL2 with float targets"}; }
    var gl = ctx.gl;
    var frame = {header: job.header, buf: unb64(job.body)};
    state.regions = job.default_regions;
    adopt(frame);
    var w = job.header.width, h = job.header.height, res = {cases: [], views: [], probes: []};
    res.mask_pct = state.maskPct;
    res.base = b64(ctx.readComposite("baseline"));
    job.cases.forEach(function (c) {
      var p = c.params;
      state.mode = p.mode; state.strength = p.strength; state.preserve = p.preserve;
      ctx.setParams({strength: p.strength, mode: p.mode, preserve: p.preserve,
                     regions: p.regions || job.default_regions});
      captured = {};
      computeStats();
      var m = captured.metrics; delete m.compose_ms;
      var out = {name: c.name, model: b64(ctx.readComposite("model")),
                 peak_nits: ctx.peakNits(), base_peak_nits: ctx.basePeakNits(),
                 mean_nits: ctx.meanNits(), metrics: m, scopes: state.scopeData};
      if (c.name === job.cases[0].name) {
        var s = ctx.sample();
        out.sample = {width: s.width, height: s.height, model: b64(s.model),
                      baseline: b64(s.baseline), index: b64(s.index)};
        out.vector = b64(captured.vector);
        job.probes.forEach(function (pt) { res.probes.push(ctx.probe(pt[0], pt[1])); });
      }
      res.cases.push(out);
    });
    var first = job.cases[0].params;
    ctx.setParams({strength: first.strength, mode: first.mode, preserve: first.preserve,
                   regions: first.regions || job.default_regions});
    job.views.forEach(function (v) {
      ctx.setParams(v.params);
      ctx.present();
      res.views.push({name: v.name, pixels: b64(canvasPixels(gl, w, h))});
    });
    res.canvas = [canvas.width, canvas.height];
    return res;
  };
}());
"""


def pack(sdr: np.ndarray, fields: dict, shadow_weight: float, model) -> tuple[dict, bytes]:
    """ui/server.py's /api/frame body and header for one frame."""
    residual = fields["residual"][0].permute(1, 2, 0).numpy()
    highlight = fields["highlight"][0].permute(1, 2, 0).numpy()
    shadow = fields["shadow"][0, 0].numpy()

    def half(a):
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=MAX_FIELD_MAGNITUDE, neginf=-MAX_FIELD_MAGNITUDE),
                       -MAX_FIELD_MAGNITUDE, MAX_FIELD_MAGNITUDE).astype(np.float16)

    sdr_u8 = (np.clip(sdr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    body = half(np.concatenate([residual, highlight], axis=-1)).tobytes() + half(shadow).tobytes() + sdr_u8.tobytes()
    h, w = sdr_u8.shape[:2]
    header = {"width": w, "height": h, "shadow_weight": shadow_weight,
              "log_scale": float(getattr(model, "log_scale", 16.0)),
              "max_hdr": float(getattr(model, "max_hdr", 4.0)),
              "corpus_ev": float(getattr(model, "corpus_ev", -1.0)),
              "curve": (None if fields.get("curve") is None else [float(v) for v in fields["curve"][0].float()]),
              "offsets": {"fields": 0, "shadow": w * h * 8, "sdr": w * h * 10, "total": len(body)}}
    return header, body


def jsonable(v):
    if isinstance(v, float) and not math.isfinite(v):
        return "nan" if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, dict):
        return {k: jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [jsonable(x) for x in v]
    return v


def f32(b64: str, h: int, w: int, c: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), np.float32).reshape(h, w, c)


def main() -> int:
    from playwright.sync_api import sync_playwright

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    torch.set_num_threads(1)
    model, _ = load_models(str(CHECKPOINT), None, torch.device("cpu"))
    script = HARNESS.replace("/*APP*/", app_scope())
    default_regions = [{"label": "highlights", "low_nits": 400, "high_nits": 2000, "ev": 0},
                       {"label": "speculars", "low_nits": 2000, "high_nits": 8000, "ev": 0},
                       {"label": "shadows", "low_nits": 0.05, "high_nits": 12, "ev": 0}]
    index = {"oracle": "ui/compositor.js, ui/app.js (adopt, computeStats, buildScopes, drawVector)",
             "checkpoint": CHECKPOINT.name, "renderer": None, "default_regions": default_regions,
             "rows": "image order: row 0 is the top of the picture (readPixels rows flipped)",
             "frames": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                          "--enable-unsafe-swiftshader"])
        for name, h, w in FRAMES:
            sdr = synthetic_frame(h, w)
            t = torch.from_numpy(np.ascontiguousarray(np.transpose(sdr, (2, 0, 1))))[None]
            with torch.inference_mode():
                fields = predict_fields(model, t, tile_size=0, overlap=64)
                sw = model.predict_shadow_weight(t) if getattr(model, "shadow_gate", None) is not None else None
            header, body = pack(sdr, fields, 1.0 if sw is None else float(sw.reshape(-1)[0]), model)
            (OUT / f"{name}_body.bin").write_bytes(body)
            probes = [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w // 2, h // 2],
                      [3 * w // 4, h // 2], [w // 5, 5 * h // 6], [w // 3, h // 7]]
            job = {"header": header, "body": base64.b64encode(body).decode(),
                   "default_regions": default_regions, "probes": probes,
                   "cases": [{"name": n, "params": prm} for n, prm in COMPOSITE_CASES],
                   "views": [{"name": n, "params": prm} for n, prm in VIEW_CASES]}
            page = browser.new_page()
            page.set_content(f'<canvas id="gl" width="{w}" height="{h}"></canvas>')
            page.add_script_tag(content=COMPOSITOR.read_text(encoding="utf-8"))
            page.add_script_tag(content=script)
            index["renderer"] = page.evaluate(
                "() => { const g = document.createElement('canvas').getContext('webgl2');"
                " const d = g.getExtension('WEBGL_debug_renderer_info');"
                " return d ? g.getParameter(d.UNMASKED_RENDERER_WEBGL) : g.getParameter(g.RENDERER); }")
            res = page.evaluate("job => window.runOracle(job)", job)
            page.close()
            if "error" in res:
                raise SystemExit(res["error"])
            if res["canvas"] != [w, h]:
                raise SystemExit(f"canvas is {res['canvas']}, frame is {w}x{h}")
            np.save(OUT / f"{name}_base.npy", np.ascontiguousarray(f32(res["base"], h, w, 4)[..., :3]))
            block = {"name": name, "header": header, "body": f"{name}_body.bin", "base": f"{name}_base.npy",
                     "mask_pct": res["mask_pct"], "probes": [], "cases": [], "views": []}
            for k, c in enumerate(res["cases"]):
                # Every composite of the small frame; only the first of the long one (size).
                keep = name == FRAMES[0][0] or k == 0
                if keep:
                    np.save(OUT / f"{name}_{c['name']}_model.npy",
                            np.ascontiguousarray(f32(c["model"], h, w, 4)[..., :3]))
                entry = {"name": c["name"], "params": dict(COMPOSITE_CASES)[c["name"]],
                         "model": f"{name}_{c['name']}_model.npy" if keep else None,
                         "peak_nits": c["peak_nits"], "base_peak_nits": c["base_peak_nits"],
                         "mean_nits": c["mean_nits"], "metrics": c["metrics"], "scopes": c["scopes"]}
                if "sample" in c:
                    s = c["sample"]
                    sh, sww = s["height"], s["width"]
                    idx = np.frombuffer(base64.b64decode(s["index"]), np.int32).reshape(sh, sww)
                    # sample() is a point sample of the two composites at `index`; checked here so
                    # only the index needs storing.
                    full = f32(c["model"], h, w, 4).reshape(-1, 4)
                    base_full = f32(res["base"], h, w, 4).reshape(-1, 4)
                    if not (np.array_equal(f32(s["model"], sh, sww, 4).reshape(-1, 4), full[idx.ravel()])
                            and np.array_equal(f32(s["baseline"], sh, sww, 4).reshape(-1, 4), base_full[idx.ravel()])):
                        raise SystemExit(f"{name}: sample() is not a point sample of the composite at its index")
                    np.save(OUT / f"{name}_sample_index.npy", np.ascontiguousarray(idx.astype(np.float32)))
                    vec = np.frombuffer(base64.b64decode(c["vector"]), np.uint8).reshape(256, 256, 4)
                    np.save(OUT / f"{name}_vector.npy", np.ascontiguousarray(vec))
                    entry["sample"] = {"width": sww, "height": sh, "index": f"{name}_sample_index.npy",
                                       "is": "model and baseline composites at index (row-major source pixel)"}
                    entry["vector"] = f"{name}_vector.npy"
                block["cases"].append(entry)
            for pt, pr in zip(probes, res["probes"]):
                block["probes"].append({"at": pt, **pr})
            for v in res["views"]:
                px = np.frombuffer(base64.b64decode(v["pixels"]), np.uint8).reshape(h, w, 4)[::-1, :, :3]
                np.save(OUT / f"{name}_{v['name']}.npy", np.ascontiguousarray(px))
                block["views"].append({"name": v["name"], "params": dict(VIEW_CASES)[v["name"]],
                                       "pixels": f"{name}_{v['name']}.npy"})
            index["frames"].append(block)
            print(f"{name} {w}x{h}: {len(block['cases'])} composites, {len(block['views'])} views, "
                  f"peak {block['cases'][0]['peak_nits']:.1f} nits, MaxCLL {block['cases'][0]['metrics']['maxcll']}")
        browser.close()
    (OUT / "index.json").write_text(json.dumps(jsonable(index), indent=2), encoding="utf-8", newline="\n")
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"renderer: {index['renderer']}; {len(list(OUT.iterdir()))} files, {total / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
