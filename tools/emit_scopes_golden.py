"""Golden scope drawings for the native scope widgets (Phase 3 step 5).

The page draws the waveform and the histogram as SVG from the scope data
(drawScopes in ui/app.js): gridlines and their labels, the diffuse-white
line, the envelope and interquartile bands, the median spine, the ceiling
band, MaxCLL, and the zone-coloured histogram bars. That markup is a display
list, so the oracle is exact: the whole page runs headless (as for the
session golden), drawScopes runs on scope data given here, and every element
it made is recorded with its attributes and text. core/scope_draw.cpp must
build the same list from the same data; the widgets paint that list.

The vectorscope's picture is Phase 2's (core/scopes.cpp vectorscope, equal to
the page's canvas); its frame -- the ring, the inner ring and the six hue
labels -- is recorded from the page's markup and styles.

    python tools/emit_scopes_golden.py        # writes native/tests/golden/scopes/ (drawings.json, rasters)

The rasters (<case>_wave.png, <case>_hist.png, vector.png) are element
screenshots of the page with Plex served from native/app/fonts; the app's
widgets are held to them within 2 codes on 99 % of pixels. They depend on the
browser's rasteriser, so CI re-emits and diffs drawings.json only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.emit_session_golden import serve  # noqa: E402

FONTS = REPO / "native" / "app" / "fonts"
# The page asks Google Fonts for Plex; the harness answers with the app's own
# files, so the rasters show the same glyphs the app draws.
FONT_CSS = "".join(
    "@font-face{font-family:'%s';font-weight:%d;src:url(http://fonts.test/%s) format('truetype')}"
    % (fam, w, f) for fam, w, f in [
        ("IBM Plex Mono", 400, "IBMPlexMono-Regular.ttf"), ("IBM Plex Mono", 500, "IBMPlexMono-Medium.ttf"),
        ("IBM Plex Sans Condensed", 400, "IBMPlexSansCondensed-Regular.ttf"),
        ("IBM Plex Sans Condensed", 500, "IBMPlexSansCondensed-Medium.ttf"),
        ("IBM Plex Sans Condensed", 600, "IBMPlexSansCondensed-SemiBold.ttf"),
        ("IBM Plex Sans Condensed", 700, "IBMPlexSansCondensed-Bold.ttf")])


def fonts(route) -> None:
    url = route.request.url
    if "fonts.googleapis.com" in url:
        route.fulfill(status=200, content_type="text/css", body=FONT_CSS)
    elif "fonts.test/" in url:
        route.fulfill(status=200, content_type="font/ttf", body=(FONTS / url.rsplit("/", 1)[1]).read_bytes())
    else:
        route.abort()


def vector_sample(w: int = 48, h: int = 32) -> list[float]:
    """An RGBA float sample (the network convention, 1.0 = 10 000 nits) that
    spreads hues around the vectorscope: a hue wheel at several levels."""
    import math
    out = []
    for y in range(h):
        for x in range(w):
            a = 2 * math.pi * x / w
            level = [0.0001, 0.002, 0.02, 0.2][y * 4 // h]
            sat = 0.2 + 0.8 * (y % 8) / 7
            rgb = [level * (1 + sat * math.cos(a + k * 2 * math.pi / 3)) / 2 for k in range(3)]
            out += [float(v) for v in rgb] + [1.0]
    return out

OUT = REPO / "native" / "tests" / "golden" / "scopes"
COLUMNS, COL_BINS, HIST_BINS = 230, 512, 76


def lcg(seed: int):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield (state >> 11) / float(1 << 53)


def scope_data(seed: int, clip_every: int = 0, sparse: bool = False) -> dict:
    """Scope data in the page's form: quantiles are (k + 0.5) / 512 or 1, the
    histogram is normalised to its tallest bin."""
    r = lcg(seed)
    s = {k: [] for k in ("lo", "q1", "mid", "q3", "hi")}
    for c in range(COLUMNS):
        base = 0.15 + 0.5 * (c / COLUMNS) + 0.08 * next(r)
        qs = sorted(min(0.998, max(0.0, base + d * 0.12 + 0.05 * (next(r) - 0.5))) for d in (-2, -1, 0, 1, 2))
        ks = [int(q * COL_BINS) for q in qs]
        vals = [(k + 0.5) / COL_BINS for k in ks]
        if clip_every and c % clip_every == 0:
            vals[4] = 1
        for key, v in zip(("lo", "q1", "mid", "q3", "hi"), vals):
            s[key].append(v)
    hist = []
    for b in range(HIST_BINS):
        v = next(r) ** 2
        if sparse and b % 3 == 0:
            v = 0.003 * next(r)          # at or under the 0.004 the page skips
        hist.append(v)
    peak = max(hist)
    s["histogram"] = [v / peak for v in hist]
    return s


CASES = [
    ("plain", scope_data(1), None),
    ("clipped_with_maxcll", scope_data(2, clip_every=7), {"maxcll": 1234.56}),
    ("sparse_histogram", scope_data(3, sparse=True), {"maxcll": None}),
    ("maxcll_off_the_scale", scope_data(4, clip_every=2), {"maxcll": 25000.0}),
    ("maxcll_below_the_floor", scope_data(5), {"maxcll": 0.001}),
]

RUN = r"""
(c) => {
  const S = window.__studio;
  const m = c.m === null ? null : {maxcll: c.m.maxcll === null ? NaN : c.m.maxcll};
  S.drawScopes(c.s, m);
  const walk = (el) => Array.from(el.children).map((e) => {
    const attrs = {};
    for (const a of e.attributes) attrs[a.name] = a.value;
    const out = {tag: e.tagName.toLowerCase(), attrs: attrs};
    if (e.tagName.toLowerCase() === "text") out.text = e.textContent;
    if (e.children.length && e.tagName.toLowerCase() !== "text") out.children = walk(e);
    return out;
  });
  const svg = (id) => { const e = document.getElementById(id);
    return {viewBox: e.getAttribute("viewBox"), width: e.getAttribute("width"), height: e.getAttribute("height"),
            preserveAspectRatio: e.getAttribute("preserveAspectRatio"), elements: walk(e)}; };
  return {wave: svg("wave"), hist: svg("hist")};
}
"""

VECTOR = r"""
() => {
  const plot = document.querySelector(".plot.vs");
  const cv = document.getElementById("vector"), ring = plot.querySelector(".vsring");
  const cs = (e, p) => getComputedStyle(e).getPropertyValue(p);
  const before = getComputedStyle(ring, "::before");
  return {canvas: [cv.width, cv.height], display: [cv.clientWidth, cv.clientHeight],
          canvas_background: cs(cv, "background-color"),
          ring: {border: cs(ring, "border-top-color"), size: [ring.offsetWidth, ring.offsetHeight]},
          inner_ring: {border: before.getPropertyValue("border-top-color"), inset: before.getPropertyValue("top")},
          labels: Array.from(plot.querySelectorAll("b")).map((b) => ({text: b.textContent,
            left: b.style.left, top: b.style.top, color: cs(b, "color"), size: cs(b, "font-size")}))};
}
"""


def main() -> int:
    from playwright.sync_api import sync_playwright

    out = {"oracle": "ui/app.js drawScopes (SVG markup) and the page's vectorscope frame", "cases": []}
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.route("https://fonts.googleapis.com/**", fonts)
        page.route("https://fonts.gstatic.com/**", lambda r: r.abort())
        page.route("http://fonts.test/**", fonts)
        page.route("http://studio.test/**", serve)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto("http://studio.test/index.html")
        page.wait_for_function("window.__studio !== undefined")
        page.evaluate("document.fonts.ready")
        page.wait_for_function("document.fonts.check('8px \"IBM Plex Mono\"')")
        for name, s, m in CASES:
            drawn = page.evaluate(RUN, {"s": s, "m": m})
            page.wait_for_timeout(30)
            for part in ("wave", "hist"):
                el = page.locator("#" + part)
                el.screenshot(path=str(OUT / f"{name}_{part}.png"))
                # The same with the labels hidden: the geometry and colour on
                # their own (text is compared exactly in the display list).
                page.evaluate("(id) => document.querySelectorAll('#' + id + ' text')"
                              ".forEach((t) => t.style.display = 'none')", part)
                el.screenshot(path=str(OUT / f"{name}_{part}_notext.png"))
                drawn[part]["box"] = page.evaluate("(id) => { const r = document.getElementById(id)"
                                                   ".getBoundingClientRect(); return [r.width, r.height]; }", part)
            out["cases"].append({"name": name, "scopes": s, "metrics": m, "drawn": drawn})
        out["vector"] = page.evaluate(VECTOR)
        # The vectorscope, drawn from a sample by the page's drawVector.
        sample = vector_sample()
        page.evaluate("(p) => window.__studio.drawVector({width: p.w, height: p.h, model: new Float32Array(p.v)})",
                      {"w": 48, "h": 32, "v": sample})
        page.wait_for_timeout(30)
        page.locator(".plot.vs").screenshot(path=str(OUT / "vector.png"))
        out["vector"]["sample"] = {"width": 48, "height": 32, "rgba": sample}
        out["vector"]["plot_box"] = page.evaluate("() => { const r = document.querySelector('.plot.vs')"
                                                  ".getBoundingClientRect(); return [r.width, r.height]; }")
        browser.close()
        if errors:
            print(f"page errors: {errors}")
            return 1
    with open(OUT / "drawings.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    n = sum(len(c["drawn"]["wave"]["elements"]) + len(c["drawn"]["hist"]["elements"]) for c in out["cases"])
    print(f"scopes: {len(CASES)} cases, {n} top-level elements -> {OUT / 'drawings.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
