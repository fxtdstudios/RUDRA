"""Press every control in RUDRA Studio and check that it did something.

A menu item that runs nothing looks exactly like a menu item that works, and
this project shipped four of those on 28 Aug 2026 -- the Reconstruct mode
entries were markup with no handler behind them. Reading the code did not
catch it; pressing the buttons did. So the buttons get pressed.

It also caught a closed overlay whose `display:flex` outranked the `hidden`
attribute, leaving an invisible sheet over the whole page eating every click,
and a pointer capture on a node that its own redraw had already detached.

Needs a running server and playwright's chromium; no GPU (it runs on
SwiftShader) and no HDR display:

    python ui/server.py --port 8100 --no-browser --preload --device cpu
    python tests/ui_smoke/press_everything.py --url http://127.0.0.1:8100/

Exits non-zero if any check fails or the console logs an error.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

results, errors = [], []


def make_frames(directory: Path) -> list:
    """Three frames with a blown specular, a ramp and a crushed foreground --
    something for every control on the page to have an effect on."""
    from PIL import Image
    paths = []
    for i, shift in enumerate([0.0, 0.12, 0.24]):
        height, width = 360, 640
        yy, xx = np.mgrid[0:height, 0:width]
        frame = np.zeros((height, width, 3), np.float32)
        frame[..., 0] = 0.25 + 0.55 * (yy / height) + shift
        frame[..., 1] = 0.30 + 0.55 * (yy / height) + shift
        frame[..., 2] = 0.55 + 0.40 * (yy / height)
        radius = ((xx - (470 - i * 40)) ** 2 + (yy - 90) ** 2) ** 0.5
        frame[radius < 34] = 1.0                    # the censored highlight
        frame[240:, :] *= 0.06                      # crushed shadows
        path = directory / ("frame%02d.png" % i)
        Image.fromarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).save(path)
        paths.append(str(path))
    return paths


def main(URL, FRAMES):
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                                    "--use-angle=swiftshader"])
        c = b.new_context(viewport={"width": 1680, "height": 950},
                          permissions=["clipboard-read", "clipboard-write"])
        pg = c.new_page()
        pg.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
        # A blocked web font is an environment problem, not a page bug; a
        # failed request to our own origin is very much a page bug.
        pg.on("console", lambda m: errors.append("console.error: " + m.text)
              if m.type == "error" and "Failed to load resource" not in m.text else None)
        pg.on("requestfailed", lambda r: errors.append("request failed: " + r.url)
              if r.url.startswith(URL) else None)
        pg.goto(URL); pg.wait_for_timeout(1200)

        def canvas():
            try:
                return hashlib.sha1(pg.evaluate(
                    "() => document.getElementById('gl').toDataURL()").encode()).hexdigest()[:10]
            except Exception:
                return "no-canvas"
        def logtail(n=1):
            t = pg.inner_text("#log").strip().splitlines()
            return " / ".join(t[-n:]) if t else ""
        def record(name, ok, detail=""):
            results.append((name, "OK" if ok else "FAIL", detail))

        # ---- load three frames -------------------------------------------
        pg.set_input_files("#file", FRAMES)
        pg.wait_for_function("() => !document.getElementById('plate').hidden", timeout=120000)
        pg.wait_for_timeout(2500)
        record("File input: 3 frames", pg.inner_text("#shotCount") == "3",
               "shotCount=" + pg.inner_text("#shotCount"))
        record("Frames rail lists them",
               len(pg.query_selector_all(".shot")) == 3)
        record("Timecode counts frames", pg.inner_text("#tc") == "001 / 003",
               pg.inner_text("#tc"))

        # ---- menus: every item ------------------------------------------
        titles = pg.eval_on_selector_all(".menu .mtitle", "els => els.map(e => e.textContent)")
        record("Menubar has 8 menus", len(titles) == 8, ", ".join(titles))
        for i, title in enumerate(titles):
            pg.click(".menu:nth-child(%d) .mtitle" % (i + 1))
            pg.wait_for_timeout(120)
            opened = pg.eval_on_selector(".menu:nth-child(%d)" % (i + 1),
                                         "e => e.classList.contains('open')")
            items = pg.eval_on_selector_all(
                ".menu:nth-child(%d) .drop button" % (i + 1),
                "els => els.map(e => ({act: e.dataset.act, disabled: e.disabled}))")
            record("Menu %-12s opens with %d items" % (title, len(items)),
                   opened and len(items) > 0)
            pg.keyboard.press("Escape")

        acts = pg.eval_on_selector_all(".menu .drop button[data-act]",
                                       "els => [...new Set(els.map(e => e.dataset.act))]")
        missing = pg.evaluate("() => []")
        record("Every item carries an action", len(acts) >= 25, "%d distinct actions" % len(acts))
        record("No orphan menu items",
               "no action" not in pg.inner_text("#log") and
               "menu items with no action" not in pg.inner_text("#log"))

        def menu_click(act):
            """Open the owning menu the way a person does, then click the item."""
            pg.keyboard.press("Escape")
            idx = pg.evaluate(
                "a => [...document.querySelectorAll('.menu')].indexOf("
                "document.querySelector('.menu .drop button[data-act=\"'+a+'\"]')"
                ".closest('.menu')) + 1", act)
            pg.click(".menu:nth-child(%d) .mtitle" % idx)
            pg.wait_for_timeout(140)
            item = '.menu:nth-child(%d) .drop button[data-act="%s"]' % (idx, act)
            if pg.eval_on_selector(item, "e => e.disabled"):
                record("menu item " + act, False, "disabled when it should be usable")
                pg.keyboard.press("Escape")
                return
            pg.click(item)
            pg.wait_for_timeout(600)

        # ---- Clip menu / transport ---------------------------------------
        menu_click("next")
        record("Clip ▸ Next frame", pg.inner_text("#tc") == "002 / 003", pg.inner_text("#tc"))
        menu_click("last")
        record("Clip ▸ Last frame", pg.inner_text("#tc") == "003 / 003", pg.inner_text("#tc"))
        menu_click("first")
        record("Clip ▸ First frame", pg.inner_text("#tc") == "001 / 003", pg.inner_text("#tc"))
        pg.click("#btnNext"); pg.wait_for_timeout(1200)
        record("Transport ▸ next button", pg.inner_text("#tc") == "002 / 003", pg.inner_text("#tc"))
        pg.click("#btnPrev"); pg.wait_for_timeout(1200)
        record("Transport ▸ prev button", pg.inner_text("#tc") == "001 / 003", pg.inner_text("#tc"))
        box = pg.query_selector("#scrub").bounding_box()
        pg.mouse.click(box["x"] + box["width"] - 4, box["y"] + box["height"] / 2)
        pg.wait_for_timeout(1500)
        record("Scrub bar seeks", pg.inner_text("#tc") == "003 / 003", pg.inner_text("#tc"))
        pg.click(".shot:nth-child(1)"); pg.wait_for_timeout(1200)
        record("Frames rail row selects", pg.inner_text("#tc") == "001 / 003", pg.inner_text("#tc"))
        pg.click("#btnPlay"); pg.wait_for_timeout(900)
        playing = pg.eval_on_selector("#btnPlay", "e => e.classList.contains('on')")
        pg.wait_for_timeout(1800); pg.click("#btnPlay"); pg.wait_for_timeout(300)
        record("Transport ▸ play/pause", playing and
               not pg.eval_on_selector("#btnPlay", "e => e.classList.contains('on')"))

        # ---- Reconstruct menu --------------------------------------------
        before = canvas()
        menu_click("mode-shadows")
        record("Reconstruct ▸ Shadows", canvas() != before and
               pg.eval_on_selector('#mode button[data-mode="shadows"]',
                                   "e => e.classList.contains('on')"))
        before = canvas(); menu_click("mode-all")
        record("Reconstruct ▸ All", canvas() != before)
        before = canvas(); menu_click("strength-up")
        record("Reconstruct ▸ Stronger residual",
               canvas() != before and pg.inner_text("#strengthVal") == "1.10",
               pg.inner_text("#strengthVal"))
        before = canvas(); menu_click("preserve")
        record("Reconstruct ▸ Preserve toggle", canvas() != before)
        menu_click("preserve")

        # ---- Edit menu ----------------------------------------------------
        before = canvas(); menu_click("undo")
        record("Edit ▸ Undo", "undo" in logtail(3))
        menu_click("redo")
        record("Edit ▸ Redo", "redo" in logtail(3))
        menu_click("reset-recon")
        record("Edit ▸ Reset reconstruction", pg.inner_text("#strengthVal") == "1.00",
               pg.inner_text("#strengthVal"))

        # ---- Region EV ----------------------------------------------------
        record("Region rows rendered", len(pg.query_selector_all("#regions .region")) == 3)
        before = canvas()
        ev = pg.query_selector("#regions .region:nth-child(1) .ev").bounding_box()
        pg.mouse.move(ev["x"] + ev["width"] / 2, ev["y"] + ev["height"] / 2)
        pg.mouse.down(); pg.mouse.move(ev["x"] + ev["width"] / 2 + 90,
                                       ev["y"] + ev["height"] / 2, steps=8)
        pg.mouse.up(); pg.wait_for_timeout(900)
        value = pg.inner_text("#regions .region:nth-child(1) .ev")
        record("Region EV drag grades", canvas() != before and value != "+0.00", "EV " + value)
        record("Plate label says it is graded",
               "region ev" in pg.inner_text("#plateLabel").lower(), pg.inner_text("#plateLabel"))
        graded_metrics = pg.inner_text("#measA").replace("\n", " ")
        menu_click("reset-regions")
        record("Edit ▸ Reset region EV",
               pg.inner_text("#regions .region:nth-child(1) .ev") == "+0.00")

        # ---- Measure menu --------------------------------------------------
        menu_click("copy-metrics")
        clip = pg.evaluate("() => navigator.clipboard.readText()")
        record("Measure ▸ Copy measurements", "maxcll" in clip, clip[:44].replace("\n", " "))
        menu_click("copy-scopes")
        clip = pg.evaluate("() => navigator.clipboard.readText()")
        record("Measure ▸ Copy scope data", "histogram" in clip)
        menu_click("remeasure")
        record("Measure ▸ Re-measure", "re-measured" in logtail(2))

        # ---- Deliver menu --------------------------------------------------
        menu_click("container-linear")
        record("Deliver ▸ Container linear",
               "linear Rec.2020" in pg.inner_text("#containerField") and
               pg.inner_text("#primariesField") == "Rec.2020",
               pg.inner_text("#containerField"))
        menu_click("container-aces")
        record("Deliver ▸ Container ACES",
               "ACES 2065-1" in pg.inner_text("#containerField"))
        menu_click("copy-delivery")
        clip = json.loads(pg.evaluate("() => navigator.clipboard.readText()"))
        record("Deliver ▸ Copy delivery metadata",
               clip.get("container", "").startswith("ACES") and "measurements" in clip)

        # ---- Window menu ---------------------------------------------------
        menu_click("rail-left")
        record("Window ▸ Frames rail hides",
               pg.eval_on_selector("#railLeft", "e => e.classList.contains('hidden')"))
        menu_click("rail-left")
        menu_click("scopes")
        record("Window ▸ Scopes hide",
               pg.eval_on_selector("#scopes", "e => e.classList.contains('hidden')"))
        menu_click("scopes")
        menu_click("zoom-actual")
        record("Window ▸ Actual pixels",
               pg.eval_on_selector("#viewer", "e => e.classList.contains('actual')"))
        menu_click("zoom-fit")
        menu_click("rail-right")
        record("Window ▸ Reconstruction rail hides",
               pg.eval_on_selector("#railRight", "e => e.classList.contains('hidden')"))
        menu_click("rail-right")

        # ---- Help menu ------------------------------------------------------
        menu_click("shortcuts")
        record("Help ▸ Keyboard shortcuts",
               not pg.eval_on_selector("#overlay", "e => e.hidden") and
               "Flip" in pg.inner_text("#overlaySheet"))
        pg.keyboard.press("Escape")
        menu_click("about")
        record("Help ▸ About",
               "Checkpoint" in pg.inner_text("#overlaySheet"))
        pg.keyboard.press("Escape")

        # ---- keyboard --------------------------------------------------------
        before = canvas(); pg.keyboard.press("2"); pg.wait_for_timeout(500)
        record("Key 2 → highlights", canvas() != before)
        pg.keyboard.press("1"); pg.wait_for_timeout(400)
        before = canvas(); pg.keyboard.press("]"); pg.wait_for_timeout(500)
        record("Key ] → stronger", canvas() != before)
        pg.keyboard.press("["); pg.wait_for_timeout(400)
        pg.keyboard.press("?"); pg.wait_for_timeout(300)
        record("Key ? → shortcuts", not pg.eval_on_selector("#overlay", "e => e.hidden"))
        pg.keyboard.press("Escape")
        # The index has drifted through the play test, so assert the step, not
        # an absolute position -- "." wraps 003 back to 001, which is correct.
        was = int(pg.inner_text("#tc").split("/")[0])
        pg.keyboard.press("."); pg.wait_for_timeout(1500)
        now = int(pg.inner_text("#tc").split("/")[0])
        record("Key . → next frame", now == (was % 3) + 1,
               "%03d -> %03d" % (was, now))
        pg.keyboard.press(","); pg.wait_for_timeout(1200)
        before = canvas(); pg.keyboard.down("b"); pg.wait_for_timeout(400)
        held = canvas(); pg.keyboard.up("b"); pg.wait_for_timeout(400)
        record("Key B → hold flip", held != before and canvas() == before)

        # ---- Master with a grade ---------------------------------------------
        ev = pg.query_selector("#regions .region:nth-child(3) .ev").bounding_box()
        pg.mouse.move(ev["x"] + ev["width"] / 2, ev["y"] + ev["height"] / 2)
        pg.mouse.down(); pg.mouse.move(ev["x"] + ev["width"] / 2 - 70,
                                       ev["y"] + ev["height"] / 2, steps=6)
        pg.mouse.up(); pg.wait_for_timeout(900)
        pg.click("#btnMaster"); pg.wait_for_timeout(30000)
        tail = pg.inner_text("#log")
        record("Master EXR writes a graded file",
               "MaxCLL" in tail and "failed" not in tail.splitlines()[-1], logtail(2))

        # ---- File ▸ Close all --------------------------------------------------
        menu_click("close")
        record("File ▸ Close all frames",
               pg.eval_on_selector("#plate", "e => e.hidden") and
               pg.inner_text("#shotCount") == "0")

        b.close()

    width = max(len(r[0]) for r in results)
    print()
    for name, verdict, detail in results:
        print("  %-*s  %-4s %s" % (width, name, verdict, detail))
    bad = [r for r in results if r[1] == "FAIL"]
    print("\n  %d checks, %d failed" % (len(results), len(bad)))
    print("\n=== console ===")
    seen = set()
    for e in errors:
        if e not in seen:
            seen.add(e); print("  ", e[:160])
    if not seen: print("   clean")
    return 1 if bad or seen else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8100/")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(main(args.url, make_frames(Path(tmp))))
