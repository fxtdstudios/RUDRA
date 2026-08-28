"""Capture the README screenshot of RUDRA Studio, refusing to save a bad one.

Chrome's headless --screenshot exits 0 and writes a perfectly valid PNG when
the page fails to load, so a capture taken while the server is down silently
produces a picture of "This site can't be reached". That is exactly what got
committed on 28 Aug 2026: a 22 KB ERR_CONNECTION_REFUSED page replaced a good
1.36 MB capture, and nothing in the toolchain complained.

So this checks before and after:

  before  /api/model must answer AND report a loaded checkpoint -- a demo-mode
          page would be captured with "NO MODEL" in the header
  after   the PNG must be larger than MIN_BYTES; Chrome's error page is ~20 KB
          while the real UI is over a megabyte

    python ui/capture_shot.py                        # localhost:8422 -> docs/
    python ui/capture_shot.py --port 8081 --out docs/shot.png
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MIN_BYTES = 200_000

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
)


def find_chrome(explicit: str | None) -> str:
    for candidate in ([explicit] if explicit else []) + list(CHROME_CANDIDATES):
        if not candidate:
            continue
        if os.path.isfile(candidate):
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("error: could not find Chrome. Pass --chrome <path>.")


def check_server(port: int) -> dict:
    url = f"http://localhost:{port}/api/model"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            info = json.loads(response.read())
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(
            f"error: {url} is not answering ({exc}).\n"
            f"       Start it first, in its own window:\n"
            f"         python ui/server.py --port {port} --checkpoint <path-to>/best.pt --preload\n"
            f"       Capturing now would screenshot Chrome's error page, which is a\n"
            f"       valid PNG and would sail straight into the README."
        )
    if not info.get("loaded"):
        raise SystemExit(
            f"error: the server is up but has no model loaded "
            f"({info.get('reason', 'unknown')}).\n"
            f"       The page would be captured in demo mode, with NO MODEL in the header."
        )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "rudra_studio.png")
    parser.add_argument("--chrome", default=None)
    # 16:9. The app is laid out for a landscape colour suite; a taller window
    # just stretches the viewer and squeezes the scopes.
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--wait-ms", type=int, default=45000,
                        help="virtual time budget: the model has to finish inferring "
                             "before the frame is grabbed")
    args = parser.parse_args()

    info = check_server(args.port)
    print(f"  server  : {info['gpu']} ({info['device']})")
    print(f"  model   : {info.get('name')}  step {info.get('step')}")

    chrome = find_chrome(args.chrome)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    previous = args.out.stat().st_size if args.out.exists() else 0

    # A temp target, so a failed capture cannot clobber a good committed one.
    staging = args.out.with_suffix(".new.png")
    staging.unlink(missing_ok=True)
    # --disable-gpu was fine when the page was two <img> tags. The viewer now
    # composes in WebGL2 with float render targets, and without a GL backend it
    # renders its own "no WebGL2" error -- a valid PNG of a broken page.
    # SwiftShader gives headless Chrome a real GL2 implementation; the
    # reconstruction itself still runs on the server's GPU.
    command = [
        chrome, "--headless=new", "--hide-scrollbars",
        "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
        f"--window-size={args.width},{args.height}",
        f"--virtual-time-budget={args.wait_ms}",
        f"--screenshot={staging}",
        f"http://localhost:{args.port}/?demo=1",
    ]
    # Chrome refuses to start as root without this, which is every container
    # and most CI runners -- and regenerating the README screenshot is exactly
    # the sort of thing that should be able to run there.
    if os.name == "posix" and hasattr(os, "geteuid") and os.geteuid() == 0:
        command.insert(1, "--no-sandbox")
    subprocess.run(command, check=True)

    if not staging.exists():
        raise SystemExit("error: Chrome wrote no file.")
    size = staging.stat().st_size
    if size < MIN_BYTES:
        staging.unlink(missing_ok=True)
        raise SystemExit(
            f"error: capture is only {size:,} bytes (expected > {MIN_BYTES:,}).\n"
            f"       That is the size of an error or blank page, not the UI.\n"
            f"       {args.out} was left untouched."
        )

    staging.replace(args.out)
    print(f"  wrote   : {args.out}  ({size:,} bytes"
          + (f", was {previous:,}" if previous else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
