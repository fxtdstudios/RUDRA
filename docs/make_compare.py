#!/usr/bin/env python3
"""Build the comparison strips in docs/compare/.

Reads a scored benchmark tree (the one training/run_bench.ps1 writes) and
renders held-out frames as PNG contact sheets a browser can show.

Frames are scene-linear with diffuse white at 1.0, so on an ordinary display
every method looks the same: the part they disagree about sits above white,
where a browser cannot show it.  Each strip therefore stops the HDR columns
down by a fixed number of stops before the sRGB encode, which brings the
highlight range into the visible part of the curve.  The exposure is chosen
per row from the reference alone, never from a method's output, and every
HDR column in a row gets the same one.  The 8-bit input is shown at 0 EV,
the way a viewer would see it.

  python docs/make_compare.py --bench E:/RUDRA_v3_20260822/bench

Captions are read back out of the benchmark's own per-frame result JSON, so
a strip cannot claim a gain the benchmark does not have.  --list-candidates
prints the per-frame margin that picked these frames.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FIGURES = {
    "hard_deployment": {
        "title": "Degraded input: unknown tone curve, 4:2:0 chroma, banding, JPEG",
        "note": "The analytic inverse tone map has nothing above its own ceiling, so it "
                "clips. RUDRA keeps the roll-off the reference has.",
        "condition": "hard",
        "columns": [
            ("sdr", "8-bit input"),
            ("baseline", "analytic inverse ACES"),
            ("shadow_v1", "RUDRA + shadow gate"),
            ("ref", "reference"),
        ],
        "frames": [
            ("lot_01_2k_16c6ec", "0001791_lot_01_2k_c1"),
            ("quarry_cloudy_2k_645dd1", "0001953_quarry_cloudy_2k_c0"),
            ("autumn_park_2k_d8bfe8", "0001417_autumn_park_2k_c1"),
        ],
    },
    "clean_gate": {
        "title": "Well-graded input: what the shadow gate fixes",
        "note": "These are low-headroom frames that need no reconstruction. v5's shadow "
                "arm fires anyway and lifts them; the gate turns it down.",
        "condition": "clean",
        "columns": [
            ("sdr", "8-bit input"),
            ("baseline", "analytic inverse ACES"),
            ("v5", "v5, shipped, no gate"),
            ("shadow_v1", "RUDRA + shadow gate"),
            ("ref", "reference"),
        ],
        "frames": [
            ("studio_small_05_2k_e9dced", "0002145_studio_small_05_2k_c1"),
            ("wooden_studio_17_2k_251cc2", "0002295_wooden_studio_17_2k_c0"),
            ("table_mountain_2_2k_d7d2b4", "0002186_table_mountain_2_2k_c0"),
        ],
    },
    "published_method": {
        "title": "Against a published method: ExpandNet, from the authors' released weights",
        "note": "Run outside its training domain, and given a per-frame exposure fit that "
                "RUDRA does not get. Section 5.1 of the paper carries the caveats.",
        "condition": "clean",
        "columns": [
            ("baseline", "analytic inverse ACES"),
            ("expandnet", "ExpandNet"),
            ("shadow_v1", "RUDRA + shadow gate"),
            ("ref", "reference"),
        ],
        "frames": [
            ("lot_01_2k_16c6ec", "0001791_lot_01_2k_c1"),
            ("quarry_cloudy_2k_645dd1", "0001953_quarry_cloudy_2k_c0"),
            ("table_mountain_2_2k_d7d2b4", "0002186_table_mountain_2_2k_c0"),
        ],
    },
}

HEADER_H = 40
LABEL_H = 20
CAPTION_H = 20
GUTTER = 8
FONT = cv2.FONT_HERSHEY_DUPLEX
BG = 16
NITS_PER_UNIT = 203.0  # BT.2408 diffuse white


def srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def read_linear(path: pathlib.Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"cannot read {path} (OpenEXR support built into cv2?)")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    if path.suffix.lower() == ".png":
        return to_linear(rgb / 255.0)  # display-referred, clipped at diffuse white
    return rgb


def render(lin: np.ndarray, stops: float) -> np.ndarray:
    x = np.clip(lin * (2.0 ** -stops), 0.0, 1.0)
    return (srgb(x) * 255.0 + 0.5).astype(np.uint8)


def choose_stops(ref: np.ndarray) -> float:
    """Pick the exposure from the reference alone, rounded to a whole stop.

    The 99.7th percentile is put just under clipping, so the frame's own
    highlight range is what the strip shows.
    """
    top = float(np.percentile(ref, 99.7))
    return float(np.clip(round(np.log2(max(top, 1e-6))), 0.0, 7.0))


def text(canvas: np.ndarray, s: str, x: int, y: int, scale: float, shade: int) -> None:
    cv2.putText(canvas, s, (x, y), FONT, scale, (shade,) * 3, 1, cv2.LINE_AA)


def per_frame(bench: pathlib.Path, condition: str, name: str, frame: str) -> dict | None:
    path = bench / "results" / f"{condition}_{name}.json"
    if not path.exists():
        return None
    for row in json.loads(path.read_text())["results"]:
        if row["frame"] == frame:
            return row
    return None


def locate(bench: pathlib.Path, condition: str, name: str, clip: str, frame: str) -> pathlib.Path:
    # The SDR inputs are exported once, under clean/, whatever the condition.
    root = bench / ("clean" if name == "sdr" else condition) / name / clip
    for ext in (".exr", ".png"):
        if (root / f"{frame}{ext}").exists():
            return root / f"{frame}{ext}"
    raise SystemExit(f"missing panel: {root / frame}.[exr|png]")


def build(bench: pathlib.Path, spec: dict, width: int) -> np.ndarray:
    cols, frames = spec["columns"], spec["frames"]
    rows = []
    for clip, frame in frames:
        ref = read_linear(locate(bench, spec["condition"], "ref", clip, frame))
        stops = choose_stops(ref)
        base = per_frame(bench, spec["condition"], "baseline", frame)
        imgs, caps, tags = [], [], []
        for name, _ in cols:
            lin = read_linear(locate(bench, spec["condition"], name, clip, frame))
            ev = 0.0 if name == "sdr" else stops
            img = render(lin, ev)
            h, w = img.shape[:2]
            imgs.append(cv2.resize(img, (width, int(round(h * width / w))),
                                   interpolation=cv2.INTER_AREA))
            tags.append("0 EV" if ev == 0 else f"-{ev:.0f} EV")

            got = per_frame(bench, spec["condition"], name, frame)
            if name == "sdr":
                caps.append(f"{frame}   peak {lin.max() * NITS_PER_UNIT:,.0f} nits")
            elif name == "ref":
                caps.append(f"peak {lin.max() * NITS_PER_UNIT:,.0f} nits")
            elif got is None or base is None:
                caps.append("")
            elif name == "baseline":
                caps.append(f"{got['pu_psnr_db']:.1f} dB   {got['cvvdp_jod']:.2f} JOD")
            else:
                caps.append(
                    f"{got['pu_psnr_db']:.1f} dB {got['pu_psnr_db'] - base['pu_psnr_db']:+.1f}"
                    f"   {got['cvvdp_jod']:.2f} JOD {got['cvvdp_jod'] - base['cvvdp_jod']:+.2f}"
                )
        rows.append({"imgs": imgs, "caps": caps, "tags": tags, "frame": frame})

    ph = rows[0]["imgs"][0].shape[0]
    row_h = LABEL_H + ph + CAPTION_H + GUTTER
    total_w = len(cols) * width + (len(cols) - 1) * GUTTER
    canvas = np.full((HEADER_H + len(rows) * row_h - GUTTER, total_w, 3), BG, np.uint8)

    text(canvas, spec["title"], 2, 15, 0.46, 225)
    text(canvas, spec["note"], 2, 31, 0.38, 140)

    for r, row in enumerate(rows):
        top = HEADER_H + r * row_h
        for c, img in enumerate(row["imgs"]):
            x = c * (width + GUTTER)
            shade = 225 if r == 0 else 150
            text(canvas, cols[c][1], x + 2, top + 14, 0.40, shade)
            tag = row["tags"][c]
            (tw, _), _ = cv2.getTextSize(tag, FONT, 0.36, 1)
            text(canvas, tag, x + width - tw - 2, top + 14, 0.36, 110)
            canvas[top + LABEL_H: top + LABEL_H + ph, x: x + width] = img
            if row["caps"][c]:
                text(canvas, row["caps"][c], x + 2, top + LABEL_H + ph + 14, 0.36, 185)
    return canvas


def list_candidates(bench: pathlib.Path, condition: str, a: str, b: str, n: int) -> None:
    def load(name: str) -> dict:
        path = bench / "results" / f"{condition}_{name}.json"
        return {r["frame"]: r for r in json.loads(path.read_text())["results"]}

    left, right = load(a), load(b)
    rows = [(f, right[f]["pu_psnr_db"] - left[f]["pu_psnr_db"],
             right[f]["cvvdp_jod"] - left[f]["cvvdp_jod"]) for f in left]
    rows.sort(key=lambda r: -min(r[1] / 3.0, r[2]))
    for frame, d, j in rows[:n]:
        print(f"{frame:55s} {d:+7.2f} dB {j:+7.3f} JOD")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, type=pathlib.Path, help="scored benchmark root")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/compare"))
    ap.add_argument("--width", type=int, default=360, help="panel width in pixels")
    ap.add_argument("--only", help="build one figure by name")
    ap.add_argument("--list-candidates", nargs=4, metavar=("COND", "A", "B", "N"),
                    help="print the per-frame margin of B over A, best first, and exit")
    args = ap.parse_args()

    if args.list_candidates:
        cond, a, b, n = args.list_candidates
        list_candidates(args.bench, cond, a, b, int(n))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    for key, spec in FIGURES.items():
        if args.only and key != args.only:
            continue
        canvas = build(args.bench, spec, args.width)
        dst = args.out / f"{key}.png"
        cv2.imwrite(str(dst), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"wrote {dst}  {canvas.shape[1]}x{canvas.shape[0]}  "
              f"{dst.stat().st_size / 1000:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
