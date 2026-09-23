"""Golden files for the native core, emitted by the Python it ports.

Principle P4 in docs/NATIVE_ARCHITECTURE.md: Python is the oracle. Every C++
module in native/core is tested against arrays this script writes from the
code it replaces, so a change on the Python side shows up as a failing native
test instead of as a silent disagreement between the app and the paper.

    python tools/emit_core_golden.py            # writes native/tests/golden/core/

Deterministic; the files are small and committed. Re-run and commit when the
Python they come from changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import CurveHead, sdr_to_baseline_hdr  # noqa: E402
from training.infer_sdr2hdr import _tile_starts, _tile_weight  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "core"

TILE_CASES = [
    # (full_h, full_w, tile, overlap)
    (128, 300, 128, 32),
    (720, 1280, 512, 64),
    (61, 97, 32, 8),
    (512, 512, 512, 64),
    (1080, 1920, 512, 64),
]


def save(name: str, arr) -> dict:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    np.save(OUT / f"{name}.npy", a, allow_pickle=False)
    return {"file": f"{name}.npy", "shape": list(a.shape)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict = {"baseline": [], "curve": None, "tiles": []}

    rng = np.random.default_rng(20260923)
    codes = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    ramp = np.stack([codes, codes[::-1], np.roll(codes, 337)])[None, :, None, :]  # (1,3,1,1024)
    img = rng.random((1, 3, 24, 32), dtype=np.float32)
    index["inputs"] = {"ramp": save("in_ramp", ramp), "image": save("in_image", img)}
    for ev in (-1.0, 0.0, 1.0):
        tag = f"ev{ev:+.0f}".replace("+", "p").replace("-", "m")
        for src, arr in (("ramp", ramp), ("image", img)):
            out = sdr_to_baseline_hdr(torch.from_numpy(arr), ev).numpy()
            index["baseline"].append({"corpus_ev": ev, "input": src,
                                      "expected": save(f"baseline_{src}_{tag}", out)})

    torch.manual_seed(5)
    head = CurveHead()
    params = (torch.rand(1, 1 + head.knots) - 0.5) * 3.0
    corr = head.correction_log2(torch.from_numpy(img), params).numpy()
    index["curve"] = {"knots": head.knots, "params": save("curve_params", params.numpy()),
                      "input": "image", "expected": save("curve_correction", corr)}

    for full_h, full_w, tile, overlap in TILE_CASES:
        ys = _tile_starts(full_h, tile, overlap)
        xs = _tile_starts(full_w, tile, overlap)
        case = {"full_h": full_h, "full_w": full_w, "tile": tile, "overlap": overlap,
                "ys": ys, "xs": xs, "weights": []}
        # Every distinct tile position class: first, interior, last in each axis.
        for yi in sorted({0, len(ys) // 2, len(ys) - 1}):
            for xi in sorted({0, len(xs) // 2, len(xs) - 1}):
                y, x = ys[yi], xs[xi]
                th, tw = min(tile, full_h - y), min(tile, full_w - x)
                w = _tile_weight(th, tw, overlap, y, x, full_h, full_w, torch.device("cpu"))
                name = f"tw_{full_h}x{full_w}_t{tile}_o{overlap}_y{y}_x{x}"
                w2 = w[0, 0].numpy()
                # The weight is an outer product, so its centre row and column
                # pin it; the full grid is stored only for small tiles.
                entry = {"y": y, "x": x, "h": th, "w": tw,
                         "row": save(name + "_row", w2[th // 2]),
                         "col": save(name + "_col", w2[:, tw // 2])}
                if tile <= 128:
                    entry["full"] = save(name, w2)
                case["weights"].append(entry)
        index["tiles"].append(case)

    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {len(list(OUT.glob('*.npy')))} arrays to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
