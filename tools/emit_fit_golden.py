"""Golden preview downscales for the native app (the Studio's max_side).

The oracle is ui/server.py _fit: a frame larger than max_side on its long
side goes to int(side * ratio) with cv2.INTER_AREA in float32. The inputs are
the decode goldens' own decoded frames (native/tests/golden/decode/*.npy), fit
at several sizes, integer and fractional ratios among them; media/still.cpp
fit_max_side must give the same floats.

    python tools/emit_fit_golden.py   # writes native/tests/golden/fit/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DECODE = REPO / "native" / "tests" / "golden" / "decode"
OUT = REPO / "native" / "tests" / "golden" / "fit"
CASES = [("png8_rgb", 32), ("png8_rgb", 20), ("png16_rgb", 11), ("jpeg_q92_420", 17), ("bmp24", 25), ("png8_rgb", 0), ("png8_rgb", 37)]


def main() -> int:
    from ui.server import _fit
    import cv2

    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, side in CASES:
        chw = np.load(DECODE / f"{name}.npy").astype(np.float32)   # the decode goldens are 3 x H x W
        rgb = np.ascontiguousarray(chw.transpose(1, 2, 0))
        out = _fit(rgb, side).transpose(2, 0, 1)                  # back to 3 x H x W
        file = f"{name}_{side}.npy"
        np.save(OUT / file, np.ascontiguousarray(out, dtype=np.float32))
        cases.append({"input": f"{name}.npy", "max_side": side, "expected": file,
                      "in_shape": list(chw.shape), "out_shape": list(out.shape)})
    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "ui/server.py _fit (cv2 %s INTER_AREA, float32)" % cv2.__version__, "cases": cases}, f, indent=1)
        f.write("\n")
    print(f"fit: {len(cases)} cases -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
