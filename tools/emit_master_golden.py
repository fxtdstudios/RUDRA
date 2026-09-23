"""Golden masters for `rudra-native master` (Phase 1 step 7).

The oracle is the Studio's own master path, ui/server.py _render_master,
called directly on the shipped checkpoint on CPU: decode, reconstruct, Region
EV, anchor, chroma carry, measure, ACES or linear EXR, sidecar. The native
tool renders the same stills from the exported model package and
`rudra-native master-check` compares the EXR pixels (within 1 half-float ulp),
the EXR header and the sidecar.

    python tools/emit_master_golden.py        # writes native/tests/golden/master/

The fixture stills are inputs: once committed they are reused.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.infer_sdr2hdr import load_models  # noqa: E402
from ui.server import _render_master  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "master"
CHECKPOINT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"
H, W = 97, 131

GRADED = [
    {"label": "highlights", "low_nits": 400.0, "high_nits": 2000.0, "ev": 0.6},
    {"label": "speculars", "low_nits": 2000.0, "high_nits": 8000.0, "ev": -0.4},
    {"label": "shadows", "low_nits": 0.05, "high_nits": 12.0, "ev": 0.8},
]


def still(rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) sRGB codes: sky ramp, clipped sun, warm wall, deep shadow."""
    y, x = np.mgrid[:H, :W]
    img = np.zeros((H, W, 3))
    t = y / (H - 1)
    img[..., 0] = 0.55 + 0.4 * (1 - t)
    img[..., 1] = 0.65 + 0.3 * (1 - t)
    img[..., 2] = 0.85 + 0.15 * (1 - t)
    sun = (y - 18) ** 2 + (x - 95) ** 2 < 14 ** 2
    img[sun] = 1.0
    wall = (y > 55) & (x < 70)
    img[wall] = np.stack([0.75 - 0.3 * x[wall] / 70, 0.45 - 0.2 * x[wall] / 70, 0.3 - 0.1 * x[wall] / 70], -1)
    img[(y > 80) & (x >= 70)] = 0.03
    img += rng.normal(0, 0.008, img.shape)
    return np.clip(img, 0.0, 1.0)


FIXTURES = {
    "still_png8.png": lambda rgb: cv2.imencode(".png", (rgb[..., ::-1] * 255).round().astype(np.uint8))[1],
    "still_png16.png": lambda rgb: cv2.imencode(".png", (rgb[..., ::-1] * 65535).round().astype(np.uint16))[1],
    "still_q95.jpg": lambda rgb: cv2.imencode(".jpg", (rgb[..., ::-1] * 255).round().astype(np.uint8),
                                               [cv2.IMWRITE_JPEG_QUALITY, 95])[1],
}

CASES = [
    ("default", "still_png8.png", {}),
    ("graded_linear", "still_png16.png", {"regions": GRADED, "region_softness_stops": 0.5, "container": "linear"}),
    ("plain_highlights", "still_q95.jpg", {"anchor": False, "carry_chroma": False, "recovery_mode": "highlights",
                                           "strength": 1.4, "preserve_outside": False}),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260924)
    for name, enc in FIXTURES.items():
        path = OUT / name
        if not path.exists():
            path.write_bytes(enc(still(rng)).tobytes())
    for old in OUT.glob("case_*"):
        old.unlink()

    torch.set_num_threads(1)
    model, _ = load_models(str(CHECKPOINT), None, torch.device("cpu"))
    cases = []
    for name, image, extra in CASES:
        params = {"checkpoint": CHECKPOINT.name, **extra}
        out = OUT / f"case_{name}.exr"
        result = _render_master(model, (OUT / image).read_bytes(), params, None, out)
        cases.append({"name": name, "image": image, "params": params, "exr": out.name,
                      "sidecar": out.with_suffix(".json").name,
                      "maxcll": result["maxcll"], "maxfall": result["maxfall"]})
    (OUT / "index.json").write_text(json.dumps({"oracle": "ui/server.py _render_master",
                                                "checkpoint": CHECKPOINT.name, "cases": cases}, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} master cases to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
