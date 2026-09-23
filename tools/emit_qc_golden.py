"""Golden reports for the native QC (Phase 1 step 8, native/deliver qc.cpp).

The oracle is rudra/qc.py check_frame and format_report on six synthetic
reconstructions chosen to walk every branch: an anchored one with a hard
clip edge, a lifted one with NaN and negative samples, a noisy one with
nothing clipped, one too small to measure, an identity one, and a smooth
recovery that passes every check. The native test must produce the same statuses, the
same values and the same report text.

    python tools/emit_qc_golden.py        # writes native/tests/golden/qc/
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.qc import DEFAULT_THRESHOLDS, check_frame, format_report, load_thresholds, srgb_to_linear  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "qc"


def scene(rng, h, w, clip=True):
    y, x = np.mgrid[:h, :w]
    t = y / (h - 1)
    img = np.stack([0.5 + 0.4 * (1 - t), 0.6 + 0.3 * (1 - t), 0.8 + 0.18 * (1 - t)], -1)
    img[(y > h * 0.6) & (x < w * 0.5)] = [0.55, 0.35, 0.25]
    if clip:
        img[(y - h * 0.2) ** 2 + (x - w * 0.75) ** 2 < (h * 0.15) ** 2] = 1.0
    img = img + rng.normal(0, 0.004, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def recover(sdr, rng, lift=1.0, noise=0.0):
    src = srgb_to_linear(sdr.astype(np.float64)) * 203.0
    hdr = src * lift
    clipped = sdr.max(-1) >= 254 / 255
    tex = np.broadcast_to(1.0 + 0.3 * np.sin(np.arange(sdr.shape[1]) / 3.0)[None, :], sdr.shape[:2])
    hdr[clipped] *= (3.0 * tex[clipped])[:, None]
    if noise:
        hdr = hdr * (1.0 + rng.normal(0, noise, hdr.shape))
    return hdr


def dome(sdr):
    """A smooth, textured recovery that fades to identity at the clip edge: passes everything."""
    h, w = sdr.shape[:2]
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - h * 0.2) ** 2 + (x - w * 0.75) ** 2)
    gain = 1.0 + 4.0 * np.clip(1.0 - r / (h * 0.15), 0.0, 1.0) ** 2 * (1.0 + 0.2 * np.sin(x / 3.0))
    return srgb_to_linear(sdr.astype(np.float64)) * 203.0 * gain[..., None]


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(20260924)
    shutil.copy(DEFAULT_THRESHOLDS, OUT / "qc_reconstruction.json")
    t = load_thresholds(DEFAULT_THRESHOLDS)
    cases = []
    good = scene(rng, 128, 176)
    lifted = scene(rng, 128, 176)
    noisy = scene(rng, 128, 176, clip=False)
    tiny = scene(rng, 40, 56)
    plain = scene(rng, 128, 176, clip=False)
    clean = scene(rng, 128, 176)
    specs = [("anchored", good, recover(good, rng)),
             ("lifted_nan", lifted, recover(lifted, rng, lift=2.0)),
             ("noisy_unclipped", noisy, recover(noisy, rng, noise=0.05)),
             ("tiny", tiny, recover(tiny, rng)),
             ("identity_unclipped", plain, recover(plain, rng)),
             ("clean_pass", clean, dome(clean))]
    specs[1][2][5, 7] = [np.nan, 10.0, -1.0]
    specs[1][2][9, 9, 2] = -3.0
    for name, sdr, hdr in specs:
        # Rounded to float32 first, so the stored file is the exact input (and half the size).
        hdr = hdr.astype(np.float32).astype(np.float64)
        rep = check_frame(hdr, sdr, thresholds=t)
        np.save(OUT / f"{name}_sdr.npy", np.ascontiguousarray(np.transpose(sdr, (2, 0, 1))), allow_pickle=False)
        np.save(OUT / f"{name}_hdr.npy", np.ascontiguousarray(np.transpose(hdr, (2, 0, 1)).astype(np.float32)),
                allow_pickle=False)
        text = format_report(rep, f"{name}.png", f"{sdr.shape[1]}x{sdr.shape[0]}")
        (OUT / f"{name}_report.txt").write_text(text, encoding="utf-8")
        cases.append({"name": name, "sdr": f"{name}_sdr.npy", "hdr": f"{name}_hdr.npy",
                      "report_text": f"{name}_report.txt",
                      "report": {"verdict": rep.verdict,
                                 "checks": [{"name": c.name, "status": c.status,
                                             "value": None if c.value is None else
                                             ("nan" if np.isnan(c.value) else float(c.value)),
                                             "threshold": c.threshold, "detail": c.detail}
                                            for c in rep.checks]}})
    (OUT / "index.json").write_text(json.dumps({"oracle": "rudra/qc.py", "thresholds": "qc_reconstruction.json",
                                                "cases": cases}, indent=2))
    for c in cases:
        print(c["name"], c["report"]["verdict"], [(k["name"], k["status"]) for k in c["report"]["checks"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
