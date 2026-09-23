"""Golden files for the native composite, master chain and measurements.

Principle P4 in docs/NATIVE_ARCHITECTURE.md: Python is the oracle. The native
composite (native/core composite.hpp), the master pixel chain (master.hpp) and
the measurements (measure.hpp) are tested against arrays this script writes
from the functions they port:

    composite   training.infer_sdr2hdr.predict_image, fed the fields
                predict_fields returns for the same frame
    region EV   rudra.delivery.controls.apply_region_ev (+ the ceiling clip
                ui/server.py _render_master applies)
    anchor      rudra.anchor.anchor_to_sdr
    chroma      rudra.chroma.carry_source_chroma
    gamut       rudra.delivery.colorspace.rgb_to_rgb_matrix / convert
    master      the stage sequence of ui/server.py _render_master
    measure     rudra.delivery.metadata.analyze_frame / maxcll_maxfall and
                ui/server.py measure

    python tools/emit_composite_golden.py            # writes native/tests/golden/composite/
    python tools/emit_composite_golden.py --checkpoint checkpoints/sdr2hdr_shadow_v1.pt

Deterministic on CPU. The files are small and committed; re-run and commit
when any of the Python above changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.anchor import anchor_to_sdr  # noqa: E402
from rudra.chroma import carry_source_chroma  # noqa: E402
from rudra.delivery import metadata as dm  # noqa: E402
from rudra.delivery.colorspace import convert, rgb_to_rgb_matrix  # noqa: E402
from rudra.delivery.controls import apply_region_ev  # noqa: E402
from training.infer_sdr2hdr import load_models, predict_fields, predict_image  # noqa: E402
from ui.server import DIFFUSE_WHITE_NITS, NETWORK_PEAK_NITS, measure  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "composite"
DEFAULT_CHECKPOINT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"

COMPOSITE_CASES = [
    # (recovery_mode, strength, preserve_outside)
    ("all", 1.0, True),
    ("all", 0.6, False),
    ("highlights", 1.3, True),
    ("shadows", 1.0, False),
    ("off", 1.0, True),
]

GRADED_BANDS = [
    {"label": "highlights", "low_nits": 400.0, "high_nits": 2000.0, "ev": 0.7},
    {"label": "speculars", "low_nits": 2000.0, "high_nits": 8000.0, "ev": -0.5},
    {"label": "shadows", "low_nits": 0.05, "high_nits": 12.0, "ev": 1.0},
]

MATRIX_PAIRS = [("rec709", "ap0"), ("rec2020", "ap0"), ("rec709", "rec2020"),
                ("p3d65", "rec2020"), ("ap0", "ap1"), ("rec2020", "rec709")]


def synthetic_frame(h: int = 48, w: int = 80) -> np.ndarray:
    """(H, W, 3) sRGB codes that exercise every stage: a grey ramp through the
    anchor knee, a warm ramp with a clipped disk, deep shadows and a saturated
    patch. Mild noise so no two pixels tie."""
    rng = np.random.default_rng(20260923)
    img = np.zeros((h, w, 3), np.float64)
    ramp = np.linspace(0.0, 1.0, w)
    img[: h // 3] = ramp[None, :, None]
    warm = np.stack([ramp, ramp * 0.8, ramp * 0.55], -1)
    img[h // 3: 2 * h // 3] = warm[None]
    yy, xx = np.mgrid[:h, :w]
    disk = (yy - h // 2) ** 2 + (xx - 3 * w // 4) ** 2 < (h // 6) ** 2
    img[disk] = 1.0
    img[2 * h // 3:] = (ramp * 0.15)[None, :, None]
    img[2 * h // 3:, : w // 5] = (1.0, 0.2, 0.1)
    img = img + rng.normal(0.0, 0.006, img.shape)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def chw(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1)))


class Writer:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        for old in OUT.glob("*.npy"):
            old.unlink()

    def f32(self, name: str, arr) -> dict:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
        np.save(OUT / f"{name}.npy", a, allow_pickle=False)
        return {"file": f"{name}.npy", "shape": list(a.shape)}

    def f64(self, name: str, arr) -> dict:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        np.save(OUT / f"{name}.npy", a, allow_pickle=False)
        return {"file": f"{name}.npy", "shape": list(a.shape)}


def stats_json(s: dm.FrameStats) -> dict:
    return {"min_nits": s.min_nits, "avg_nits": s.avg_nits, "max_nits": s.max_nits,
            "maxscl_nits": list(s.maxscl_nits), "percentiles_nits": list(s.percentiles_nits),
            "log_hist": [float(v) for v in s.log_hist]}


def frame_block(model, name: str, sdr_hwc: np.ndarray, out: Writer) -> tuple[dict, dict]:
    t = torch.from_numpy(chw(sdr_hwc))[None]
    with torch.inference_mode():
        fields = predict_fields(model, t, tile_size=0, overlap=64)
        sw = model.predict_shadow_weight(t) if hasattr(model, "predict_shadow_weight") else None
        probe = model(t, preserve_outside=True)
    block = {
        "sdr": out.f32(f"{name}_sdr", chw(sdr_hwc)),
        "residual": out.f32(f"{name}_residual", fields["residual"][0].numpy()),
        "highlight": out.f32(f"{name}_highlight", fields["highlight"][0].numpy()),
        "shadow": out.f32(f"{name}_shadow", fields["shadow"][0].numpy()),
        "baseline": out.f32(f"{name}_baseline", probe.baseline[0].float().numpy()),
        "shadow_weight": 1.0 if sw is None else float(sw.reshape(-1)[0]),
        "curve_params": [0.0] if fields["curve"] is None
        else [float(v) for v in fields["curve"].reshape(-1)],
        "composites": [],
    }
    network = {}
    for mode, strength, preserve in COMPOSITE_CASES:
        with torch.inference_mode():
            hdr = predict_image(model, t, preserve_outside=preserve, tile_size=0, overlap=64,
                                recovery_mode=mode, recovery_strength=strength)[0].float().numpy()
        tag = f"{name}_{mode}_s{int(round(strength * 100))}_{'p' if preserve else 'n'}"
        network[(mode, strength, preserve)] = hdr
        block["composites"].append({"mode": mode, "strength": strength, "preserve": preserve,
                                    "expected": out.f32(tag, hdr)})
    return block, network


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = ap.parse_args()

    torch.set_num_threads(1)
    model, _ = load_models(str(args.checkpoint), None, torch.device("cpu"))
    out = Writer()
    index: dict = {
        "checkpoint": args.checkpoint.name,
        "model": {"log_scale": float(model.log_scale), "max_hdr": float(model.max_hdr),
                  "corpus_ev": float(model.corpus_ev)},
        "network_peak_nits": NETWORK_PEAK_NITS, "diffuse_white_nits": DIFFUSE_WHITE_NITS,
        "frames": {}, "matrices": [],
    }

    frames = {"main": synthetic_frame(), "small": synthetic_frame(12, 16) * 0.5}
    for name, sdr_hwc in frames.items():
        block, network = frame_block(model, name, sdr_hwc, out)
        sdr64 = sdr_hwc.astype(np.float64)
        ceiling = float(model.max_hdr) * NETWORK_PEAK_NITS
        # The master chain exactly as ui/server.py _render_master runs it,
        # stage by stage, each stage's output kept so a failure names its stage.
        nits = np.transpose(network[("all", 1.0, True)], (1, 2, 0)).astype(np.float64) * NETWORK_PEAK_NITS
        stages = {}
        for softness in (1.0, 0.5):
            graded = np.clip(apply_region_ev(nits, GRADED_BANDS, softness), 0.0, ceiling)
            stages[f"region_soft{softness:g}"] = out.f64(
                f"{name}_region_soft{str(softness).replace('.', 'p')}", chw(graded))
        graded = np.clip(apply_region_ev(nits, GRADED_BANDS, 1.0), 0.0, ceiling)
        anchored = anchor_to_sdr(graded, sdr64, knee=0.9)
        carried = carry_source_chroma(anchored, sdr64, knee=0.99)
        scene_linear = (carried / DIFFUSE_WHITE_NITS).astype(np.float32)
        aces = convert(scene_linear, "rec709", "ap0")
        stages.update({
            "anchored": out.f64(f"{name}_anchored", chw(anchored)),
            "carried": out.f64(f"{name}_carried", chw(carried)),
            "scene_linear": out.f32(f"{name}_scene_linear", chw(scene_linear)),
            "aces_ap0": out.f32(f"{name}_aces_ap0", chw(aces)),
        })
        # Each stage alone on its predecessor's golden, too.
        stages["anchor_band_pixels"] = int(
            ((sdr64.max(-1) > 0.86) & (sdr64.max(-1) < 0.94)).sum())
        block["master"] = {"bands": GRADED_BANDS, "region_softness": 1.0,
                           "anchor_knee": 0.9, "chroma_knee": 0.99, "source_space": "rec709",
                           "stages": stages}

        stats = dm.analyze_frame(carried, index=0)
        maxcll, maxfall = dm.maxcll_maxfall([stats])
        block["analyze"] = {"input": "carried", "stats": stats_json(stats),
                            "maxcll": maxcll, "maxfall": maxfall}
        hdr0 = network[("all", 1.0, True)]
        base = np.load(OUT / block["baseline"]["file"])
        hl = np.load(OUT / block["highlight"]["file"])[0]
        sh = np.load(OUT / block["shadow"]["file"])[0]
        # measure() rounds for the page; the native port returns the raw
        # numbers and the test checks them against these to the rounding step.
        block["measure"] = {"input": "all_s100_p",
                            "result": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                       for k, v in measure(hdr0, base, hl, sh).items()}}
        index["frames"][name] = block

    for src, dst in MATRIX_PAIRS:
        index["matrices"].append({"src": src, "dst": dst,
                                  "m": [[float(v) for v in row] for row in rgb_to_rgb_matrix(src, dst)]})

    (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8", newline="\n")
    total = sum(f.stat().st_size for f in OUT.glob("*.npy"))
    print(f"wrote {len(list(OUT.glob('*.npy')))} arrays ({total / 1024:.0f} KB) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
