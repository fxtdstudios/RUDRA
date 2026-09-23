"""Golden files for the native delivery modules (Phase 1 steps 3 to 6).

Principle P4: every file here is written by the Python the C++ ports.

    grade      rudra/delivery/controls.py   apply_grade, itm_strength_map
    hdr10      rudra/hdr10.py, delivery/profiles.py   PQ, master_to_peak, encode_master
    metadata   rudra/delivery/metadata.py   analyze_frame over a 12-frame, 3-shot
               sequence, detect_shots, and the three sidecars from write_all_sidecars
    exr        rudra/delivery/exr.py, aces.py   write_exr, write_aces_exr,
               write_acescg_exr, generate_ocio_config, float16 rounding

    python tools/emit_delivery_golden.py      # writes native/tests/golden/delivery/

The sidecars, EXRs and OCIO config are compared byte for byte by the native
tests. Deterministic.
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

from rudra.delivery import metadata as dm  # noqa: E402
from rudra.delivery.aces import generate_ocio_config, write_aces_exr, write_acescg_exr  # noqa: E402
from rudra.delivery.colorspace import REC2020_CHROMATICITIES  # noqa: E402
from rudra.delivery.controls import GradeControls, RegionEV, apply_grade, itm_strength_map, qualifier_mask  # noqa: E402
from rudra.delivery.exr import write_exr  # noqa: E402
from rudra.delivery.profiles import PROFILES, encode_master  # noqa: E402
from rudra.hdr10 import master_to_peak, master_to_pq, pq_eotf, pq_oetf  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "delivery"


class W:
    def __init__(self) -> None:
        if OUT.exists():
            shutil.rmtree(OUT)
        OUT.mkdir(parents=True)

    def f32(self, name, arr):
        np.save(OUT / f"{name}.npy", np.ascontiguousarray(np.asarray(arr, np.float32)), allow_pickle=False)
        return f"{name}.npy"

    def f64(self, name, arr):
        np.save(OUT / f"{name}.npy", np.ascontiguousarray(np.asarray(arr, np.float64)), allow_pickle=False)
        return f"{name}.npy"


def chw(a):
    return np.ascontiguousarray(np.transpose(a, (2, 0, 1)))


def nits_frame(rng, h, w, scale=1.0):
    """(h, w, 3) absolute nits spanning deep shadow to beyond 10 000."""
    base = np.exp(rng.normal(np.log(60.0), 2.2, (h, w, 1)))
    tint = rng.uniform(0.6, 1.3, (h, w, 3))
    return base * tint * scale


def grade_block(w: W, rng) -> dict:
    nits = nits_frame(rng, 24, 32)
    hi = qualifier_mask(nits, 400.0, 2000.0, 1.0)
    sh = qualifier_mask(nits, 0.05, 12.0, 0.5)
    block = {"input": w.f64("grade_in", chw(nits)), "masks": {"hi": w.f32("grade_mask_hi", hi[None]),
                                                              "sh": w.f32("grade_mask_sh", sh[None])},
             "cases": []}
    cases = [
        ("default", GradeControls()),
        ("exposure_peak4000_knee2000", GradeControls(exposure_ev=0.5, peak_nits=4000.0, knee_nits=2000.0)),
        ("regions_desat", GradeControls(peak_nits=1000.0, highlight_desat=0.5,
                                        regions=[RegionEV(hi, 0.7, "hi"), RegionEV(sh, -0.4, "sh")])),
        ("desat_full_peak600", GradeControls(peak_nits=600.0, highlight_desat=1.0)),
    ]
    for name, g in cases:
        block["cases"].append({
            "name": name, "exposure_ev": g.exposure_ev, "peak_nits": g.peak_nits, "knee_nits": g.knee_nits,
            "highlight_desat": g.highlight_desat,
            "regions": [{"mask": r.label, "ev": r.ev} for r in g.regions],
            "expected": w.f32(f"grade_{name}", chw(apply_grade(nits, g))),
        })
    itm = itm_strength_map((24, 32), 0.8, [RegionEV(hi, 0.7, "hi"), RegionEV(sh, -1.5, "sh")])
    block["itm"] = {"base": 0.8, "regions": [{"mask": "hi", "ev": 0.7}, {"mask": "sh", "ev": -1.5}],
                    "expected": w.f32("grade_itm", itm[0])}
    return block


def hdr10_block(w: W, rng) -> dict:
    nits = np.concatenate([[0.0, -5.0, 1e-4, 203.0, 1000.0, 10000.0, 25000.0], np.logspace(-4, 4.3, 400)])
    codes = np.concatenate([[0.0, 1.0, 0.5, -0.1, 1.2], np.linspace(0, 1, 300)])
    norm = (nits_frame(rng, 16, 24) / 10000.0).astype(np.float32)
    block = {
        "pq_in": w.f64("pq_in", nits), "pq_oetf": w.f32("pq_oetf", pq_oetf(nits)),
        "eotf_in": w.f64("eotf_in", codes), "pq_eotf": w.f32("pq_eotf", pq_eotf(codes)),
        "pq12": [dm._pq12(float(v)) for v in nits],
        "norm": w.f32("norm_in", chw(norm)),
        "master": [], "encode": [],
    }
    for peak, knee in ((1000.0, None), (4000.0, 1000.0), (600.0, 0.0)):
        tag = f"p{int(peak)}_k{'none' if knee is None else int(knee)}"
        pq, mastered = master_to_pq(norm, peak, knee)
        block["master"].append({"peak": peak, "knee": knee,
                                "mastered": w.f32(f"master_{tag}", chw(master_to_peak(norm, peak, knee))),
                                "pq": w.f32(f"masterpq_{tag}", chw(pq)),
                                "pq_mastered": w.f32(f"masterpq_lin_{tag}", chw(mastered))})
    for profile in list(PROFILES):
        for peak in ((1000.0, 2000.0) if profile == "hlg" else (1000.0,)):
            code, disp = encode_master(norm, profile, peak)
            tag = f"{profile}_p{int(peak)}"
            block["encode"].append({"profile": profile, "peak": peak,
                                    "code": w.f32(f"encode_{tag}_code", chw(code)),
                                    "display": w.f32(f"encode_{tag}_display", chw(disp))})
    block["profiles"] = PROFILES
    return block


def metadata_block(w: W, rng) -> dict:
    frames = []
    a = nits_frame(rng, 16, 24)
    for i in range(5):
        frames.append(a * (1.0 + 0.02 * i))
    b = nits_frame(rng, 16, 24, scale=40.0)
    for i in range(4):
        frames.append(b * (1.0 - 0.03 * i))
    c = nits_frame(rng, 16, 24, scale=0.01)
    for i in range(3):
        frames.append(c * (1.0 + 0.05 * i))
    frames[7][2, 3] = [np.nan, np.inf, -1.0]            # analyze_frame's sanitising
    stats = [dm.analyze_frame(f, index=i) for i, f in enumerate(frames)]
    shots = dm.detect_shots(stats)
    paths = dm.write_all_sidecars(stats, OUT / "seq", mastering_peak_nits=1000.0)
    return {"frames": w.f64("meta_frames", np.stack([chw(f) for f in frames])),
            "shots": [list(s) for s in shots],
            "sidecars": {k: p.name for k, p in paths.items()}}


def exr_block(w: W, rng) -> dict:
    img = rng.uniform(-0.5, 3.0, (12, 20, 3)).astype(np.float32)
    img[0, 0] = [np.nan, np.inf, -np.inf]
    img[0, 1] = [70000.0, -70000.0, 65519.0]
    img[0, 2] = [1e-6, 3e-8, 6.1e-5]
    img[0, 3] = [2.0 ** -25, 2.0 ** -24 * 1.5, 65504.0]
    rgba = np.dstack([img, rng.uniform(0, 1, (12, 20, 1)).astype(np.float32)])
    files = {}
    write_exr(OUT / "exr_half.exr", img, half=True)
    files["half"] = "exr_half.exr"
    write_exr(OUT / "exr_float_chroma.exr", img, half=False, chromaticities=REC2020_CHROMATICITIES,
              attributes={"rudra:checkpoint": "sdr2hdr_shadow_v1.pt", "rudra:note": "tab\there \"quoted\""})
    files["float_chroma"] = "exr_float_chroma.exr"
    write_exr(OUT / "exr_rgba.exr", rgba, half=True)
    files["rgba"] = "exr_rgba.exr"
    provenance = {"rudra:maxCLL": "2152", "rudra:checkpoint": "sdr2hdr_shadow_v1.pt", "rudra:anchored": "True"}
    write_aces_exr(np.abs(img), OUT / "exr_aces.exr", source_space="rec709", provenance=provenance)
    files["aces"] = "exr_aces.exr"
    write_acescg_exr(np.abs(img), OUT / "exr_acescg.exr", source_space="rec2020")
    files["acescg"] = "exr_acescg.exr"
    generate_ocio_config(OUT / "rudra_config.ocio")
    # float16 rounding, across the range and at every edge case.
    vals = np.concatenate([
        rng.uniform(-65504, 65504, 300), np.exp(rng.uniform(np.log(1e-9), np.log(7e4), 700)),
        [0.0, -0.0, 65504.0, 65519.9, 65520.0, 2.0 ** -24, 2.0 ** -25, 2.0 ** -25 * 1.0001, 2.0 ** -14,
         2.0 ** -14 * (1 - 2 ** -11), 1.0 + 2 ** -11, 1.0 + 3 * 2 ** -11]]).astype(np.float32)
    bits = vals.astype(np.float16).view(np.uint16).astype(np.float32)
    return {"input": w.f32("exr_in", chw(img)), "input_rgba": w.f32("exr_in_rgba", chw(rgba)),
            "provenance": provenance, "files": files, "ocio": "rudra_config.ocio",
            "half_in": w.f32("half_in", vals), "half_bits": w.f32("half_bits", bits)}


def main() -> int:
    w = W()
    rng = np.random.default_rng(20260923)
    index = {"grade": grade_block(w, rng), "hdr10": hdr10_block(w, rng),
             "metadata": metadata_block(w, rng), "exr": exr_block(w, rng)}
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"wrote {len(list(OUT.iterdir()))} files ({total / 1024:.0f} KB) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
