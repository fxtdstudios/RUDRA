"""Benchmark harness: RUDRA vs the field, in publishable units (torch-free core).

No SDR->HDR competitor publishes quality numbers; the only public measuring
stick is the academic SOTA's CVVDP / PU-PSNR on the Stuttgart HDR and UBC
sets. This harness produces exactly those numbers from paired directories of
reference and test frames, so every RUDRA claim ships with a reproducible
command line instead of adjectives.

Metrics:
  - PU21-PSNR (banding+glare variant, Mantiuk et al. 2021): pure numpy,
    runs anywhere. Valid input range 0.005..10000 cd/m² (clamped).
  - CVVDP (JOD): used when torch + pycvvdp are installed, via
    ``rudra.hdrvdp.hdr_vdp3_jod``; otherwise recorded as unavailable —
    never silently substituted.

Layout convention (Stuttgart / UBC or any paired set):
    <root>/ref/<clip>/<frame>.exr      ground-truth HDR
    <root>/test/<clip>/<frame>.exr     method output, same stems
.npy frames (H, W, 3 linear nits) are accepted alongside .exr.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .exr import read_exr

__all__ = ["pu21_encode", "pu_psnr", "run_benchmark", "load_frame"]

# PU21 banding+glare coefficients (Mantiuk et al., 2021 reference implementation).
_PU21 = (0.353487901, 0.3734658629, 8.277049286e-05,
         0.9062562627, 0.09150303166, 0.9099517204, 596.3148142)
_PU21_MIN, _PU21_MAX = 0.005, 10000.0


def pu21_encode(luminance_nits: np.ndarray) -> np.ndarray:
    """Perceptually uniform encoding of absolute luminance (banding+glare)."""
    p0, p1, p2, p3, p4, p5, p6 = _PU21
    y = np.clip(np.asarray(luminance_nits, dtype=np.float64), _PU21_MIN, _PU21_MAX)
    yp = np.power(y, p3)
    return p6 * (np.power((p0 + p1 * yp) / (1.0 + p2 * yp), p4) - p5)


def pu_psnr(test_nits: np.ndarray, ref_nits: np.ndarray) -> float:
    """PU21-PSNR in dB between linear frames in absolute nits (per channel)."""
    test = pu21_encode(test_nits)
    ref = pu21_encode(ref_nits)
    peak = float(pu21_encode(np.array(_PU21_MAX)))
    mse = float(np.mean((test - ref) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def load_frame(path: Path, nits_scale: float = 1.0) -> np.ndarray:
    """Load a linear frame (.exr or .npy) as (H, W, 3) float64 nits."""
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".exr":
        arr, _ = read_exr(path)
    else:
        raise ValueError(f"unsupported frame format: {path}")
    arr = np.asarray(arr, dtype=np.float64)[..., :3]
    return np.maximum(arr * float(nits_scale), 0.0)


@dataclass
class PairResult:
    clip: str
    frame: str
    pu_psnr_db: float
    cvvdp_jod: float | None


def _cvvdp_fn():
    """Return a jod(test, ref) callable if torch + pycvvdp exist, else None."""
    try:
        import torch  # noqa: F401
        from ..hdrvdp import colorvideovdp_available, hdr_vdp3_jod
        if not colorvideovdp_available():
            return None

        def jod(test: np.ndarray, ref: np.ndarray) -> float:
            import torch as _t
            to = lambda a: _t.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).permute(2, 0, 1)[None]
            value, backend = hdr_vdp3_jod(to(test), to(ref), color_space="rec2020",
                                          diffuse_white_nits=1.0)  # inputs already absolute nits
            return value if backend == "colorvideovdp" else None
        return jod
    except Exception:
        return None


def run_benchmark(
    root: str | Path,
    nits_scale: float = 1.0,
    output: str | Path | None = None,
    limit: int | None = None,
) -> dict:
    """Score every ref/test pair under ``root``; write results JSON + CSV.

    ``nits_scale`` converts stored values to absolute nits (e.g. 203.0 for
    diffuse-white-relative frames where 1.0 = 203 cd/m²).
    """
    root = Path(root)
    ref_root, test_root = root / "ref", root / "test"
    if not ref_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"expected {ref_root} and {test_root}")

    pairs: list[tuple[str, str, Path, Path]] = []
    for ref_path in sorted(ref_root.rglob("*")):
        if ref_path.suffix.lower() not in (".exr", ".npy"):
            continue
        rel = ref_path.relative_to(ref_root)
        for suffix in (ref_path.suffix, ".exr", ".npy"):
            candidate = (test_root / rel).with_suffix(suffix)
            if candidate.exists():
                pairs.append((str(rel.parent), rel.stem, ref_path, candidate))
                break
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise FileNotFoundError(f"no ref/test pairs found under {root}")

    cvvdp = _cvvdp_fn()
    results: list[PairResult] = []
    for clip, frame, ref_path, test_path in pairs:
        ref = load_frame(ref_path, nits_scale)
        test = load_frame(test_path, nits_scale)
        if ref.shape != test.shape:
            raise ValueError(f"{clip}/{frame}: shape mismatch {test.shape} vs {ref.shape}")
        results.append(PairResult(
            clip=clip, frame=frame,
            pu_psnr_db=pu_psnr(test, ref),
            cvvdp_jod=(cvvdp(test, ref) if cvvdp else None),
        ))

    finite = [r.pu_psnr_db for r in results if np.isfinite(r.pu_psnr_db)]
    jods = [r.cvvdp_jod for r in results if r.cvvdp_jod is not None]
    per_clip: dict[str, list[float]] = {}
    for r in results:
        per_clip.setdefault(r.clip, []).append(r.pu_psnr_db)
    summary = {
        "pairs": len(results),
        "pu_psnr_db_mean": float(np.mean(finite)) if finite else None,
        "pu_psnr_db_per_clip": {c: float(np.mean(v)) for c, v in per_clip.items()},
        "cvvdp_jod_mean": float(np.mean(jods)) if jods else None,
        "cvvdp_backend": "colorvideovdp" if jods else "unavailable (install torch + pycvvdp)",
        "nits_scale": nits_scale,
        "results": [r.__dict__ for r in results],
    }
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        csv = out.with_suffix(".csv")
        rows = ["clip,frame,pu_psnr_db,cvvdp_jod"]
        rows += [f"{r.clip},{r.frame},{r.pu_psnr_db:.4f},{'' if r.cvvdp_jod is None else f'{r.cvvdp_jod:.4f}'}"
                 for r in results]
        csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return summary
