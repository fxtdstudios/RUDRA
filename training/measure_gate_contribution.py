"""Does the shadow gate earn its place?

The gate predicts ONE scalar per frame -- `shadow_weight` -- which scales the
shadow prior inside a max(). It costs a training stage, a checkpoint, a
registry entry and a section of the README. On 10 Sep 2026 three checkpoints
that differ only in that gate (`sdr2hdr_image_v5` has none, `shadow_v1` and
`shadow_s3` have one each) produced outputs on a 4.2 MP frame that differed by
6.9e-6 network units -- 0.07 nits. One frame proves nothing; this measures the
same thing across a set.

Two questions, cheapest first:

  1. IS THE GATE A CONSTANT? Collect its output over the frames. If the spread
     is small, the gate is an expensive way to store one number and the honest
     move is to store the number.

  2. DOES IT CHANGE THE PICTURE? Re-run each frame with the weight forced to
     the set's own mean and measure the difference against the gated output,
     in nits and in PU21-PSNR. If forcing the mean is invisible, question 1
     has already been answered in the affirmative and the bench only has to
     confirm parity.

This does NOT score against ground truth -- that is `score_checkpoint.ps1`,
which is the expensive half and should only be spent once this says the gate
is worth the row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FRAME_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
NETWORK_PEAK_NITS = 10_000.0
# Below this, a "spread" is not a measurement.
MIN_FRAMES = 8


def pu21_psnr(test_nits: np.ndarray, ref_nits: np.ndarray) -> float:
    """The repo's own PU21 metric, so numbers here compare with the bench.

    rudra.delivery.bench, not rudra.bench -- the latter is a stale path that
    imports a rudra.exr module which no longer exists.
    """
    from rudra.delivery.bench import pu_psnr
    return float(pu_psnr(test_nits, ref_nits))


def load_frames(source: Path, limit: int) -> list[Path]:
    if source.is_file() and source.suffix == ".jsonl":
        rows = [json.loads(line) for line in
                source.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [Path(r.get("sdr") or r.get("sdr_path") or r.get("path")) for r in rows]
    elif source.is_dir():
        paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in FRAME_SUFFIXES)
    else:
        raise SystemExit(f"not a manifest or a folder: {source}")
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise SystemExit(f"no frames found under {source}")
    return paths[:limit] if limit else paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True,
                    help="folder of SDR frames, or a manifest .jsonl")
    ap.add_argument("--gated", type=Path, required=True, help="a checkpoint WITH a shadow gate")
    ap.add_argument("--ungated", type=Path, help="a checkpoint without one, e.g. v5")
    ap.add_argument("--also", type=Path, nargs="*", default=[],
                    help="further gated checkpoints (other seeds) to cross-compare")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-side", type=int, default=1600,
                    help="downscale long side before inference; 0 keeps full res")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                    help=f"refuse to report a spread below this many frames (default {MIN_FRAMES})")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO / "ui"))
    import server as S
    from training.infer_sdr2hdr import predict_image

    frames = load_frames(args.frames, args.limit)
    print(f"   {len(frames)} frame(s)")
    if len(frames) < args.min_frames:
        # The standard deviation of one sample is zero, so a single frame
        # reports cv_pct 0.00 -- which reads as "the gate is a constant", the
        # exact conclusion this script tells the reader to act on. On
        # 10 Sep 2026 it pointed at a folder holding one image and produced
        # precisely that false pass. Unmeasured has to fail, never pass.
        raise SystemExit(
            f"only {len(frames)} frame(s): a spread needs at least {args.min_frames}. "
            f"Point --frames at the test split (bench/clean/sdr), or pass "
            f"--min-frames to say you meant it.")

    models = {"gated": S.load_model(args.gated, args.device)[0]}
    if args.ungated:
        models["ungated"] = S.load_model(args.ungated, args.device)[0]
    for i, extra in enumerate(args.also):
        models[f"also{i + 1}"] = S.load_model(extra, args.device)[0]
    if getattr(models["gated"], "shadow_gate", None) is None:
        raise SystemExit(f"{args.gated} has no shadow gate -- pass the gated one to --gated")

    def to_tensor(path: Path):
        im = Image.open(path).convert("RGB")
        if args.max_side and max(im.size) > args.max_side:
            r = args.max_side / max(im.size)
            im = im.resize((max(1, int(im.width * r)), max(1, int(im.height * r))), Image.LANCZOS)
        a = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(
            next(models["gated"].parameters()).device)

    # -- pass one: what does the gate say? -----------------------------------
    weights = []
    for path in frames:
        with torch.no_grad():
            gate_out = models["gated"].predict_shadow_weight(to_tensor(path))
        weights.append(float(gate_out.flatten()[0]))
    w = np.array(weights)
    mean = float(w.mean())
    print(f"\n   shadow_weight over {len(w)} frames:")
    print(f"     mean {mean:.4f}   std {w.std():.4f}   min {w.min():.4f}   max {w.max():.4f}"
          f"   range {w.max() - w.min():.4f}")
    print(f"     coefficient of variation {100 * w.std() / max(mean, 1e-9):.2f} %")

    # -- pass two: does forcing that mean change the picture? ---------------
    rows = []
    for path in frames:
        x = to_tensor(path)
        with torch.no_grad():
            gated = predict_image(models["gated"], x, preserve_outside=True, tile_size=0,
                                  overlap=64, recovery_mode="all", recovery_strength=1.0)
            forced = models["gated"](x, preserve_outside=True, recovery_mode="all",
                                     residual_strength=1.0, shadow_weight=mean).hdr
        g = gated[0].float().cpu().numpy().transpose(1, 2, 0) * NETWORK_PEAK_NITS
        f = forced[0].float().cpu().numpy().transpose(1, 2, 0) * NETWORK_PEAK_NITS
        row = {"frame": path.name,
               "gate_vs_constant_max_nits": float(np.abs(g - f).max()),
               "gate_vs_constant_p99_nits": float(np.percentile(np.abs(g - f), 99)),
               "gate_vs_constant_pu21_db": pu21_psnr(f, g)}
        for name, model in models.items():
            if name == "gated":
                continue
            with torch.no_grad():
                other = predict_image(model, x, preserve_outside=True, tile_size=0, overlap=64,
                                      recovery_mode="all", recovery_strength=1.0)
            o = other[0].float().cpu().numpy().transpose(1, 2, 0) * NETWORK_PEAK_NITS
            row[f"vs_{name}_max_nits"] = float(np.abs(g - o).max())
            row[f"vs_{name}_pu21_db"] = pu21_psnr(o, g)
        rows.append(row)
        print(f"     {path.name[:44]:<44} "
              f"constant {row['gate_vs_constant_max_nits']:>9.2f} nits max, "
              f"PU21 {row['gate_vs_constant_pu21_db']:>6.1f} dB")

    def agg(key):
        """Median and worst case. For nits, worst is the largest difference;
        for PU21-PSNR, higher means MORE alike, so worst is the smallest."""
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if not vals:
            return None
        worst = float(np.min(vals)) if key.endswith("_db") else float(np.max(vals))
        return {"median": float(np.median(vals)), "worst": worst}

    summary = {
        "frames": len(rows),
        "shadow_weight": {"mean": mean, "std": float(w.std()),
                          "min": float(w.min()), "max": float(w.max()),
                          "cv_pct": float(100 * w.std() / max(mean, 1e-9))},
        "gate_vs_constant": {"max_nits": agg("gate_vs_constant_max_nits"),
                             "pu21_db": agg("gate_vs_constant_pu21_db")},
    }
    for name in models:
        if name != "gated" and f"vs_{name}_max_nits" in rows[0]:
            summary[f"gated_vs_{name}"] = {"max_nits": agg(f"vs_{name}_max_nits"),
                                           "pu21_db": agg(f"vs_{name}_pu21_db")}
    print("\n" + json.dumps(summary, indent=2))
    print("\n   How to read this. PU21-PSNR above ~60 dB between two outputs means")
    print("   they are the same picture to any measurement that matters; the")
    print("   bench's own spread between real methods is single-digit dB. If the")
    print("   gate against its own constant lands there, the gate IS that")
    print("   constant, and score_checkpoint.ps1 only has to confirm parity.")

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2),
                            encoding="utf-8")
        print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
