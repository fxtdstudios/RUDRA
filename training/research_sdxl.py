#!/usr/bin/env python3
"""
research_sdxl.py — the RUDRA research program on SDXL, end to end.

SDXL is the paper's intended backbone for RUDRA-Full (configs/rudra_full_train.yaml,
and §3.5 describes U-Net cross-attention injection, which SDXL actually has).
It is also ~5x lighter than Flux, so the whole program fits a 16 GB card.

Phases (run all, or pick with --phase):
  prereq   — ensure SDXL weights + pairs exist (download / report)
  stage2   — DR-gated LoRA (RUDRA-Lite conditioning), 10k steps
  stage3   — DRE + cross-attention injection (THE CORE THESIS), 10k steps
  sweep    — §7.3 λ conditioning-strength sweep (fixed λ: 0.0 .. 1.25), 6 short runs
  ablate   — §7.2 descriptor channel ablation (L / L,E / L,H / L,x,y), 4 short runs
  report   — list every produced checkpoint + the evaluation next-steps

Usage
-----
  python training/research_sdxl.py --phase prereq --dry-run
  python training/research_sdxl.py --phase stage3            # just the core
  python training/research_sdxl.py                            # everything, in order
  python training/research_sdxl.py --phase sweep --sweep-steps 4000
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "training" / "train_rudra.py"
COMFY_ROOT = Path(os.environ.get("COMFY_ROOT", r"D:\A.I\ComfyUI"))
CKPT = REPO / "hdrdata" / "checkpoints"

LAMBDAS = [0.0, 0.25, 0.5, 0.8, 1.0, 1.25]                     # paper §7.3
CHANNEL_SETS = ["L", "L,E", "L,H", "L,x,y"]                     # paper §7.2 (full R5 = stage3)


def env():
    e = dict(os.environ)
    extra = f"{COMFY_ROOT / 'custom_nodes'};{COMFY_ROOT / 'custom_nodes' / 'radiance'}"
    e["PYTHONPATH"] = extra + (os.pathsep + e["PYTHONPATH"] if e.get("PYTHONPATH") else "")
    e.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    return e


def find_sdxl() -> str:
    p = os.environ.get("SDXL_CKPT", "")
    if p and os.path.exists(p):
        return p
    for pat in ("models/checkpoints/sd_xl_base*.safetensors",
                "models/checkpoints/*sdxl*.safetensors",
                "models/diffusion_models/sd_xl*.safetensors"):
        hits = sorted(glob.glob(str(COMFY_ROOT / pat)))
        if hits:
            return hits[0]
    return ""


def run(cmd: list[str] | None, dry: bool) -> bool:
    if cmd is None:        # base_cmd returned None = phase already complete
        return True
    print("  " + " ".join(cmd))
    if dry:
        return True
    return subprocess.run(cmd, env=env()).returncode == 0


_STAGE_SUB = {"lora": "rudra_stage2", "dre": "rudra_stage3"}


def latest_ckpt(stage: str, out: str) -> tuple[str | None, int]:
    """Find the newest stageN checkpoint + its step for resuming."""
    sub = CKPT / out / f"sdxl_{_STAGE_SUB[stage]}"
    cks = sorted(glob.glob(str(sub / f"{_STAGE_SUB[stage]}_step*.pth")),
                 key=lambda p: int(re.search(r"step0*(\d+)", p).group(1)) if re.search(r"step0*(\d+)", p) else -1)
    if not cks:
        return None, 0
    step = int(re.search(r"step0*(\d+)", os.path.basename(cks[-1])).group(1))
    return cks[-1], step


def base_cmd(stage: str, model_path: str, pairs: str, out: str, steps: int, args) -> list[str] | None:
    cmd = [sys.executable, str(TRAIN), "--stage", stage, "--model_type", "sdxl",
           "--model_path", model_path, "--pair_dir", str(REPO / pairs),
           "--output_dir", str(CKPT / out), "--steps", str(steps),
           "--batch_size", str(args.batch_size), "--lr", str(args.lr)]
    if args.resume and stage in _STAGE_SUB:
        ck, st = latest_ckpt(stage, out)
        if ck and st >= steps:
            print(f"  [skip] {out} already at step {st}/{steps}")
            return None
        if ck:
            cmd += ["--resume", ck]
            print(f"  [resume] {out} from step {st}")
    return cmd


def main():
    ap = argparse.ArgumentParser(description="RUDRA research program on SDXL.")
    ap.add_argument("--phase", default="all",
                    choices=["all", "prereq", "stage2", "stage3", "sweep", "ablate", "report"])
    ap.add_argument("--pairs", default="hdrdata/sdxl_pairs",
                    help="Pair set (use a 256px set from make_pairs_256.py for ~2-4x speed).")
    ap.add_argument("--steps", type=int, default=10000, help="Steps for stage2/stage3.")
    ap.add_argument("--sweep-steps", type=int, default=6000, help="Steps per sweep/ablation run.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resume", action="store_true",
                    help="Continue stage2/stage3 from their latest checkpoint; skip if already at target steps.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    phases = [args.phase] if args.phase != "all" else ["prereq", "stage2", "stage3", "sweep", "ablate", "report"]
    failures = []

    # ── prereq ────────────────────────────────────────────────────────────────
    model_path = find_sdxl()
    if "prereq" in phases or not model_path:
        print("=== prereq ===")
        if model_path:
            print(f"  SDXL weights: {model_path}")
        else:
            print("  SDXL weights MISSING. Download with:")
            print("    python training/download_models.py --only sd_xl_base_1.0.safetensors")
            if not args.dry_run:
                return 1
        pair_dir = REPO / args.pairs
        n = len(list(pair_dir.glob("*.npz"))) if pair_dir.is_dir() else 0
        print(f"  pairs: {args.pairs} -> {n} pairs" + ("" if n else "  (MISSING — run build_all_decoders --only sdxl or make_pairs_256.py)"))
        if n == 0 and not args.dry_run:
            return 1
        if n and n < 3000:
            print(f"  note: only {n} pairs — consider make_pairs_256.py --model-type sdxl --crops 4 for more data + speed")

    # ── stage2: DR-gated LoRA ─────────────────────────────────────────────────
    if "stage2" in phases:
        print("\n=== stage2 — DR-gated LoRA (RUDRA-Lite) ===")
        if not run(base_cmd("lora", model_path, args.pairs, "sdxl_lora", args.steps, args), args.dry_run):
            failures.append("stage2")

    # ── stage3: DRE + cross-attention (the core) ──────────────────────────────
    if "stage3" in phases:
        print("\n=== stage3 — DRE + cross-attention injection (core thesis) ===")
        cmd = base_cmd("dre", model_path, args.pairs, "sdxl_dre", args.steps, args)
        if cmd is not None:
            cmd += ["--lambda_init", "0.8"]      # learnable λ, paper operating point
        if not run(cmd, args.dry_run):
            failures.append("stage3")

    # ── §7.3 λ sweep ──────────────────────────────────────────────────────────
    if "sweep" in phases:
        print("\n=== λ conditioning-strength sweep (§7.3, fixed λ) ===")
        for lam in LAMBDAS:
            tag = f"sdxl_dre_lam{str(lam).replace('.', 'p')}"
            cmd = base_cmd("dre", model_path, args.pairs, tag, args.sweep_steps, args)
            if cmd is not None:
                cmd += ["--lambda_init", str(lam), "--fixed_lambda"]
            if not run(cmd, args.dry_run):
                failures.append(tag)

    # ── §7.2 descriptor channel ablation ──────────────────────────────────────
    if "ablate" in phases:
        print("\n=== descriptor channel ablation (§7.2) ===")
        for chans in CHANNEL_SETS:
            tag = "sdxl_dre_ch_" + chans.replace(",", "")
            cmd = base_cmd("dre", model_path, args.pairs, tag, args.sweep_steps, args)
            if cmd is not None:
                cmd += ["--desc_channels", chans]
            if not run(cmd, args.dry_run):
                failures.append(tag)

    # ── report ────────────────────────────────────────────────────────────────
    if "report" in phases:
        print("\n=== produced checkpoints ===")
        for d in sorted(CKPT.glob("sdxl_*")):
            steps = sorted(d.rglob("*step*.pth"))
            print(f"  {d.name:<24} {'✓ ' + steps[-1].name if steps else '—'}")
        print("\nNext (evaluation):")
        print("  1. Generate HDR samples with vs. without DRE tokens (same seeds/prompts).")
        print("  2. Score with training/benchmark_hdr.py (ColorVideoVDP JOD + ΔE + EV + HRA).")
        print("  3. Fill the paper's §6/§7 tables from the sweep + ablation runs.")

    if failures:
        print("\nFailed:", ", ".join(failures)); return 2
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
