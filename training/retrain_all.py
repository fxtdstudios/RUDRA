#!/usr/bin/env python3
"""
retrain_all.py — one-shot RUDRA retraining orchestrator (post-fix).

Cleans (archives) the old broken checkpoints, runs a fast sanity gate, then
retrains every backbone/stage in sequence by shelling out to train_rudra.py.
Supports per-run resume from the latest checkpoint.

Examples
--------
  # Dry run: print everything it WOULD do, touch nothing
  python training/retrain_all.py --dry-run

  # Archive old checkpoints, run sanity gate, train all decoders + LoRA + DRE
  python training/retrain_all.py --clean archive

  # Resume an interrupted rebuild (no cleaning; picks up latest ckpt per run)
  python training/retrain_all.py --resume

  # Only the Flux runs; skip Stage 2/3 backbone stages
  python training/retrain_all.py --only flux1_turbo,flux1_full --no-backbone-stages

Notes
-----
* Decoder stages need only --pair_dir. Stage 2 (lora) / Stage 3 (dre) need the
  backbone weights — set them in BACKBONES below or via env (FLUX_CKPT, WAN_CKPT).
* --clean and --resume are mutually exclusive (resume needs the old files).
* Runs are sequential and stop on first failure unless --keep-going.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "training" / "train_rudra.py"
CKPT_ROOT = REPO / "hdrdata" / "checkpoints"
COLOR_SPACE = "rec2020"          # single working space for the whole rebuild

# Where ComfyUI keeps diffusion backbones; used to auto-discover Stage 2/3 weights.
COMFY_DIFFUSION_DIR = os.environ.get(
    "COMFY_DIFFUSION_DIR", r"D:\A.I\ComfyUI\models\diffusion_models")


# model_type -> canonical filename in RADIANCE_MODEL_MAP (for auto-download).
_REGISTRY_FILE = {
    "flux": "flux1-dev-fp8.safetensors",
    "ltx-video": "ltx-2.3-22b-dev-fp8.safetensors",
    "sdxl": "sd_xl_base_1.0.safetensors",
}
_ENV_VAR = {"flux": "FLUX_CKPT", "wan": "WAN_CKPT", "ltx-video": "LTX_CKPT", "sdxl": "SDXL_CKPT"}

# Specific, prioritized filename globs per backbone. The "flux" config targets
# flux1-dev (16-ch latents, T5/4096) — it must NOT match flux-2/Klein, which is a
# different architecture (32/128-ch VAE, Qwen3/7680).
_BACKBONE_PATTERNS = {
    "flux": ["*flux1-dev*.safetensors", "*flux1*dev*.safetensors", "*flux1*.safetensors"],
    "wan": ["*wan*.safetensors"],
    "ltx-video": ["*ltx*.safetensors"],
    "sdxl": ["*sd_xl*.safetensors", "*sdxl*.safetensors"],
}


def _find_backbone(env_var: str, *patterns: str) -> str:
    """Resolve a backbone checkpoint: explicit env var first, else glob ComfyUI dir."""
    import glob
    p = os.environ.get(env_var, "")
    if p and os.path.exists(p):
        return p
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(COMFY_DIFFUSION_DIR, pat)))
        if hits:
            return hits[0]
    return ""


def resolve_backbone(model_type: str, auto_download: bool = True) -> str:
    """Find the backbone weights for a model_type; download from the registry if
    missing and auto_download is on. Returns "" if it can't be resolved."""
    env = _ENV_VAR.get(model_type, "")
    patterns = _BACKBONE_PATTERNS.get(model_type, [f"*{model_type}*.safetensors"])
    local = _find_backbone(env, *patterns)
    if local and os.path.exists(local):
        return local

    fname = _REGISTRY_FILE.get(model_type)
    if not fname:
        print(f"   [skip] no registered backbone URL for model_type '{model_type}'"
              f" (set {env or 'the *_CKPT env var'} manually).")
        return ""
    dest = os.path.join(COMFY_DIFFUSION_DIR, fname)
    if os.path.exists(dest):
        return dest
    if not auto_download:
        print(f"   [skip] {fname} missing and --no-download set.")
        return ""
    try:
        from download_models import _load_model_map, _download
        url = _load_model_map()[fname]["url"]
        print(f"   backbone '{fname}' missing — downloading to {dest} ...")
        _download(url, Path(dest))
        return dest if os.path.exists(dest) else ""
    except Exception as e:
        print(f"   [skip] backbone auto-download failed: {e}")
        return ""


# ── Run table ────────────────────────────────────────────────────────────────
# Each run: name (checkpoint subdir), stage, model_type, and stage-specific args.
def _decoder(name, model_type, size, pair_dir, steps, bs, lr):
    return dict(name=name, stage="decoder", model_type=model_type, size=size,
                pair_dir=pair_dir, steps=steps, batch_size=bs, lr=lr)

def _backbone(name, stage, model_type, pair_dir, steps, bs, lr):
    return dict(name=name, stage=stage, model_type=model_type,
                pair_dir=pair_dir, steps=steps, batch_size=bs, lr=lr, needs_backbone=True)

RUNS = [
    # ── Decoders (turbo 50k, full 100k) ──────────────────────────────────────
    _decoder("flux1_turbo", "flux", "turbo", "hdrdata/hdr_pairs",     50_000, 8, 3e-4),
    _decoder("flux2_turbo", "flux", "turbo", "hdrdata/hdr_pairs",     50_000, 8, 3e-4),
    _decoder("wan21_turbo", "wan",  "turbo", "hdrdata/wan_hdr_pairs", 50_000, 8, 3e-4),
    _decoder("wan22_turbo", "wan",  "turbo", "hdrdata/wan_hdr_pairs", 50_000, 8, 3e-4),
    _decoder("ltx_turbo",   "ltx-video", "turbo", "hdrdata/ltx_pairs", 50_000, 4, 3e-4),
    # Full decoders are heavy; on a 16 GB card use batch 1 and fewer steps so a
    # run finishes in a day, not a week. (flux2/wan22 are duplicates of flux1/wan21.)
    _decoder("flux1_full",  "flux", "full",  "hdrdata/hdr_pairs",     40_000, 1, 1e-4),
    _decoder("wan21_full",  "wan",  "full",  "hdrdata/wan_hdr_pairs", 40_000, 1, 1e-4),
    # ── Backbone stages (need BACKBONES weights) ─────────────────────────────
    # batch_size 1 — Flux LoRA backprops through the full 12B backbone; 16 GB
    # cards can't fit batch 2 at high latent resolution.
    _backbone("flux1_lora", "lora", "flux", "hdrdata/hdr_pairs", 10_000, 1, 1e-4),
    _backbone("flux1_dre",  "dre",  "flux", "hdrdata/hdr_pairs", 10_000, 1, 1e-4),
]

STAGE_DIR = {"decoder": "rudra_stage1", "lora": "rudra_stage2", "dre": "rudra_stage3"}


# ── Helpers ──────────────────────────────────────────────────────────────────
def run_output_dir(run) -> Path:
    return CKPT_ROOT / run["name"]

def stage_subdir(run) -> Path:
    # train_rudra.py appends f"{model_type}_{STAGE_DIR[stage]}" to --output_dir.
    return run_output_dir(run) / f"{run['model_type']}_{STAGE_DIR[run['stage']]}"

def latest_checkpoint(run) -> str | None:
    d = stage_subdir(run)
    if not d.is_dir():
        return None
    ckpts = sorted(d.glob("*step*.pth"),
                   key=lambda p: int(re.search(r"step0*(\d+)", p.name).group(1))
                   if re.search(r"step0*(\d+)", p.name) else -1)
    return str(ckpts[-1]) if ckpts else None

def build_command(run, resume: bool, auto_download: bool = True) -> list[str] | None:
    cmd = [sys.executable, str(TRAIN),
           "--stage", run["stage"],
           "--model_type", run["model_type"],
           "--pair_dir", str(REPO / run["pair_dir"]),
           "--output_dir", str(run_output_dir(run)),
           "--steps", str(run["steps"]),
           "--batch_size", str(run["batch_size"]),
           "--lr", str(run["lr"]),
           "--color_space", COLOR_SPACE]
    if run["stage"] == "decoder":
        cmd += ["--model_size", run["size"], "--multi_curve",
                "--val_split", "0.05", "--patience", "8"]
    if run.get("needs_backbone"):
        ckpt = resolve_backbone(run["model_type"], auto_download=auto_download)
        if not ckpt or not os.path.exists(ckpt):
            print(f"  [skip] {run['name']}: backbone for '{run['model_type']}' unavailable.")
            return None
        cmd += ["--model_path", ckpt]
    if resume:
        last = latest_checkpoint(run)
        if last:
            cmd += ["--resume", last]
            print(f"  [resume] {run['name']} from {Path(last).name}")
    return cmd


def clean_checkpoints(mode: str, dry: bool):
    if mode == "none":
        return
    if not CKPT_ROOT.exists():
        return
    existing = [p for p in CKPT_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_archive")]
    if not existing:
        return
    if mode == "archive":
        dest = CKPT_ROOT / f"_archive_{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"[clean] archiving {len(existing)} run dir(s) -> {dest}")
        if not dry:
            dest.mkdir(parents=True, exist_ok=True)
            for p in existing:
                shutil.move(str(p), str(dest / p.name))
    elif mode == "delete":
        print(f"[clean] DELETING {len(existing)} run dir(s) in {CKPT_ROOT}")
        if not dry:
            for p in existing:
                shutil.rmtree(p, ignore_errors=True)


def sanity_gate(dry: bool) -> bool:
    """Run a 300-step Flux turbo decoder and verify the §3.1/§3.2 fixes took."""
    out = CKPT_ROOT / "_sanity"
    cmd = [sys.executable, str(TRAIN), "--stage", "decoder", "--model_type", "flux",
           "--model_size", "turbo", "--pair_dir", str(REPO / "hdrdata/hdr_pairs"),
           "--output_dir", str(out), "--steps", "300", "--batch_size", "1",
           "--multi_curve", "--color_space", COLOR_SPACE]
    print("[sanity] " + " ".join(cmd))
    if dry:
        return True
    if subprocess.run(cmd).returncode != 0:
        print("[sanity] training process failed.")
        return False
    log = out / "flux_rudra_stage1" / "rudra_decoder_log.jsonl"
    if not log.exists():
        print("[sanity] no log produced."); return False
    import json, math
    lines = [json.loads(l) for l in log.read_text().splitlines() if '"eval"' not in l]
    if not lines:
        print("[sanity] no train lines."); return False
    first_loss = lines[0].get("loss", float("nan"))
    last_loss = lines[-1].get("loss", float("nan"))
    # The meaningful early signal: the loss is finite (the NaN bug is fixed) and
    # trending down. highlight/exposure are weighted 0 before step 10k and the
    # highlight mask is strict, so they are NOT expected to be non-zero here.
    finite = math.isfinite(first_loss) and math.isfinite(last_loss)
    decreasing = last_loss < first_loss
    print(f"[sanity] loss {first_loss:.4f} -> {last_loss:.4f}  (finite={finite}, decreasing={decreasing})")
    ok = finite and decreasing
    print("[sanity] PASS — loss is finite and falling; fixes are live." if ok else
          "[sanity] FAIL — loss is non-finite or not decreasing. Do NOT launch full runs.")
    shutil.rmtree(out, ignore_errors=True)
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Retrain all RUDRA models with resume + cleanup.")
    ap.add_argument("--clean", choices=["none", "archive", "delete"], default="archive",
                    help="What to do with existing checkpoints before training (default: archive).")
    ap.add_argument("--resume", action="store_true", help="Resume each run from its latest checkpoint.")
    ap.add_argument("--only", default="", help="Comma-separated run names to include (default: all).")
    ap.add_argument("--no-backbone-stages", action="store_true", help="Skip Stage 2/3 (lora/dre).")
    ap.add_argument("--skip-sanity", action="store_true", help="Skip the 300-step sanity gate.")
    ap.add_argument("--keep-going", action="store_true", help="Continue to next run if one fails.")
    ap.add_argument("--no-download", action="store_true",
                    help="Do NOT auto-download missing backbone weights for Stage 2/3.")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without executing.")
    args = ap.parse_args()

    # Reduce CUDA fragmentation OOM across the long sequential runs; children inherit this.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Avoid the Windows/conda OpenMP duplicate-runtime crash (libiomp5md.dll loaded
    # twice) that triggers when Stage 2/3 load the full diffusion backbone.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if args.resume and args.clean != "none":
        print("[info] --resume set; forcing --clean none (resume needs existing checkpoints).")
        args.clean = "none"

    runs = RUNS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        runs = [r for r in runs if r["name"] in wanted]
    if args.no_backbone_stages:
        runs = [r for r in runs if not r.get("needs_backbone")]
    if not runs:
        print("No runs selected."); return 1

    clean_checkpoints(args.clean, args.dry_run)

    # The sanity gate trains a decoder; only meaningful if a decoder run is queued.
    has_decoder = any(r["stage"] == "decoder" for r in runs)
    if not args.skip_sanity and has_decoder:
        if not sanity_gate(args.dry_run):
            print("Aborting: sanity gate did not pass.")
            return 2

    print(f"\n=== {len(runs)} run(s) queued ===")
    failures = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {run['name']}  ({run['stage']}, {run['model_type']})")
        cmd = build_command(run, args.resume, auto_download=not args.no_download)
        if cmd is None:
            continue
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"  [FAIL] {run['name']} exited {rc}")
            failures.append(run["name"])
            if not args.keep_going:
                print("Stopping (use --keep-going to continue past failures).")
                return 3

    print("\n=== done ===")
    if failures:
        print("Failed runs:", ", ".join(failures))
        return 4
    print("All runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
