#!/usr/bin/env python3
"""
download_models.py — fetch any missing backbone/VAE/text-encoder weights.

Reads the URLs already defined in config/model_map.py (RADIANCE_MODEL_MAP),
checks your ComfyUI models tree, and downloads only the files that are missing,
into the correct subfolder (diffusion_models / vae / text_encoders / checkpoints).
Downloads are resumable — re-run if interrupted.

Usage
-----
  python training/download_models.py --list                 # show what exists / is missing
  python training/download_models.py --dry-run              # show what WOULD download
  python training/download_models.py                        # download all missing
  python training/download_models.py --only flux1-dev-fp8.safetensors,ae.safetensors
  python training/download_models.py --comfy "D:\\A.I\\ComfyUI\\models"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

# ComfyUI "type" tag (from model_map) -> models subfolder.
TYPE_TO_SUBDIR = {
    "diffusion_models": "diffusion_models",
    "text_encoders": "text_encoders",
    "vae": "vae",
    "checkpoints": "checkpoints",
    "loras": "loras",
    "clip": "clip",
}

DEFAULT_COMFY = r"D:\A.I\ComfyUI\models"


def _load_model_map():
    spec = importlib.util.spec_from_file_location(
        "model_map", os.path.join(_REPO, "config", "model_map.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RADIANCE_MODEL_MAP


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def _download(url: str, dest: Path):
    """Resumable streaming download with a simple progress line."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    req = urllib.request.Request(url, headers={"User-Agent": "rudra-dl/1.0"})
    if existing:
        req.add_header("Range", f"bytes={existing}-")
        print(f"   resuming at {_human(existing)}")

    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0)) + existing
        mode = "ab" if existing else "wb"
        done = existing
        chunk = 1 << 20  # 1 MB
        with open(tmp, mode) as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100.0 * done / total
                    sys.stdout.write(f"\r   {pct:5.1f}%  {_human(done)} / {_human(total)}")
                    sys.stdout.flush()
    sys.stdout.write("\n")
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser(description="Download missing RUDRA backbone models.")
    ap.add_argument("--comfy", default=os.environ.get("COMFY_MODELS_DIR", DEFAULT_COMFY),
                    help="ComfyUI models root (contains diffusion_models/, vae/, ...).")
    ap.add_argument("--only", default="", help="Comma-separated filenames to consider.")
    ap.add_argument("--list", action="store_true", help="Show present/missing and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Show what would download.")
    args = ap.parse_args()

    mm = _load_model_map()
    comfy = Path(args.comfy)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    present, missing = [], []
    for name, info in mm.items():
        if only and name not in only:
            continue
        sub = TYPE_TO_SUBDIR.get(info.get("type", ""), info.get("type", "diffusion_models"))
        dest = comfy / sub / name
        (present if dest.exists() else missing).append((name, info["url"], dest))

    print(f"ComfyUI models root: {comfy}")
    print(f"  present: {len(present)}   missing: {len(missing)}\n")
    for name, _, dest in present:
        print(f"  ✓ {name:<40} {dest}")
    for name, _, dest in missing:
        print(f"  ✗ {name:<40} -> {dest}")

    if args.list:
        return 0
    if not missing:
        print("\nNothing to download — all known models present.")
        return 0
    if args.dry_run:
        print(f"\n[dry-run] would download {len(missing)} file(s).")
        return 0

    print()
    for i, (name, url, dest) in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {name}")
        try:
            _download(url, dest)
            print(f"   saved -> {dest}")
        except Exception as e:
            print(f"   [FAIL] {name}: {e}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
