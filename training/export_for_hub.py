#!/usr/bin/env python3
"""Prepare the committed checkpoints for huggingface.co/fxtdstudios/RUDRA.

As of 3 Sep 2026 the Hub repo holds 14 VAE decoders (line A), flat at the root,
last touched 29 June. None of the SDR->HDR work is there: not v5, not v6, not
the shadow gate the paper is about. This builds an upload directory that adds
them, in safetensors rather than pickle, with each file's architecture config
carried in its own metadata so a downloader needs nothing else to load it.

    python training/export_for_hub.py --out dist/hub
    huggingface-cli upload fxtdstudios/RUDRA dist/hub . --repo-type model

It does NOT move the existing decoders. Renaming on the Hub means deleting the
old paths, every published URL breaks, and that is a decision with consequences
outside this repo -- so the layout is additive and the model card documents what
is already there. `--decoder-folders` prints the commands to reorganise them, to
run or not.

Nothing here needs a token; the upload is a separate step you run yourself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The SDR->HDR models, in the order the card lists them.
MODELS = [
    ("sdr2hdr_shadow_v1", "the shipped model: v5 backbone + trained shadow gate"),
    ("sdr2hdr_shadow_s2", "same recipe, seed 2"),
    ("sdr2hdr_shadow_s3", "same recipe, seed 3"),
    ("sdr2hdr_image_v5", "the backbone alone, no gate"),
    ("sdr2hdr_image_v6", "4x capacity ablation, 64 base channels"),
    ("sdr2hdr_temporal_v1", "temporal refiner -- NOT an SDR2HDRNet"),
]


def convert(src: Path, dest: Path) -> dict:
    """.pt payload dict -> .safetensors with the config in its metadata."""
    import torch
    from safetensors.torch import save_file

    payload = torch.load(src, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    meta = {
        "architecture": "SDR2HDRNet",
        "config": json.dumps(payload.get("config", {})),
        "step": str(payload.get("step", "")),
        "source": "https://github.com/fxtdstudios/RUDRA",
        "units_in": "8-bit normalised sRGB RGB",
        "units_out": "scene-linear RGB normalised to 10000 nits",
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, dest, metadata=meta)
    return {"params": int(sum(v.numel() for v in state.values())),
            "step": payload.get("step"),
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}


def verify(dest: Path) -> int:
    """Load it back the way a downloader would. Returns the parameter count."""
    import torch  # noqa: F401
    from safetensors import safe_open
    from safetensors.torch import load_file

    from rudra.sdr2hdr import SDR2HDRNet
    with safe_open(dest, framework="pt") as f:
        meta = f.metadata()
    model = SDR2HDRNet.from_config(json.loads(meta["config"]))
    model.load_state_dict(load_file(dest), strict=True)
    return int(sum(p.numel() for p in model.parameters()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="dist/hub")
    parser.add_argument("--checkpoints", default=str(REPO / "checkpoints"))
    parser.add_argument("--decoder-folders", action="store_true",
                        help="print the commands that would move the 14 decoders "
                             "into decoders/full and decoders/turbo, and stop")
    args = parser.parse_args()

    if args.decoder_folders:
        print("\n   Moving published files breaks every existing URL, so this only")
        print("   prints. Run it yourself if you want the folders:\n")
        for kind in ("full", "turbo"):
            print(f"   huggingface-cli repo-files fxtdstudios/RUDRA ...  # list, then per file:")
            print(f"   #   download rudra_{kind}_decoder_<b>_ema.safetensors")
            print(f"   #   upload   to decoders/{kind}/rudra_{kind}_decoder_<b>_ema.safetensors")
            print(f"   #   delete   the old flat path")
        print("\n   The model card documents the flat layout, so doing nothing is fine.\n")
        return 0

    src_root, out = Path(args.checkpoints), Path(args.out)
    (out / "sdr2hdr").mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print("   preparing the SDR->HDR models for the Hub")
    print("=" * 72)
    index = []
    for name, note in MODELS:
        src = src_root / f"{name}.pt"
        if not src.is_file():
            print(f"   {name:<22} MISSING at {src}")
            continue
        dest = out / "sdr2hdr" / f"{name}.safetensors"
        info = convert(src, dest)
        # The temporal refiner is a different architecture; it converts fine but
        # must not be advertised as loadable by SDR2HDRNet.
        loaded = None
        if name != "sdr2hdr_temporal_v1":
            loaded = verify(dest)
            assert loaded == info["params"], (name, loaded, info["params"])
        cfg = src_root / f"{name}.config.json"
        if cfg.is_file():
            shutil.copy(cfg, out / "sdr2hdr" / cfg.name)
        print("   %-22s %9d params  %6.2f MB  %s"
              % (name, info["params"], dest.stat().st_size / 1e6,
                 "verified" if loaded else "converted (not an SDR2HDRNet)"))
        index.append({"file": f"sdr2hdr/{name}.safetensors", "note": note, **info})

    (out / "sdr2hdr" / "SHA256SUMS").write_text(
        "\n".join(f"{m['sha256']}  {Path(m['file']).name}" for m in index) + "\n",
        encoding="utf-8")
    (out / "sdr2hdr" / "index.json").write_text(json.dumps(index, indent=2),
                                                encoding="utf-8")

    card = REPO / "docs" / "HUB_MODEL_CARD.md"
    if card.is_file():
        shutil.copy(card, out / "README.md")
        print(f"   {'README.md':<22} model card from {card.relative_to(REPO)}")
    else:
        print(f"   WARNING: no model card at {card}; the Hub README will not be updated")

    print("=" * 72)
    print(f"   {len(index)} model(s) -> {out}\n")
    print("   huggingface-cli login")
    print(f"   huggingface-cli upload fxtdstudios/RUDRA {out} . --repo-type model\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
