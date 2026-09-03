#!/usr/bin/env python3
"""Run ExpandNet over an exported `sdr/` tree, for the comparison the paper owes.

Every number in `PAPER_DRAFT_2026-08-29.md` is against RUDRA's own analytic
baseline. §9 calls that the paper's largest gap. This closes the mechanical half
of it for the first third-party method: ExpandNet (Marnerides et al., CGF 2018),
which is the lightest of the four public methods §2 names.

    git clone https://github.com/dmarnerides/hdr-expandnet.git
    python training/export_bench_pairs.py ... --write-sdr        # <bench>/sdr/**.png
    python training/run_expandnet.py --sdr <bench>/sdr --repo hdr-expandnet \\
        --out <bench>/expandnet_raw --device cuda
    python training/import_method_output.py --out <bench> \\
        --from <bench>/expandnet_raw --name expandnet
    rudra bench <bench> --nits-scale 203 --test-dir expandnet

THIS RUNS THEIR CODE, NOT A RE-IMPLEMENTATION. `model.ExpandNet` and the two
helpers come from the clone by import; the pre- and post-processing below is
`expand.py:create_images` line for line, because a comparison against a method
you reimplemented from the paper is a comparison against your reading of it.
The only thing changed is where the pixels come from and go.

WHAT COMES OUT, AND WHY IT NEEDS ALIGNING
-----------------------------------------
ExpandNet ends in a sigmoid and `expand.py` then min/max-normalises the result
to [0,1] per image. The output is therefore *relative* radiance with no nit
anchor -- which is not a flaw, it is what the method predicts. Scoring it
directly against a reference in cd/m^2 would measure its exposure guess.
`import_method_output.py` fits one global scalar per frame; run it on RUDRA's
own tree too and report both rows, or the table flatters us for free.

The input tree must be the one `--write-sdr` produced: those are the exact 8-bit
frames RUDRA was given, after the same `--condition` and `--max-side`. Handing
ExpandNet a different decode of the same source would make the two columns
answer different questions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.exr import write_exr                          # noqa: E402


def load_expandnet(clone: Path, weights: Path | None, device):
    """Import the author's model and weights from a clone of their repository."""
    import torch
    if not (clone / "model.py").is_file():
        raise SystemExit(f"error: {clone} does not look like hdr-expandnet "
                         f"(no model.py). Clone it first:\n"
                         f"  git clone https://github.com/dmarnerides/hdr-expandnet.git")
    sys.path.insert(0, str(clone))
    from model import ExpandNet                                   # noqa: E402
    from util import cv2torch, map_range, torch2cv                # noqa: E402

    path = weights or (clone / "weights.pth")
    if not path.is_file():
        raise SystemExit(f"error: no weights at {path}")
    net = ExpandNet()
    net.load_state_dict(torch.load(path, map_location=lambda s, l: s))
    net.eval().to(device)
    return net, cv2torch, map_range, torch2cv


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sdr", required=True, help="the <bench>/sdr tree")
    parser.add_argument("--repo", required=True, help="a clone of hdr-expandnet")
    parser.add_argument("--out", required=True, help="where to write their EXRs")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: cuda when available)")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="their stitched-inference tile; 0 runs whole frames")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import torch
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sdr_root, out_root = Path(args.sdr), Path(args.out)
    frames = sorted(p for p in sdr_root.rglob("*") if p.suffix.lower() == ".png")
    if not frames:
        raise SystemExit(f"error: no PNGs under {sdr_root}. Did you pass --write-sdr?")
    if args.limit:
        frames = frames[:args.limit]

    net, cv2torch, map_range, torch2cv = load_expandnet(
        Path(args.repo), Path(args.weights) if args.weights else None, device)

    print("\n" + "=" * 66)
    print("   ExpandNet (Marnerides et al. 2018) on the RUDRA test split")
    print("=" * 66)
    print(f"   frames     : {len(frames)} under {sdr_root}")
    print(f"   weights    : {args.weights or Path(args.repo) / 'weights.pth'}")
    print(f"   device     : {device}")

    started, written, failed = time.time(), 0, []
    for position, path in enumerate(frames, 1):
        rel = path.relative_to(sdr_root)
        try:
            # expand.py:create_images, unchanged.
            loaded = cv2.imread(str(path), flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR)
            if loaded is None:
                raise RuntimeError("cv2 could not read it")
            ldr = map_range(loaded.astype("float32"))
            t_in = cv2torch(ldr).to(device)
            with torch.no_grad():
                pred = net.predict(t_in, args.patch_size).cpu()
            bgr = map_range(torch2cv(pred), 0, 1)
        except Exception as exc:                                  # noqa: BLE001
            failed.append((rel.as_posix(), f"{type(exc).__name__}: {exc}"))
            continue

        dest = out_root / rel.parent / f"{path.stem}.exr"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Their pipeline hands cv2 a BGR array; we store RGB, like every other
        # tree the benchmark reads.
        write_exr(dest, np.ascontiguousarray(bgr[:, :, ::-1], dtype=np.float32),
                  half=True)
        written += 1
        if position % 25 == 0 or position == len(frames):
            rate = position / max(time.time() - started, 1e-6)
            print(f"   {position:>5}/{len(frames)}  {rate:5.2f} frames/s")

    manifest = {
        "method": "ExpandNet",
        "citation": "Marnerides, Bashford-Rogers, Hatchett, Debattista. "
                    "Computer Graphics Forum 37(2), 2018. arXiv:1803.02266",
        "code": "https://github.com/dmarnerides/hdr-expandnet",
        "repo": str(Path(args.repo).resolve()),
        "weights": str((Path(args.weights) if args.weights
                        else Path(args.repo) / "weights.pth").resolve()),
        "sdr_tree": str(sdr_root.resolve()),
        "patch_size": args.patch_size,
        "device": str(device),
        "frames": written,
        "failed": failed,
        "units": "relative, min/max normalised to [0,1] per frame by their "
                 "postprocess -- align before scoring",
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "expandnet.json").write_text(json.dumps(manifest, indent=2),
                                             encoding="utf-8")
    print("=" * 66)
    print(f"   wrote {written} frame(s) to {out_root}")
    if failed:
        print(f"   failed {len(failed)}: " + ", ".join(r for r, _ in failed[:3])
              + (" ..." if len(failed) > 3 else ""))
    print(f"\n   python training/import_method_output.py --out <bench> "
          f"--from {out_root} --name expandnet\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
