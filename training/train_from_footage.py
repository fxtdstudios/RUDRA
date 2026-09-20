#!/usr/bin/env python3
"""Train RUDRA on your own footage. One command, seven stages, resumable.

    python training/train_from_footage.py /path/to/your_hdr_footage

That is the whole thing. It inventories what you have, builds SDR/HDR pairs,
splits them by SCENE, verifies the corpus, trains the backbone, trains the
shadow gate on top, and scores the result against the analytic baseline on
your own held-out frames.

WHY THIS EXISTS. The pipeline underneath is five programs with about forty
flags between them, and the defaults that matter are not the ones argparse
prints. Every check in it was added after something got past it once. A
studio should be able to point this at a folder and come back to a model and a
number, and only reach for the underlying scripts when they want something the
defaults do not give.

    --work DIR       where everything lands (default: work/)
    --steps N        backbone steps (default 100000; the shipped model is 81000)
    --device cuda    or cpu, if you enjoy waiting
    --from STAGE     restart from a stage: scan, pairs, manifests, verify,
                     backbone, gate, bench
    --only STAGE     run exactly one stage
    --dry-run        print the commands and stop
    --yes            do not ask before the long stages

RESUMABLE BY DEFAULT. A stage whose output already exists is skipped with a
note. Delete that output, or pass --from, to make it run again. Interrupting
the backbone and re-running picks up from its last checkpoint.

WHAT IT WILL NOT DO. It will not train from SDR-only footage, because there
would be no target to learn: the SDR half of every pair is generated from your
HDR. If the scan finds no usable range it stops and says so rather than
producing a model that has learned the identity function.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STAGES = ("scan", "pairs", "manifests", "verify", "backbone", "gate", "bench")

# Stages that take long enough to be worth confirming, and roughly how long on
# one RTX 4080. Honest ranges, not marketing: pairs is disk-bound and scales
# with how much footage you point at it.
SLOW = {
    "pairs": "minutes to hours, disk-bound",
    "backbone": "the long one — hours",
}


class Stop(SystemExit):
    """A failure the user can act on, rather than a traceback."""

    def __init__(self, message: str, hint: str = ""):
        body = f"\n  STOPPED: {message}"
        if hint:
            body += f"\n  -> {hint}"
        super().__init__(body + "\n")


def say(message: str) -> None:
    print(message, flush=True)


def banner(index: int, name: str, detail: str = "") -> None:
    say("")
    say(f"[{index}/{len(STAGES)}] {name}" + (f"   ({detail})" if detail else ""))
    say("-" * 68)


def run(command: list[str], dry: bool) -> None:
    say("   $ " + " ".join(str(c) for c in command))
    if dry:
        return
    started = time.time()
    result = subprocess.run([str(c) for c in command], cwd=str(REPO))
    if result.returncode != 0:
        raise Stop(f"{Path(command[1]).name} exited {result.returncode}",
                   "The error above is from that script. Nothing after this "
                   "stage has run; fix it and re-run this command.")
    say(f"   done in {time.time() - started:.0f}s")


def confirm(name: str, detail: str, assume_yes: bool) -> None:
    if assume_yes or not sys.stdin.isatty():
        return
    reply = input(f"   {name} takes {detail}. Continue? [Y/n] ").strip().lower()
    if reply and not reply.startswith("y"):
        raise Stop("stopped at your request",
                   f"Resume with --from {name} when you are ready.")


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def preflight(source: Path, work: Path, device: str) -> None:
    """Everything that can be known before any work is done."""
    say("preflight")
    say("-" * 68)

    if not source.is_dir():
        raise Stop(f"{source} is not a directory",
                   "Point this at the folder holding your HDR footage.")

    for module, why in (("torch", "training"), ("cv2", "image IO"),
                        ("numpy", "everything")):
        try:
            __import__(module)
        except ImportError:
            raise Stop(f"{module} is not installed ({why})",
                       "pip install -r requirements.txt")
    say("   packages   : ok")

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise Stop("torch reports no CUDA device",
                   "Pass --device cpu to run anyway. The backbone will take "
                   "days rather than hours.")
    if device == "cuda":
        say(f"   gpu        : {torch.cuda.get_device_name(0)}")
    else:
        say("   gpu        : none, running on cpu")

    if not shutil.which("ffmpeg"):
        say("   ffmpeg     : NOT FOUND — video sources will be skipped. "
            "Install it if your footage is MXF/MOV/MP4.")
    else:
        say("   ffmpeg     : ok")

    work.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(work).free / 1e9
    say(f"   free disk  : {free_gb:,.0f} GB at {work}")
    if free_gb < 50:
        say("   NOTE: pairs are large. 50 GB is a tight floor and the pairs "
            "stage will stop rather than half-write if it runs out.")


def check_inventory(inventory: Path) -> None:
    """Refuse to build pairs from footage with no range in it."""
    usable = total = 0
    with inventory.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            peak = record.get("peak_nits") or 0.0
            if record.get("encoding") not in (None, "", "unknown") and peak > 300.0:
                usable += 1
    say(f"   inventory  : {total} file(s), {usable} with usable HDR range")
    if total == 0:
        raise Stop("the scan found no readable image or video files",
                   "Check the path. Accepted: .exr .hdr .tif .dpx .png .jxl "
                   ".avif .heic and .mxf .mov .mp4 .mkv .avi .m2ts .ts .webm")
    if usable == 0:
        raise Stop("nothing in that folder carries HDR range",
                   "RUDRA learns to invert a tone map, so it needs the answer: "
                   "scene-referred EXR, or graded HDR masters. There is "
                   "nothing to learn from 8-bit SDR.")
    if usable < 200:
        say(f"   WARNING: {usable} usable files is very little. Scene diversity "
            "matters more than frame count -- expect the numbers to describe "
            "your test split rather than your footage.")


def report(bench_json: Path) -> None:
    """The only number most people need."""
    if not bench_json.is_file():
        return
    payload = json.loads(bench_json.read_text(encoding="utf-8"))
    say("")
    say("=" * 68)
    say("   YOUR MODEL, on your own held-out frames, degraded condition")
    say("=" * 68)
    for key in ("pu21_psnr_db", "pu21_db", "cvvdp_jod"):
        if key in payload:
            say(f"   {key:16s} {payload[key]:.3f}")
    say("")
    say("   Both are measured against the analytic inverse tone map the")
    say("   network sits on top of, so this is 'better than doing nothing'.")
    say("   Read the JOD: it is calibrated against human observers, and the")
    say("   two metrics disagree by orders of magnitude on small differences.")
    say(f"   Full per-frame results: {bench_json}")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path, help="folder of your HDR footage")
    ap.add_argument("--work", type=Path, default=Path("work"))
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--crops", type=int, default=3,
                    help="random crops per source image. More data from the "
                         "same footage, but they are not new scenes.")
    ap.add_argument("--video-stride", type=int, default=2,
                    help="keep every Nth video frame. Consecutive frames are "
                         "nearly the same picture.")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--from", dest="start", choices=STAGES, default=None)
    ap.add_argument("--only", choices=STAGES, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    work = args.work.resolve()
    py = sys.executable
    inventory = work / "inventory.jsonl"
    pairs = work / "pairs"
    manifest = work / "sdr_hdr_manifest.jsonl"
    video_manifest = work / "video_manifest_9f.jsonl"
    backbone = work / "checkpoints" / "image"
    gate = work / "checkpoints" / "shadow"
    bench = work / "bench"
    results = work / "results.json"

    wanted = set(STAGES)
    if args.only:
        wanted = {args.only}
    elif args.start:
        wanted = set(STAGES[STAGES.index(args.start):])

    def skip(stage: str, output: Path) -> bool:
        if stage not in wanted:
            say("   skipped (not selected)")
            return True
        if args.start == stage or args.only == stage:
            return False
        if output.exists():
            say(f"   already done: {output}  (--from {stage} to redo)")
            return True
        return False

    if not args.dry_run:
        preflight(args.source, work, args.device)

    banner(1, "scan", "what is on disk and what range it carries")
    if not skip("scan", inventory):
        run([py, REPO / "pipeline" / "scan_sources.py", args.source,
             "--out", inventory], args.dry_run)
        if not args.dry_run:
            check_inventory(inventory)

    banner(2, "pairs", SLOW["pairs"])
    if not skip("pairs", pairs):
        confirm("pairs", SLOW["pairs"], args.yes)
        run([py, REPO / "pipeline" / "prepare_pairs.py",
             "--inventory", inventory, "--dst", pairs,
             "--mode", "log2_extended", "--crops", args.crops,
             "--crop-size", 512, "--video-stride", args.video_stride,
             "--seed", args.seed], args.dry_run)

    banner(3, "manifests", "splits held out by SCENE, not by frame")
    if not skip("manifests", manifest):
        run([py, REPO / "pipeline" / "build_manifests.py",
             "--pairs-dir", pairs, "--out-dir", work,
             "--val-frac", 0.10, "--test-frac", 0.10,
             "--seed", args.seed], args.dry_run)

    banner(4, "verify", "the cheapest hour you will spend")
    if "verify" in wanted:
        command = [py, REPO / "pipeline" / "verify_dataset.py",
                   "--pairs-dir", pairs, "--manifest", manifest]
        if video_manifest.exists() or args.dry_run:
            command += ["--video-manifest", video_manifest]
        try:
            run(command, args.dry_run)
        except Stop:
            raise Stop("the corpus did not pass verification",
                       "The failures above are real -- every check exists "
                       "because something got past it once. Fix the corpus, "
                       "or re-run build_manifests.py with --allow-problems if "
                       "you have read them and disagree.")
    else:
        say("   skipped (not selected)")

    banner(5, "backbone", SLOW["backbone"])
    if not skip("backbone", backbone / "best.pt"):
        confirm("backbone", SLOW["backbone"], args.yes)
        command = [py, REPO / "training" / "train_sdr2hdr.py",
                   "--mode", "image", "--manifest", manifest,
                   "--output-dir", backbone, "--steps", args.steps,
                   "--best-metric", "composite_gain", "--device", args.device]
        latest = backbone / "last.pt"
        if latest.exists():
            say("   resuming from the last checkpoint")
            command += ["--resume", latest]
        run(command, args.dry_run)

    banner(6, "gate", "fifteen minutes on one 4080")
    if not skip("gate", gate / "best.pt"):
        run([py, REPO / "training" / "train_shadow_gate.py",
             "--manifest", manifest,
             "--init-checkpoint", backbone / "best.pt",
             "--output-dir", gate, "--seed", args.seed,
             "--device", args.device], args.dry_run)

    banner(7, "bench", "your model against doing nothing")
    if not skip("bench", results):
        run([py, REPO / "training" / "export_bench_pairs.py",
             "--checkpoint", gate / "best.pt", "--manifest", manifest,
             "--out", bench, "--condition", "hard", "--test-name", "mine",
             "--device", args.device], args.dry_run)
        run([py, "-m", "rudra.delivery.cli", "bench", bench,
             "--nits-scale", 203, "--test-dir", "mine",
             "--output", results], args.dry_run)
        if not args.dry_run:
            report(results)

    say("")
    if args.dry_run:
        say("Dry run. Nothing was executed.")
        return 0
    say(f"Model: {gate / 'best.pt'}")
    say("")
    say("Use it:")
    say(f"   python training/infer_sdr2hdr.py input/ --output-dir out/ "
        f"--checkpoint {gate / 'best.pt'}")
    say("")
    say("Or make it the default everywhere, including RUDRA Studio:")
    say(f"   set RUDRA_CHECKPOINT_ROOTS={work / 'checkpoints'}      (Windows)")
    say(f"   export RUDRA_CHECKPOINT_ROOTS={work / 'checkpoints'}   (Linux/macOS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
