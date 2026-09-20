"""Train the per-frame shadow-arm switch.

    python training/train_shadow_gate.py --manifest <manifest> \
        --init-checkpoint <dir>/sdr2hdr_image_v5/shipped_v5_step81000.pt \
        --output-dir <dir>/sdr2hdr_shadow_v1 --steps 3000

Why this and not another residual-scale run: the 1 Sep 2026 ablation measured
that disabling the shadow arm moves clean input by **+3.51 dB** (the regression
against the analytic baseline inverts to +0.51) and costs **1.10 dB** on
degraded input. So the shadow path is the whole of the clean regression and most
of the degraded gain, and the right setting is a BINARY choice on the
clean-versus-degraded axis.

That axis is the tractable part. A frame-grouped cross-validated classifier on
these same features reaches **75.5%**, where the continuous residual-scale
target had only 3% of its variance explained. At 75.5% a switch is worth
**+2.65 dB on clean for 0.27 dB on hard**.

And unlike the residual scale, the target here needs no oracle: at training time
we KNOW whether we degraded the frame. This is ordinary supervised binary
classification on a label we generate ourselves.
"""
from __future__ import annotations

import argparse, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import SDR2HDRNet, luminance, sdr_to_baseline_hdr   # noqa: E402
from training.gate_oracle import gate_view                            # noqa: E402
from training.train_gate import WholeFrameDataset                     # noqa: E402
from training.train_sdr2hdr import (deterministic_eval_order,         # noqa: E402
                                    selection_score, _atomic_save)


def gate_inputs(model: SDR2HDRNet, sdr: torch.Tensor, max_side: int):
    """Features from the downscaled view, statistics from the native frame."""
    sdr = sdr.float().clamp(0.0, 1.0)
    view = gate_view(sdr, max_side)
    with torch.no_grad():                       # the backbone is frozen
        _, _, mid = model.encode(view, sdr_to_baseline_hdr(view))
    return mid, sdr, luminance(sdr)


def evaluate(model, loader, device, max_batches, max_side):
    """Accuracy is the number that matters -- the ablation fixes what it buys."""
    model.eval()
    correct = total = 0
    conf = {"clean_on": 0, "clean_off": 0, "hard_on": 0, "hard_off": 0}
    weights = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            w = model.shadow_gate(*gate_inputs(model, batch["sdr"].to(device), max_side)).flatten()
            degraded = batch["degraded"].to(device).float()
            on = (w >= 0.5).float()
            correct += float((on == degraded).sum()); total += w.numel()
            for j in range(w.numel()):
                key = ("hard" if degraded[j] > 0.5 else "clean") + ("_on" if on[j] > 0.5 else "_off")
                conf[key] += 1
            weights.append(w.detach().cpu())
    model.train()
    w = torch.cat(weights) if weights else torch.zeros(1)
    return {"accuracy": correct / max(total, 1), "mean_weight": float(w.mean()),
            "min_weight": float(w.min()), "max_weight": float(w.max()), **conf}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--degradation-probability", type=float, default=0.5,
                    help="Must stay near 0.5: this is a balanced binary problem and a "
                         "skewed prior is the easiest way to get a head that always "
                         "answers the majority class.")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=128)
    ap.add_argument("--best-smoothing", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--eval-seed", type=int, default=20260901,
                    help="Fixed validation degradation seed shared across training seeds")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(payload.get("config", {}) or {}, shadow_conditioning=True)
    missing, unexpected = model.load_state_dict(payload.get("model", payload), strict=False)
    stray = [k for k in missing if not k.startswith("shadow_gate.")]
    if stray or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing {stray}, unexpected {list(unexpected)}")
    model.to(device)
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith("shadow_gate."))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_set = WholeFrameDataset(args.manifest, "train", args.max_side,
                                  args.degradation_probability, seed=args.seed)
    val_set = WholeFrameDataset(args.manifest, "val", args.max_side,
                                deterministic=True, seed=args.eval_seed)
    val_set = Subset(val_set, deterministic_eval_order(len(val_set)))
    largs = dict(batch_size=1, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, **largs)
    val_loader = DataLoader(val_set, shuffle=False, **largs)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    config = vars(args).copy()
    config.update({"base_channels": (payload.get("config") or {}).get("base_channels", 32),
                   "shadow_conditioning": True, "objective": "degradation_bce"})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log = out_dir / "train.jsonl"

    print("\n" + "=" * 66)
    print("   shadow-arm switch -- supervised on the degradation label")
    print("=" * 66)
    print(f"   backbone   : {args.init_checkpoint}  (frozen)")
    print(f"   trainable  : {trainable:,} parameters")
    print(f"   ceiling    : 100% accuracy is +0.51 dB clean / +1.43 dB hard over baseline")
    print(f"   as shipped : -3.00 dB clean / +1.43 dB hard")
    print("=" * 66 + "\n")

    start = evaluate(model, val_loader, device, min(args.eval_batches, 32), args.max_side)
    print(f"[start] accuracy {100*start['accuracy']:.1f}%  mean weight {start['mean_weight']:.3f} "
          f"(an untrained head answers 'on' for everything, so ~50% here is expected)\n")

    best, history, step, t0 = math.inf, [], 0, time.time()
    model.train(); it = iter(train_loader)
    while step < args.steps:
        opt.zero_grad(set_to_none=True); acc = 0.0
        for _ in range(args.grad_accum):
            try: batch = next(it)
            except StopIteration: it = iter(train_loader); batch = next(it)
            w = model.shadow_gate(*gate_inputs(model, batch["sdr"].to(device), args.max_side)).flatten()
            target = batch["degraded"].to(device).float()
            loss = F.binary_cross_entropy(w.clamp(1e-6, 1 - 1e-6), target)
            (loss / args.grad_accum).backward()
            acc += float(loss.detach()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); step += 1

        if step % 25 == 0 or step == 1:
            print(f"[shadow] step {step}/{args.steps} bce={acc:.4f} "
                  f"frames/s={step*args.grad_accum/max(time.time()-t0,1e-6):.2f}")
        with log.open("a", encoding="utf-8") as h:
            h.write(json.dumps({"step": step, "bce": acc}) + "\n")

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, val_loader, device, args.eval_batches, args.max_side)
            print(f"[eval] step {step}: {json.dumps(m)}")
            score = -m["accuracy"]
            history.append(score)
            sm = selection_score(history, args.best_smoothing)
            with log.open("a", encoding="utf-8") as h:
                h.write(json.dumps({"step": step, "eval": m, "score_raw": score,
                                    "score_smoothed": sm}) + "\n")
            if sm < best:
                best = sm
                _atomic_save({"model": model.state_dict(), "step": step, "best": best,
                              "config": config}, out_dir / "best.pt")
                print(f"       new best (smoothed accuracy {-sm:.4f}) -> best.pt")

    print(f"\n   done in {(time.time()-t0)/60:.1f} min -> {out_dir/'best.pt'}")
    print("   score it with:  score_checkpoint.ps1 -Checkpoint <best.pt> -Name shadow_v1\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
