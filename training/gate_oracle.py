"""Oracle targets for the per-frame residual scale.

The conditioning head could not learn from the reconstruction loss: two runs of
8 000 steps left it emitting alpha in [1.0125, 1.0802] with correlation -0.037
against frame headroom, where the oracle wanted 0.125 on low-headroom frames.
Three reasons, none of them the head's: the training mix under-represents the
band the failure lives in (27.2% below 400 nits against the test split's
45.2%), the training metric is a censored log-L1 on crops where the benchmark
is PU21-PSNR on full frames, and `log_l1` is 0.037 of a 0.141 total behind
terms the frozen backbone holds constant.

So stop asking the head to infer the scale from a loss that does not contain
the defect, and supervise it directly against the quantity the benchmark
measures. The target is one scalar per frame, found by a line search over the
same composite the network already builds -- an ill-posed indirect objective
becomes a well-posed regression.

Everything here is torch so the search runs on the GPU beside training.
`rudra/delivery/` stays torch-free, so the PU21 constants are imported from it
rather than copied: one definition, and tests/test_gate_oracle_2026_08_29.py
pins this implementation against the numpy one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.delivery.bench import _PU21, _PU21_MAX, _PU21_MIN   # noqa: E402

NETWORK_PEAK_NITS = 10_000.0
LOG_SCALE = 16.0
MAX_HDR = 4.0
# The sweep that measured the opportunity used this grid; keeping it means the
# targets here and the +5.84 dB clean / +0.29 dB hard figures in the README are
# the same quantity rather than two things that merely sound alike.
DEFAULT_ALPHAS = torch.arange(0.0, 1.2501, 0.0625)


def pu21_encode(nits: torch.Tensor) -> torch.Tensor:
    """Perceptually uniform encoding of absolute luminance, in torch."""
    p0, p1, p2, p3, p4, p5, p6 = _PU21
    y = nits.clamp(_PU21_MIN, _PU21_MAX).to(torch.float64)
    yp = y.pow(p3)
    return p6 * (((p0 + p1 * yp) / (1.0 + p2 * yp)).pow(p4) - p5)


def pu_psnr(test_nits: torch.Tensor, ref_nits: torch.Tensor) -> torch.Tensor:
    """PU21-PSNR in dB, per sample. Inputs (B,...) in absolute nits."""
    test, ref = pu21_encode(test_nits), pu21_encode(ref_nits)
    peak = pu21_encode(torch.tensor(_PU21_MAX, dtype=torch.float64))
    mse = (test - ref).pow(2).flatten(1).mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(peak * peak / mse)


def effective_residual(prediction: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    """The log-domain term alpha scales, recovered from a single forward pass.

    `forward` adds `residual * residual_gate` in log space, and the gate is not
    returned -- only the composed prediction is. Differencing the two log
    images gives exactly that product, so one forward at scale 1 is enough to
    evaluate every alpha on the grid. Requires the RAW prediction
    (`preserve_outside=False`); the preserve blend is reapplied below.
    """
    return (torch.log1p(prediction * LOG_SCALE)
            - torch.log1p(baseline * LOG_SCALE))


def composite_at(baseline: torch.Tensor, effective: torch.Tensor,
                 recovery: torch.Tensor | None, alpha: torch.Tensor) -> torch.Tensor:
    """The network's own composite tail, at an arbitrary per-sample scale."""
    ceiling = torch.log1p(torch.tensor(MAX_HDR * LOG_SCALE,
                                       device=baseline.device, dtype=baseline.dtype))
    while alpha.ndim < baseline.ndim:
        alpha = alpha.unsqueeze(-1)
    pred_log = (torch.log1p(baseline * LOG_SCALE) + effective * alpha).clamp(0.0, ceiling)
    pred = torch.expm1(pred_log) / LOG_SCALE
    if recovery is not None:
        pred = baseline + recovery * (pred - baseline)
    return pred


@torch.no_grad()
def oracle_alpha(baseline: torch.Tensor, effective: torch.Tensor,
                 recovery: torch.Tensor | None, target: torch.Tensor,
                 alphas: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The scale that maximises PU21-PSNR against the target, per sample.

    Returns (alpha, gain_db_over_alpha_zero). Alpha zero is the analytic
    baseline exactly -- `composite_at` with alpha 0 reduces to it -- so the
    second value is the gain over doing nothing, which is what the benchmark
    reports and what makes a target comparable across frames.
    """
    grid = (DEFAULT_ALPHAS if alphas is None else alphas).to(baseline.device)
    scores = []
    for value in grid:
        alpha = value.expand(baseline.shape[0])
        pred = composite_at(baseline, effective, recovery, alpha)
        scores.append(pu_psnr(pred * NETWORK_PEAK_NITS, target * NETWORK_PEAK_NITS))
    stacked = torch.stack(scores, dim=1)                    # (B, len(grid))
    best = stacked.argmax(dim=1)
    baseline_db = stacked[:, 0]
    return grid[best], stacked.gather(1, best[:, None]).squeeze(1) - baseline_db


def gate_view(frame: torch.Tensor, max_side: int = 512) -> torch.Tensor:
    """Downscale a whole frame the way `predict_residual_scale` does.

    The head is a whole-frame judgement, so it has to be trained on whole
    frames. Wiring it into the ordinary training step fed it 384-pixel crops
    while inference fed it the entire frame -- a component asked at test time
    for something it never saw while learning, which on its own could produce
    the near-constant 1.04 the trained head emits. Same `mode="area"` and the
    same cap as inference, so the two views match.
    """
    if frame.ndim != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(frame.shape)}")
    longest = max(frame.shape[-2:])
    if longest <= max_side:
        return frame
    scale = max_side / longest
    size = (max(1, round(frame.shape[-2] * scale)), max(1, round(frame.shape[-1] * scale)))
    return F.interpolate(frame, size=size, mode="area")
