"""Export an SDR2HDRNet checkpoint as a native model package.

The native app (``native/``, see ``docs/NATIVE_ARCHITECTURE.md`` section 5.9)
never runs Python. It loads a *model package*: the same network twice, once as
TorchScript for LibTorch and once as ONNX for ONNX Runtime, plus a manifest
that says what the network is and golden frames that prove the two graphs
compute what eager PyTorch computes.

The package has two entry points, because that is how inference is already
split in ``training/infer_sdr2hdr.py::predict_fields``:

``frame_pass(sdr) -> (residual_scale, shadow_weight, curve_params)``
    Once per frame, on the WHOLE native frame. The residual and shadow gates
    read pooled features from a <=512 px area-downscaled view and statistics
    from the native pixels (``SDR2HDRNet.predict_residual_scale``). The curve
    head reads its own <=128 px view (``SDR2HDRNet.predict_curve``). A model
    without a head returns the value that head's absence means: scale 1,
    shadow weight 1, curve all zeros.

``tile_pass(sdr, residual_scale, curve_params) -> (residual, highlight, shadow)``
    Once per tile. The three fields ``predict_fields`` returns: the log
    residual with the per-frame scale folded in, and the two recovery masks.
    None of them depends on recovery mode, strength, shadow weight or
    preserve-outside; those enter the composite, which the app runs itself.

Why the orchestration is re-expressed here instead of exported directly:
``SDR2HDRNet.forward`` returns a dataclass and the curve head sizes its resize
with Python ``max``/``round``, neither of which TorchScript compiles, and
``F.interpolate(mode="area")`` with a non-integer factor, ``torch.quantile``
and ``median`` have no ONNX equivalent. Every trained submodule (stem, encoder,
decoder, head, gate MLPs, curve conv and MLP) is reused unchanged; only the
glue between them is written out below, twice where ONNX needs it
(``onnx_safe=True``). Nothing is trusted on that basis: ``verify`` runs every
exported graph against eager PyTorch and the package is not written unless
all of them agree within the stated tolerances.

Usage::

    python tools/export_model.py checkpoints/sdr2hdr_shadow_v1.pt
    python tools/export_model.py checkpoints/sdr2hdr_shadow_v1.pt --out dist/models \\
        --bench-dir D:/bench/sdr   # optional: parity on your own frames too

Writes ``<out>/<checkpoint stem>/``::

    model.ts            TorchScript: methods frame_pass, tile_pass
    model.frame.onnx    ONNX graph of frame_pass (dynamic H, W)
    model.tile.onnx     ONNX graph of tile_pass  (dynamic H, W)
    manifest.json       contract, source hash, versions, heads, tolerances, parity
    golden/             inputs and expected outputs as .npy, indexed by golden.json
    LICENSE             the weights licence
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.radiometry import ACES_A, ACES_B, ACES_C, ACES_D, ACES_E, LUMA_REC709  # noqa: E402
from rudra.sdr2hdr import SDR2HDRNet  # noqa: E402
from training.infer_sdr2hdr import predict_fields  # noqa: E402

# Bumped when the meaning of an input or output changes. The native app refuses
# a package whose major contract it does not know (P5 in NATIVE_ARCHITECTURE.md).
CONTRACT = "1.0"
GATE_MAX_SIDE = 512          # predict_residual_scale / predict_shadow_weight default
CURVE_MAX_SIDE = 128         # CurveHead.forward
QUANTILE_CEILING = 1_000_000  # frame_conditioning_stats subsampling threshold

# Acceptance per graph on CPU fp32, numpy.allclose semantics: every element
# must satisfy |exported - eager| <= atol + rtol * |eager|.
#
# LibTorch runs eager's kernels, so it is held to 1e-5 absolute (it measures
# 0.0). ONNX Runtime has its own convolution and GroupNorm kernels; on frames
# with large flat or clipped regions GroupNorm divides by a near-zero variance
# and amplifies the different summation order. Measured 23 Sep 2026: at most
# 5.9e-5 on the shipped model; on a model with every head randomised, 2.2e-4
# where the residual is 3.2 and 1.6e-4 where it is 0.036 (the edge of a
# synthetic clipped patch), all in the tile pass. atol is set at about twice
# that worst case. In the log domain 3e-4 is a 0.03% luminance difference,
# roughly thirty times below a 1% just-noticeable difference, while a real
# export mistake (a wrong op, a dropped branch) shows up at 1e-2 to 1.
#
# "gpu_fp32" is not checked here (the export runs on CPU); it is what
# rudra-native diff holds a GPU run of model.ts to. cuDNN picks its own
# convolution algorithms and summation order, so a GPU is never bit-exact with
# CPU even in true fp32 (TF32 off). Measured 23 Sep 2026 on an RTX 4080 SUPER:
# 1.4e-5 worst, in the residual. atol is set at a little over three times that,
# still six times tighter than ONNX.
TOLERANCE = {"torchscript": {"atol": 1e-5, "rtol": 0.0},
             "onnx": {"atol": 3e-4, "rtol": 1e-4},
             "gpu_fp32": {"atol": 5e-5, "rtol": 1e-5}}


# --------------------------------------------------------------------------
# The exported module
# --------------------------------------------------------------------------

class NativeModel(nn.Module):
    """``frame_pass`` and ``tile_pass`` over the submodules of one SDR2HDRNet.

    ``onnx_safe`` swaps three operations for exportable equivalents:
    ``F.interpolate(mode="area")`` becomes two averaging matmuls with the same
    bins ``adaptive_avg_pool2d`` uses, ``torch.quantile`` becomes a sort and a
    linear interpolation between neighbours, ``median`` becomes the lower
    middle element of a sort (torch's definition). The TorchScript graph keeps
    the originals, so LibTorch computes exactly what eager computes.
    """

    def __init__(self, net: SDR2HDRNet, onnx_safe: bool = False):
        super().__init__()
        self.onnx_safe = bool(onnx_safe)
        self.gate_max_side = int(GATE_MAX_SIDE)
        self.curve_max_side = int(CURVE_MAX_SIDE)
        self.corpus_ev = float(net.corpus_ev)
        self.log_scale = float(net.log_scale)
        self.max_hdr = float(net.max_hdr)
        self.aces = [float(ACES_A), float(ACES_B), float(ACES_C), float(ACES_D), float(ACES_E)]
        self.luma = [float(w) for w in LUMA_REC709]

        self.stem, self.enc1, self.down1 = net.stem, net.enc1, net.down1
        self.enc2, self.down2, self.mid = net.enc2, net.down2, net.mid
        self.up2, self.dec2, self.up1, self.dec1, self.head = (
            net.up2, net.dec2, net.up1, net.dec1, net.head)

        self.has_gate = net.gate is not None
        self.has_shadow = net.shadow_gate is not None
        self.has_curve = net.curve is not None
        # Submodules that do not exist are replaced by an empty Sequential so
        # the scripted class has one fixed set of attributes.
        self.gate_mlp = net.gate.mlp if self.has_gate else nn.Sequential()
        self.alpha_max = float(net.gate.alpha_max) if self.has_gate else 1.0
        self.shadow_mlp = net.shadow_gate.mlp if self.has_shadow else nn.Sequential()
        if self.has_curve:
            c = net.curve
            self.curve_conv, self.curve_mlp = c.conv, c.mlp
            self.knots, self.bins = int(c.knots), int(c.bins)
            self.max_exposure, self.max_knot = float(c.max_exposure), float(c.max_knot)
        else:
            self.curve_conv, self.curve_mlp = nn.Sequential(), nn.Sequential()
            self.knots, self.bins = 0, 2
            self.max_exposure, self.max_knot = 0.0, 0.0

    # -- analytic pieces (rudra/sdr2hdr.py, same maths) ---------------------

    def _luminance(self, x: torch.Tensor) -> torch.Tensor:
        return self.luma[0] * x[:, 0:1] + self.luma[1] * x[:, 1:2] + self.luma[2] * x[:, 2:3]

    def _analytic_baseline(self, sdr: torch.Tensor) -> torch.Tensor:
        x = sdr.clamp(0.0, 1.0)
        lin = torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))
        y = lin.clamp(0.0, 0.995)
        a, b, c, d, e = self.aces[0], self.aces[1], self.aces[2], self.aces[3], self.aces[4]
        qa = y * c - a
        qb = y * d - b
        qc = y * e
        disc = (qb * qb - 4.0 * qa * qc).clamp_min(0.0)
        denom = (2.0 * qa).clamp(max=-1e-7)
        root_a = (-qb - torch.sqrt(disc)) / denom
        root_b = (-qb + torch.sqrt(disc)) / denom
        scale = (2.0 ** (-self.corpus_ev)) * (203.0 / 10000.0)
        return torch.maximum(root_a, root_b).clamp_min(0.0) * scale

    def _curve_correction(self, sdr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        k = self.knots
        pos = sdr.clamp(0.0, 1.0) * float(k - 1)
        lo = pos.floor().clamp(max=float(k - 2)).long()
        frac = pos - lo.to(pos.dtype)
        knots = params[:, 1:]
        b = sdr.shape[0]
        flat_lo = lo.reshape(b, -1)
        v0 = torch.gather(knots, 1, flat_lo).reshape_as(sdr)
        v1 = torch.gather(knots, 1, flat_lo + 1).reshape_as(sdr)
        return params[:, :1, None, None] + v0 + (v1 - v0) * frac

    def _baseline(self, sdr: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        analytic = self._analytic_baseline(sdr)
        if not self.has_curve:
            return analytic
        return analytic * self._exp2(self._curve_correction(sdr, params))

    def _exp2(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return torch.exp2(x)
        else:
            if not self.onnx_safe:
                return torch.exp2(x)
            # The legacy ONNX exporter has no exp2.
            return torch.pow(torch.tensor(2.0, dtype=x.dtype), x)

    # -- resize, quantile, median --------------------------------------------

    def _target_size(self, h: int, w: int, max_side: int) -> List[int]:
        longest = h if h > w else w
        if longest <= max_side:
            return [h, w]
        s = float(max_side) / float(longest)
        # Python's round() is half-to-even and computed in float64; so is this.
        nh = int(torch.round(torch.tensor(float(h) * s, dtype=torch.float64)).item())
        nw = int(torch.round(torch.tensor(float(w) * s, dtype=torch.float64)).item())
        return [max(1, nh), max(1, nw)]

    @torch.jit.ignore
    def _area_matrix(self, n_in: torch.Tensor, n_out: torch.Tensor,
                     dtype: torch.dtype) -> torch.Tensor:
        # adaptive_avg_pool bins: [floor(i*in/out), ceil((i+1)*in/out))
        i = torch.arange(0, n_out, dtype=torch.float64)
        start = torch.floor(i * n_in.double() / n_out.double())
        end = torch.ceil((i + 1.0) * n_in.double() / n_out.double())
        j = torch.arange(0, n_in, dtype=torch.float64)
        inside = (j[None, :] >= start[:, None]) & (j[None, :] < end[:, None])
        m = inside.double() / (end - start)[:, None]
        return m.to(dtype)

    def _area_view(self, x: torch.Tensor, max_side: int) -> torch.Tensor:
        if torch.jit.is_scripting():
            return self._area_exact(x, max_side)
        else:
            if not self.onnx_safe:
                return self._area_exact(x, max_side)
            return self._area_onnx(x, max_side)

    def _area_exact(self, x: torch.Tensor, max_side: int) -> torch.Tensor:
        size = self._target_size(int(x.shape[-2]), int(x.shape[-1]), max_side)
        if size[0] == int(x.shape[-2]) and size[1] == int(x.shape[-1]):
            return x
        return F.interpolate(x, size=size, mode="area")

    @torch.jit.ignore
    def _area_onnx(self, x: torch.Tensor, max_side: int) -> torch.Tensor:
        # Tensor arithmetic only, so the graph stays dynamic in H and W.
        shape = torch._shape_as_tensor(x)
        h, w = shape[2], shape[3]
        longest = torch.maximum(h, w).double()
        s = torch.clamp(float(max_side) / longest, max=1.0)
        nh = torch.clamp(torch.round(h.double() * s), min=1.0).long()
        nw = torch.clamp(torch.round(w.double() * s), min=1.0).long()
        mh = self._area_matrix(h, nh, x.dtype)          # (nh, h)
        mw = self._area_matrix(w, nw, x.dtype)          # (nw, w)
        return torch.matmul(torch.matmul(mh, x), mw.transpose(0, 1))

    def _quantile(self, flat: torch.Tensor, q: float) -> torch.Tensor:
        if torch.jit.is_scripting():
            return torch.quantile(flat, q, dim=1)
        else:
            if not self.onnx_safe:
                return torch.quantile(flat, q, dim=1)
            return self._quantile_onnx(flat, q)

    @torch.jit.ignore
    def _quantile_onnx(self, flat: torch.Tensor, q: float) -> torch.Tensor:
        srt = torch.sort(flat, dim=1)[0]
        rank = torch.tensor(q, dtype=torch.float64) * (torch._shape_as_tensor(flat)[1].double() - 1.0)
        below = torch.floor(rank)
        weight = (rank - below).to(flat.dtype)
        lo = below.long()
        hi = torch.clamp(lo + 1, max=torch._shape_as_tensor(flat)[1] - 1)
        v_lo = srt.index_select(1, lo.reshape(1)).reshape(-1)
        v_hi = srt.index_select(1, hi.reshape(1)).reshape(-1)
        return v_lo + weight * (v_hi - v_lo)

    def _median(self, flat: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return flat.median(dim=1).values
        else:
            if not self.onnx_safe:
                return flat.median(dim=1).values
            return self._median_onnx(flat)

    @torch.jit.ignore
    def _median_onnx(self, flat: torch.Tensor) -> torch.Tensor:
        srt = torch.sort(flat, dim=1)[0]
        mid = torch.div(torch._shape_as_tensor(flat)[1] - 1, 2, rounding_mode="floor")
        return srt.index_select(1, mid.reshape(1)).reshape(-1)

    def _subsample(self, flat: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return self._subsample_exact(flat)
        else:
            if not self.onnx_safe:
                return self._subsample_exact(flat)
            return self._subsample_onnx(flat)

    def _subsample_exact(self, flat: torch.Tensor) -> torch.Tensor:
        if flat.shape[1] > 1000000:
            flat = flat[:, :: flat.shape[1] // 1000000 + 1]
        return flat

    @torch.jit.ignore
    def _subsample_onnx(self, flat: torch.Tensor) -> torch.Tensor:
        n = torch._shape_as_tensor(flat)[1]
        big = (n > QUANTILE_CEILING).long()
        step = big * torch.div(n, QUANTILE_CEILING, rounding_mode="floor") + 1
        idx = torch.arange(0, n, step)
        return flat.index_select(1, idx)

    # -- heads ---------------------------------------------------------------

    def _frame_stats(self, sdr: torch.Tensor, sdr_y: torch.Tensor) -> torch.Tensor:
        hi = torch.sigmoid((sdr_y - 0.98) * 200.0).mean(dim=(1, 2, 3))
        lo = torch.sigmoid((0.02 - sdr_y) * 200.0).mean(dim=(1, 2, 3))
        mean = sdr_y.mean(dim=(1, 2, 3))
        std = sdr_y.flatten(1).std(dim=1)
        lap = (sdr_y[:, :, 1:-1, 1:-1] * 4.0
               - sdr_y[:, :, :-2, 1:-1] - sdr_y[:, :, 2:, 1:-1]
               - sdr_y[:, :, 1:-1, :-2] - sdr_y[:, :, 1:-1, 2:])
        hf = lap.abs().mean(dim=(1, 2, 3))
        chroma = sdr - sdr_y
        chf = (chroma[..., 1:] - chroma[..., :-1]).abs().mean(dim=(1, 2, 3))
        flat = self._subsample(sdr_y.flatten(1))
        span = self._quantile(flat, 0.99) - self._quantile(flat, 0.50)
        return torch.stack([hi, lo, mean, std, hf, chf, span], dim=1)

    def _curve_params(self, sdr: torch.Tensor) -> torch.Tensor:
        view = self._area_view(sdr, self.curve_max_side)
        pooled = self.curve_conv(view).mean(dim=(2, 3))
        y = self._luminance(view).flatten(1)
        centres = torch.linspace(0.0, 1.0, self.bins, dtype=sdr.dtype)
        width = 1.0 / float(self.bins - 1)
        hist = torch.exp(-((y[:, :, None] - centres) / width) ** 2).mean(dim=1)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(1e-6)
        stats = torch.stack((
            y.mean(1), y.std(1),
            torch.sigmoid((y - 0.98) * 200.0).mean(1),
            torch.sigmoid((0.02 - y) * 200.0).mean(1),
            view.amax(dim=(1, 2, 3)), self._median(view.flatten(1)),
        ), dim=1)
        raw = self.curve_mlp(torch.cat((pooled, hist, stats), dim=1))
        exposure = torch.tanh(raw[:, :1]) * self.max_exposure
        knots = torch.tanh(raw[:, 1:]) * self.max_knot
        return torch.cat((exposure, knots), dim=1)

    def _encode(self, sdr: torch.Tensor, baseline: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(self.stem(torch.cat((sdr, baseline), dim=1)))
        e2 = self.enc2(self.down1(e1))
        return e1, e2, self.mid(self.down2(e2))

    def _resize_like(self, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return F.interpolate(x, size=[int(like.shape[-2]), int(like.shape[-1])],
                                 mode="bilinear", align_corners=False)
        else:
            # Shape taken from the tensor, so a traced ONNX graph stays dynamic.
            return F.interpolate(x, size=like.shape[-2:], mode="bilinear", align_corners=False)

    # -- entry points --------------------------------------------------------

    @torch.jit.export
    def frame_pass(self, sdr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(residual_scale (1,1,1,1), shadow_weight (1,1,1,1), curve_params (1,P))."""
        sdr = sdr.float().clamp(0.0, 1.0)
        one = torch.ones((1, 1, 1, 1), dtype=sdr.dtype)
        if self.has_curve:
            curve = self._curve_params(sdr)
        else:
            curve = torch.zeros((1, 1), dtype=sdr.dtype)
        scale = one
        weight = one
        if self.has_gate or self.has_shadow:
            view = self._area_view(sdr, self.gate_max_side)
            # The gates' encode uses the baseline of the VIEW, whose curve is
            # estimated from the view (SDR2HDRNet.baseline_hdr(view) with no
            # params), not the frame's curve. Reproduced as written.
            view_params = self._curve_params(view) if self.has_curve else curve
            _, _, m = self._encode(view, self._baseline(view, view_params))
            pooled = torch.cat((m.mean(dim=(2, 3)), m.amax(dim=(2, 3))), dim=1)
            stats = self._frame_stats(sdr, self._luminance(sdr)).to(pooled.dtype)
            features = torch.cat((pooled, stats), dim=1)
            if self.has_gate:
                scale = (torch.sigmoid(self.gate_mlp(features)) * self.alpha_max).view(-1, 1, 1, 1)
            if self.has_shadow:
                weight = torch.sigmoid(self.shadow_mlp(features)).view(-1, 1, 1, 1)
        return scale, weight, curve

    @torch.jit.export
    def tile_pass(self, sdr: torch.Tensor, residual_scale: torch.Tensor,
                  curve_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log residual x scale (1,3,h,w), highlight (1,1,h,w), shadow (1,1,h,w))."""
        sdr = sdr.float().clamp(0.0, 1.0)
        baseline = self._baseline(sdr, curve_params)
        e1, e2, m = self._encode(sdr, baseline)
        u2 = self._resize_like(m, e2)
        u2 = self.dec2(self.up2(torch.cat((u2, e2), dim=1)))
        u1 = self._resize_like(u2, e1)
        raw = self.head(self.dec1(self.up1(torch.cat((u1, e1), dim=1))))
        residual = raw[:, 0:3]
        if self.has_gate:
            residual = residual * residual_scale
        return residual, torch.sigmoid(raw[:, 3:4]), torch.sigmoid(raw[:, 4:5])

    def forward(self, sdr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, _, curve = self.frame_pass(sdr)
        return self.tile_pass(sdr, scale, curve)


class _FrameOnly(nn.Module):
    def __init__(self, m: NativeModel):
        super().__init__()
        self.m = m

    def forward(self, sdr):
        return self.m.frame_pass(sdr)


class _TileOnly(nn.Module):
    def __init__(self, m: NativeModel):
        super().__init__()
        self.m = m

    def forward(self, sdr, residual_scale, curve_params):
        return self.m.tile_pass(sdr, residual_scale, curve_params)


# --------------------------------------------------------------------------
# Loading, reference, export
# --------------------------------------------------------------------------

def load_network(checkpoint: Path) -> tuple[SDR2HDRNet, dict]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ck.get("config", {}) if isinstance(ck, dict) else {}
    net = SDR2HDRNet.from_config(config)
    net.load_state_dict(ck.get("model", ck), strict=True)
    return net.eval(), config


@torch.inference_mode()
def eager_reference(net: SDR2HDRNet, sdr: torch.Tensor) -> dict[str, np.ndarray]:
    """What the Python path computes today for one frame, untiled."""
    scale = net.predict_residual_scale(sdr)
    weight = net.predict_shadow_weight(sdr)
    curve = net.predict_curve(sdr)
    fields = predict_fields(net, sdr, tile_size=0, overlap=0)
    one = torch.ones((1, 1, 1, 1))
    return {
        "residual_scale": (scale if scale is not None else one).float().numpy(),
        "shadow_weight": (weight if weight is not None else one).float().numpy(),
        "curve_params": (curve if curve is not None else torch.zeros((1, 1))).float().numpy(),
        "residual": fields["residual"].float().numpy(),
        "highlight": fields["highlight"].float().numpy(),
        "shadow": fields["shadow"].float().numpy(),
    }


def script_model(net: SDR2HDRNet) -> torch.jit.ScriptModule:
    return torch.jit.script(NativeModel(net, onnx_safe=False).eval())


def export_onnx(net: SDR2HDRNet, frame_path: Path, tile_path: Path, opset: int) -> None:
    m = NativeModel(net, onnx_safe=True).eval()
    sample = torch.rand(1, 3, 96, 128)
    with torch.inference_mode(False), torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            _FrameOnly(m), (sample,), str(frame_path), dynamo=False, opset_version=opset,
            input_names=["sdr"], output_names=["residual_scale", "shadow_weight", "curve_params"],
            dynamic_axes={"sdr": {2: "height", 3: "width"}})
        scale, _, curve = m.frame_pass(sample)
        torch.onnx.export(
            _TileOnly(m), (sample, scale, curve), str(tile_path), dynamo=False, opset_version=opset,
            input_names=["sdr", "residual_scale", "curve_params"],
            output_names=["residual", "highlight", "shadow"],
            dynamic_axes={"sdr": {2: "height", 3: "width"},
                          "residual": {2: "height", 3: "width"},
                          "highlight": {2: "height", 3: "width"},
                          "shadow": {2: "height", 3: "width"}})


# --------------------------------------------------------------------------
# Golden frames
# --------------------------------------------------------------------------

def golden_inputs() -> list[tuple[str, np.ndarray]]:
    """16 SDR frames (H, W, 3, float32 in [0,1]): native crops of a real image
    plus synthetic frames aimed at the edges the native code has to get right.
    Small on purpose: they ship inside every model package."""
    frames: list[tuple[str, np.ndarray]] = []
    sunset = REPO / "ui" / "assets" / "cinematic_hdr_sunset.png"
    import cv2
    img = cv2.imread(str(sunset), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"missing {sunset}")
    rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    crops = [("sun_clipped", 380, 560, 128, 128), ("sky", 60, 300, 128, 160),
             ("mountain", 400, 380, 96, 128), ("water_reflection", 700, 420, 128, 128),
             ("forest_shadow", 600, 900, 128, 96), ("rocks_crushed", 880, 700, 112, 144),
             ("horizon_wide", 520, 200, 64, 256), ("full_downscaled", 0, 0, 0, 0)]
    for name, y, x, h, w in crops:
        if h == 0:
            crop = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            crop = rgb[y:y + h, x:x + w]
        frames.append((name, np.ascontiguousarray(crop, dtype=np.float32)))

    rng = np.random.default_rng(20260923)
    gy, gx = np.mgrid[0:96, 0:128].astype(np.float32)
    ramp = np.repeat((gx / 127.0)[..., None], 3, axis=2)
    frames.append(("ramp_h", ramp))
    patch = np.full((96, 128, 3), 0.35, np.float32)
    patch[20:60, 40:100] = 1.0
    frames.append(("clipped_patch", patch))
    crushed = np.clip(rng.normal(0.02, 0.015, (96, 128, 3)), 0, 1).astype(np.float32)
    frames.append(("crushed_noise", crushed))
    frames.append(("flat_grey", np.full((64, 64, 3), 0.5, np.float32)))
    frames.append(("noise", rng.random((80, 120, 3), dtype=np.float32)))
    odd = rng.random((61, 97, 3), dtype=np.float32) ** 2.2
    frames.append(("odd_97x61", odd.astype(np.float32)))
    portrait = np.clip(np.stack([gy[:, :64] / 95.0] * 3, axis=2), 0, 1)
    frames.append(("portrait_64x96", np.ascontiguousarray(portrait, dtype=np.float32)))
    sat = np.zeros((72, 72, 3), np.float32)
    sat[..., 0] = 1.0
    sat[24:48, 24:48] = [0.0, 1.0, 0.2]
    frames.append(("saturated", sat))
    return frames


def _to_tensor(hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(hwc.transpose(2, 0, 1)))[None].float()


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

OUTPUTS = ("residual_scale", "shadow_weight", "curve_params", "residual", "highlight", "shadow")


def run_torchscript(ts: torch.jit.ScriptModule, sdr: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        scale, weight, curve = ts.frame_pass(sdr)
        r, h, s = ts.tile_pass(sdr, scale, curve)
    return dict(zip(OUTPUTS, (t.float().numpy() for t in (scale, weight, curve, r, h, s))))


def run_onnx(frame_sess, tile_sess, sdr: torch.Tensor) -> dict[str, np.ndarray]:
    x = sdr.numpy()
    f_inputs = {i.name for i in frame_sess.get_inputs()}
    scale, weight, curve = frame_sess.run(None, {"sdr": x} if "sdr" in f_inputs else {})
    feed = {"sdr": x, "residual_scale": scale, "curve_params": curve}
    t_inputs = {i.name for i in tile_sess.get_inputs()}
    r, h, s = tile_sess.run(None, {k: v for k, v in feed.items() if k in t_inputs})
    return dict(zip(OUTPUTS, (scale, weight, curve, r, h, s)))


def compare(ref: dict, got: dict) -> dict[str, float]:
    return {k: _max_abs(ref[k], got[k]) for k in OUTPUTS}


def excess(ref: dict, got: dict, tol: dict) -> float:
    """Largest |d| - (atol + rtol*|ref|) over every output; <= 0 passes."""
    worst = -float("inf")
    for k in OUTPUTS:
        a = ref[k].astype(np.float64)
        b = got[k].astype(np.float64)
        if a.size:
            worst = max(worst, float(np.max(np.abs(a - b) - (tol["atol"] + tol["rtol"] * np.abs(a)))))
    return worst


def bench_frames(folder: Path, limit: int) -> list[tuple[str, np.ndarray]]:
    import cv2
    out = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb /= float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
        out.append((p.stem, rgb))
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# Package
# --------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _save_npy(folder: Path, name: str, arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    path = folder / f"{name}.npy"
    np.save(path, arr, allow_pickle=False)
    return {"file": path.name, "shape": list(arr.shape), "dtype": "float32"}


def build_package(checkpoint: Path, out_root: Path, opset: int = 17,
                  bench_dir: Path | None = None, bench_limit: int = 0,
                  tile_golden: bool = True) -> dict:
    import onnxruntime as ort

    torch.set_num_threads(max(1, torch.get_num_threads()))
    net, config = load_network(checkpoint)
    name = checkpoint.stem
    out = out_root / name
    tmp = out_root / f".{name}.partial"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "golden").mkdir(parents=True)

    ts = script_model(net)
    ts.save(str(tmp / "model.ts"))
    ts = torch.jit.load(str(tmp / "model.ts"))      # verify what was written, not what is in memory
    export_onnx(net, tmp / "model.frame.onnx", tmp / "model.tile.onnx", opset)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    f_sess = ort.InferenceSession(str(tmp / "model.frame.onnx"), so, providers=["CPUExecutionProvider"])
    t_sess = ort.InferenceSession(str(tmp / "model.tile.onnx"), so, providers=["CPUExecutionProvider"])

    worst = {"torchscript": {k: 0.0 for k in OUTPUTS}, "onnx": {k: 0.0 for k in OUTPUTS}}
    margin = {"torchscript": -float("inf"), "onnx": -float("inf")}
    index = []
    frames = golden_inputs()
    for name_i, hwc in frames:
        sdr = _to_tensor(hwc)
        ref = eager_reference(net, sdr)
        for label, got in (("torchscript", run_torchscript(ts, sdr)),
                           ("onnx", run_onnx(f_sess, t_sess, sdr))):
            for k, v in compare(ref, got).items():
                worst[label][k] = max(worst[label][k], v)
            margin[label] = max(margin[label], excess(ref, got, TOLERANCE[label]))
        entry = {"name": name_i, "sdr": _save_npy(tmp / "golden", f"{name_i}.sdr", hwc)}
        for k in OUTPUTS:
            entry[k] = _save_npy(tmp / "golden", f"{name_i}.{k}", ref[k])
        index.append(entry)

    stitch = None
    if tile_golden:
        # The stitching reference for the native tiler: predict_fields with
        # small tiles on a frame that needs a 3x2 grid. Same algorithm at any
        # tile size, so the golden stays small.
        _, hwc = frames[1]
        big = np.ascontiguousarray(np.concatenate([hwc, hwc[:, ::-1]], axis=1)[:, :300], dtype=np.float32)
        sdr = _to_tensor(big)
        with torch.inference_mode():
            f = predict_fields(net, sdr, tile_size=128, overlap=32)
        stitch = {"name": "stitch_300x128_t128_o32", "tile_size": 128, "overlap": 32,
                  "sdr": _save_npy(tmp / "golden", "stitch.sdr", big),
                  "residual": _save_npy(tmp / "golden", "stitch.residual", f["residual"].numpy()),
                  "highlight": _save_npy(tmp / "golden", "stitch.highlight", f["highlight"].numpy()),
                  "shadow": _save_npy(tmp / "golden", "stitch.shadow", f["shadow"].numpy())}

    bench = None
    if bench_dir is not None:
        bw = {"torchscript": 0.0, "onnx": 0.0}
        count = 0
        for _, hwc in bench_frames(bench_dir, bench_limit):
            sdr = _to_tensor(hwc)
            ref = eager_reference(net, sdr)
            for label, got in (("torchscript", run_torchscript(ts, sdr)),
                               ("onnx", run_onnx(f_sess, t_sess, sdr))):
                bw[label] = max(bw[label], max(compare(ref, got).values()))
                margin[label] = max(margin[label], excess(ref, got, TOLERANCE[label]))
            count += 1
        bench = {"folder": str(bench_dir), "frames": count, "max_abs": bw}

    passed = {label: margin[label] <= 0.0 for label in margin}

    lic = REPO / "checkpoints" / "LICENSE"
    if lic.exists():
        shutil.copy2(lic, tmp / "LICENSE")
    (tmp / "golden" / "golden.json").write_text(json.dumps(
        {"frames": index, "stitch": stitch, "layout": "sdr is (H,W,3); outputs are NCHW"}, indent=2), encoding="utf-8", newline="\n")

    import onnx
    manifest = {
        "contract": CONTRACT,
        "name": name,
        "source": {"file": checkpoint.name, "sha256": sha256(checkpoint)},
        "exported": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__,
                     "onnxruntime": ort.__version__, "opset": opset},
        "network": {
            "base_channels": int(config.get("base_channels", 32)),
            "corpus_ev": float(net.corpus_ev), "log_scale": float(net.log_scale),
            "max_hdr": float(net.max_hdr),
            "heads": {"residual_gate": net.gate is not None,
                      "shadow_gate": net.shadow_gate is not None,
                      "curve": net.curve is not None},
            "curve_params": int(1 + net.curve.knots) if net.curve is not None else 1,
            "gate_max_side": GATE_MAX_SIDE, "curve_max_side": CURVE_MAX_SIDE,
        },
        "input": {"layout": "NCHW", "batch": 1, "channels": 3, "dtype": "float32",
                  "encoding": "sRGB display-referred code values in [0,1], canonicalised"
                              " as rudra.sdr2hdr.canonicalize_sdr does"},
        "tiling": {"tile_size": 512, "overlap": 64,
                   "weights": "training/infer_sdr2hdr.py::_tile_weight"},
        "files": {"torchscript": "model.ts", "onnx_frame": "model.frame.onnx",
                  "onnx_tile": "model.tile.onnx", "golden": "golden/golden.json"},
        "onnx_inputs": {"frame": [i.name for i in f_sess.get_inputs()],
                        "tile": [i.name for i in t_sess.get_inputs()]},
        "tolerance": {"rule": "|exported - eager| <= atol + rtol*|eager|, every element", **TOLERANCE},
        "parity": {"golden_frames": len(frames), "max_abs": worst,
                   "worst_excess_over_tolerance": margin, "bench": bench, "passed": passed},
    }
    for key in ("torchscript", "onnx_frame", "onnx_tile"):
        manifest["files"][key + "_sha256"] = sha256(tmp / manifest["files"][key])
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")

    if not all(passed.values()):
        raise SystemExit(
            "parity FAILED, package not written (partial output kept in "
            f"{tmp}):\n{json.dumps({'max_abs': worst, 'excess': margin, 'bench': bench}, indent=2)}")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "models")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--bench-dir", type=Path, default=None,
                    help="folder of SDR frames to add to the parity check (e.g. the 429 bench frames)")
    ap.add_argument("--bench-limit", type=int, default=0)
    args = ap.parse_args(argv)
    manifest = build_package(args.checkpoint, args.out, args.opset, args.bench_dir, args.bench_limit)
    p = manifest["parity"]
    print(f"{manifest['name']}: heads {manifest['network']['heads']}")
    for label in ("torchscript", "onnx"):
        t = TOLERANCE[label]
        print(f"  {label:11s} max |d| {max(p['max_abs'][label].values()):.2e}"
              f"  (atol {t['atol']:.0e}, rtol {t['rtol']:.0e})  {'PASS' if p['passed'][label] else 'FAIL'}")
    if p["bench"]:
        print(f"  bench: {p['bench']['frames']} frames, {p['bench']['max_abs']}")
    print(f"  written to {args.out / manifest['name']}")
    # The registry beside the packages: the app's checkpoint manager reads it
    # as the server reads checkpoints/models.json (titles, notes, the default).
    registry = REPO / "checkpoints" / "models.json"
    if registry.is_file() and not (args.out / "models.json").exists():
        shutil.copyfile(registry, args.out / "models.json")
        print(f"  registry copied to {args.out / 'models.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
