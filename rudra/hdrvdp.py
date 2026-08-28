"""Real HDR perceptual quality metric for RUDRA (review §4.1).

The paper reports an "HDR-VDP-3" column, but the codebase only shipped a
hand-rolled ``hdr_vdp_proxy`` (a JOD-like 0–10 number that does not correspond to
any published metric and whose scale doesn't match the paper's table). This
module replaces that with a *real* perceptual difference metric.

Reference HDR-VDP-3 proper is MATLAB-only. The recommended, pip-installable,
PyTorch path from the same lab (Mantiuk et al.) is **ColorVideoVDP** — the modern
successor that handles calibrated HDR content and reports the same JOD units
(10 = identical, lower = worse, may go below 0). We use it as the backend when
available and fall back to the clearly-labeled internal proxy otherwise.

Install the real metric:  ``pip install cvvdp``  (see github.com/gfxdisp/ColorVideoVDP)

Backend priority:
  1. ``colorvideovdp``  — pycvvdp.cvvdp with the ``standard_hdr_linear`` display
  2. ``proxy``          — the internal Naka-Rushton/CSF approximation (NOT HDR-VDP-3)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

_EPS = 1e-8

# Cache the (expensive to initialize) metric object per display/device.
_CVVDP_CACHE: dict = {}
_CVVDP_IMPORT_FAILED = False


# Linear RGB → XYZ (D65) for the working spaces, used to convert into the
# Rec.2020 linear space that the cvvdp HDR display model expects.
_RGB_TO_XYZ = {
    "rec709": torch.tensor(
        [[0.4123908, 0.3575843, 0.1804808],
         [0.2126390, 0.7151687, 0.0721923],
         [0.0193308, 0.1191948, 0.9505322]], dtype=torch.float64),
    "rec2020": torch.tensor(
        [[0.6369580, 0.1446169, 0.1688810],
         [0.2627002, 0.6779981, 0.0593017],
         [0.0000000, 0.0280727, 1.0609851]], dtype=torch.float64),
    "acescg": torch.tensor(
        [[0.6624542, 0.1340042, 0.1561877],
         [0.2722287, 0.6740818, 0.0536895],
         [-0.0055746, 0.0040607, 1.0103391]], dtype=torch.float64),
}


def colorvideovdp_available() -> bool:
    """True if the ColorVideoVDP backend can be imported."""
    global _CVVDP_IMPORT_FAILED
    if _CVVDP_IMPORT_FAILED:
        return False
    try:
        import pycvvdp  # noqa: F401
        return True
    except Exception:
        _CVVDP_IMPORT_FAILED = True
        return False


def _to_rec2020_linear(rgb: torch.Tensor, color_space: str) -> torch.Tensor:
    """Convert (B,3,H,W) linear RGB in *color_space* to Rec.2020 linear."""
    if color_space == "rec2020":
        return rgb
    src = _RGB_TO_XYZ.get(color_space)
    if src is None:
        return rgb
    M = (torch.linalg.inv(_RGB_TO_XYZ["rec2020"]) @ src).to(rgb.device, rgb.dtype)  # (3,3)
    B, C, H, W = rgb.shape
    flat = rgb.reshape(B, 3, -1)
    out = torch.einsum("ij,bjn->bin", M, flat).reshape(B, 3, H, W)
    return out.clamp(min=0.0)


# Why the real backend was last refused, so a caller can say something more
# useful than "install it" about a package that is already installed.
_CVVDP_LAST_ERROR: str | None = None


def cvvdp_last_error() -> str | None:
    """The reason the ColorVideoVDP backend fell back, if it has."""
    return _CVVDP_LAST_ERROR


def _get_cvvdp(device, display_name: str):
    """``device`` must be a torch.device.

    pycvvdp keeps whatever it is handed and later reads ``self.device.type``,
    so passing the string "cpu" raised AttributeError deep inside predict() --
    caught by the fallback below, reported as "proxy", and reported one level
    further up as "install torch + pycvvdp". CVVDP therefore never once ran,
    on a machine where it was installed, and the benchmark it gates was never
    produced. Hence the annotation, and cvvdp_last_error().
    """
    device = torch.device(device)
    key = f"{display_name}_{device}"
    if key not in _CVVDP_CACHE:
        import pycvvdp
        _CVVDP_CACHE[key] = pycvvdp.cvvdp(display_name=display_name, heatmap=None, device=device)
    return _CVVDP_CACHE[key]


def hdr_vdp3_jod(
    pred: torch.Tensor,
    target: torch.Tensor,
    color_space: str = "rec2020",
    diffuse_white_nits: float = 200.0,
    display_name: str = "standard_hdr_linear",
) -> Tuple[float, str]:
    """Perceptual HDR quality of ``pred`` vs ``target`` in JOD units.

    Args:
        pred, target: scene-linear RGB ``(B, 3, H, W)`` in ``color_space``.
        color_space:  working RGB space (rec2020 / rec709 / acescg).
        diffuse_white_nits: absolute luminance assigned to scene-linear 1.0.
            ColorVideoVDP's HDR display expects *absolute* cd/m² values; HDR
            highlights above 1.0 scale proportionally above this.
        display_name: cvvdp display model (``standard_hdr_linear`` for absolute
            linear input).

    Returns:
        (jod, backend) where ``jod`` ∈ ~[0, 10] (10 = identical) and ``backend``
        is ``"colorvideovdp"`` or ``"proxy"``.
    """
    global _CVVDP_LAST_ERROR
    if not colorvideovdp_available():
        _CVVDP_LAST_ERROR = "pycvvdp is not installed (pip install cvvdp)"
        from .metrics import hdr_vdp_proxy
        return hdr_vdp_proxy(pred, target), "proxy"

    try:
        device = torch.device(
            "cuda" if (isinstance(pred, torch.Tensor) and pred.is_cuda) else "cpu")
        metric = _get_cvvdp(device, display_name)

        with torch.no_grad():
            p = _to_rec2020_linear(pred.detach().float().clamp(min=0.0), color_space) * diffuse_white_nits
            t = _to_rec2020_linear(target.detach().float().clamp(min=0.0), color_space) * diffuse_white_nits

            jods = []
            for i in range(p.shape[0]):
                # cvvdp.predict expects HWC for a single image.
                test_hwc = p[i].permute(1, 2, 0).contiguous()
                ref_hwc = t[i].permute(1, 2, 0).contiguous()
                jod, _ = metric.predict(test_hwc, ref_hwc, dim_order="HWC")
                jods.append(float(jod))
        _CVVDP_LAST_ERROR = None
        return float(sum(jods) / max(len(jods), 1)), "colorvideovdp"
    except Exception as exc:
        # Any API/version mismatch falls back to the labeled proxy rather than
        # crashing a long training run -- but it no longer does so silently.
        _CVVDP_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        from .metrics import hdr_vdp_proxy
        return hdr_vdp_proxy(pred, target), "proxy"
