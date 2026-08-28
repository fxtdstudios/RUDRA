"""Artist grade controls for RUDRA HDR output (torch-free).

The control surface Hyperion refuses to expose and SwitchHDR charges for:
deterministic, order-documented operations on linear radiance — global
exposure, per-region EV via soft masks, luminance qualifiers (build a mask
from a stop range, like a Resolve qualifier), highlight desaturation, and a
hue-preserving shoulder to any mastering peak. Everything is pure numpy on
absolute nits, so a grade replays bit-exactly on any machine and can be
recorded in the delivery sidecar.

Operation order (fixed, documented, and the reason results are reproducible):
  1. global exposure_ev
  2. per-region EV adjustments (soft-mask weighted, applied in linear light)
  3. highlight desaturation toward the mastering white
  4. hue-preserving shoulder to peak_nits (knee_nits onward)

The ``itm_strength_map`` helper turns the same masks into the per-pixel
residual-strength tensor understood by ``SDR2HDRNet.forward`` — one mask
language for both the network's recovery aggressiveness and the final grade.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..hdr10 import REC2020_LUMA

__all__ = [
    "RegionEV", "GradeControls", "qualifier_mask", "apply_grade",
    "apply_region_ev", "region_ev_gain", "DEFAULT_REGION_BANDS",
    "itm_strength_map",
]

# The three bands RUDRA Studio's Region EV panel opens with. Neutral by
# default: a tool must not grade a frame nobody asked it to grade.
DEFAULT_REGION_BANDS = (
    {"label": "highlights", "low_nits": 400.0, "high_nits": 2000.0, "ev": 0.0},
    {"label": "speculars", "low_nits": 2000.0, "high_nits": 8000.0, "ev": 0.0},
    {"label": "shadows", "low_nits": 0.05, "high_nits": 12.0, "ev": 0.0},
)


@dataclass
class RegionEV:
    """A soft mask (H, W) in [0, 1] and the EV push applied where it is 1."""
    mask: np.ndarray
    ev: float
    label: str = ""


@dataclass
class GradeControls:
    exposure_ev: float = 0.0
    peak_nits: float = 1000.0
    knee_nits: float | None = None           # default: 0.75 * peak
    highlight_desat: float = 0.0             # 0..1, applied above the knee
    regions: list[RegionEV] = field(default_factory=list)

    def describe(self) -> dict:
        """JSON-serializable record of the grade for the delivery sidecar."""
        return {
            "exposure_ev": self.exposure_ev,
            "peak_nits": self.peak_nits,
            "knee_nits": self.knee_nits,
            "highlight_desat": self.highlight_desat,
            "regions": [
                {"label": r.label, "ev": r.ev, "mask_mean": float(np.mean(r.mask))}
                for r in self.regions
            ],
            "operation_order": [
                "exposure_ev", "region_ev", "highlight_desat", "shoulder_to_peak",
            ],
        }


def qualifier_mask(
    rgb_nits: np.ndarray,
    low_nits: float,
    high_nits: float,
    softness_stops: float = 1.0,
) -> np.ndarray:
    """Soft luminance qualifier: 1 inside [low, high] nits, feathered in stops.

    The feather is symmetric in log2 space, so "0.5 stops of softness" means
    the same visual transition width in shadows and highlights.
    """
    if not 0.0 < low_nits < high_nits:
        raise ValueError("need 0 < low_nits < high_nits")
    y = np.maximum(np.sum(np.asarray(rgb_nits, dtype=np.float64) * REC2020_LUMA, axis=-1), 1e-6)
    log_y = np.log2(y)
    soft = max(float(softness_stops), 1e-3)
    rise = np.clip((log_y - (np.log2(low_nits) - soft)) / soft, 0.0, 1.0)
    fall = np.clip(((np.log2(high_nits) + soft) - log_y) / soft, 0.0, 1.0)
    mask = np.minimum(rise, fall)
    # smoothstep for C1-continuous feather edges
    return (mask * mask * (3.0 - 2.0 * mask)).astype(np.float32)


def region_ev_gain(rgb_nits: np.ndarray, bands, softness_stops: float = 1.0) -> np.ndarray:
    """Per-pixel linear gain (H, W, 1) from a set of luminance-qualified EV bands.

    Step 2 of ``apply_grade`` on its own, split out because two other things
    need exactly this number and nothing else in the grade: RUDRA Studio's
    shader, which applies it on the viewer's GPU so the Region EV panel
    responds at frame rate, and the EXR writer, which has to reproduce it
    bit-for-bit or the master would not be the picture that was approved.

    The offsets add in stops, so overlapping bands compose the way a colourist
    expects rather than the way the loop happens to be ordered.
    """
    rgb = np.maximum(np.asarray(rgb_nits, dtype=np.float64), 0.0)
    total = np.zeros(rgb.shape[:-1], dtype=np.float64)
    for band in bands or ():
        ev = float(band.get("ev", 0.0))
        if ev == 0.0:
            continue
        total = total + ev * qualifier_mask(rgb, float(band["low_nits"]),
                                            float(band["high_nits"]), softness_stops)
    return np.exp2(total)[..., None]


def apply_region_ev(rgb_nits: np.ndarray, bands, softness_stops: float = 1.0) -> np.ndarray:
    """Apply ``region_ev_gain`` to a frame in absolute nits."""
    return np.asarray(rgb_nits, dtype=np.float64) * region_ev_gain(
        rgb_nits, bands, softness_stops)


def _shoulder_to_peak(rgb_nits: np.ndarray, peak_nits: float, knee_nits: float | None) -> np.ndarray:
    """Hue-preserving exponential shoulder (same math as rudra.hdr10)."""
    knee = 0.75 * peak_nits if knee_nits is None else float(knee_nits)
    if not 0.0 <= knee < peak_nits:
        raise ValueError("knee_nits must be >= 0 and below peak_nits")
    rgb = np.maximum(np.asarray(rgb_nits, dtype=np.float64), 0.0)
    lum = np.sum(rgb * REC2020_LUMA, axis=-1, keepdims=True)
    span = peak_nits - knee
    compressed = np.where(lum <= knee, lum, knee + span * (1.0 - np.exp(-(lum - knee) / span)))
    rgb = rgb * compressed / np.maximum(lum, 1e-6)
    component_max = np.max(rgb, axis=-1, keepdims=True)
    rgb = rgb * np.minimum(1.0, peak_nits / np.maximum(component_max, 1e-6))
    return np.clip(rgb, 0.0, peak_nits)


def apply_grade(rgb_nits: np.ndarray, controls: GradeControls) -> np.ndarray:
    """Apply the full grade to an (H, W, 3) linear frame in absolute nits."""
    rgb = np.maximum(np.asarray(rgb_nits, dtype=np.float64), 0.0)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {rgb.shape}")
    if not 100.0 <= controls.peak_nits <= 10000.0:
        raise ValueError("peak_nits must be between 100 and 10000")

    if controls.exposure_ev:
        rgb = rgb * (2.0 ** controls.exposure_ev)

    for region in controls.regions:
        mask = np.asarray(region.mask, dtype=np.float64)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"region {region.label!r} mask {mask.shape} != frame {rgb.shape[:2]}")
        rgb = rgb * (2.0 ** (region.ev * np.clip(mask, 0.0, 1.0)))[..., None]

    if controls.highlight_desat > 0.0:
        knee = controls.knee_nits if controls.knee_nits is not None else 0.75 * controls.peak_nits
        lum = np.sum(rgb * REC2020_LUMA, axis=-1, keepdims=True)
        above = np.clip((lum - knee) / max(controls.peak_nits - knee, 1e-6), 0.0, 1.0)
        amount = np.clip(controls.highlight_desat, 0.0, 1.0) * above
        rgb = rgb + amount * (lum - rgb)

    return _shoulder_to_peak(rgb, controls.peak_nits, controls.knee_nits).astype(np.float32)


def itm_strength_map(
    shape_hw: tuple[int, int],
    base_strength: float = 1.0,
    regions: list[RegionEV] | None = None,
) -> np.ndarray:
    """Per-pixel inverse-tone-map strength for ``SDR2HDRNet``.

    Returns an (1, 1, H, W) float32 array: ``base_strength`` everywhere,
    with each region's mask pushing strength by ``2**ev`` (ev=-inf..+1 makes
    sense here; values are clamped to [0, 2]). Pass as ``residual_strength``
    to localize how aggressively the network reconstructs highlights —
    the mask-driven local ITM no competitor exposes.
    """
    height, width = shape_hw
    strength = np.full((height, width), float(base_strength), dtype=np.float64)
    for region in regions or []:
        mask = np.clip(np.asarray(region.mask, dtype=np.float64), 0.0, 1.0)
        if mask.shape != (height, width):
            raise ValueError(f"region {region.label!r} mask {mask.shape} != {shape_hw}")
        strength = strength * (2.0 ** (region.ev * mask))
    return np.clip(strength, 0.0, 2.0).astype(np.float32)[None, None]
