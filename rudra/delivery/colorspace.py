"""RGB primary conversions for the RUDRA delivery layer (torch-free).

Matrices are derived from primaries + white points at runtime (no hardcoded
3x3 tables that can silently drift), with Bradford chromatic adaptation
between differing white points. This is what lets RUDRA hand a colorist an
ACES 2065-1 (AP0, ~D60) master from its Rec.2020 (D65) working space with
exact, auditable math.

Conventions:
  - All RGB here is LINEAR (no transfer function).
  - Exposure is untouched: RUDRA's diffuse-white-relative scene linear
    (18% grey = 0.18, diffuse white = 1.0) shares ACES's 0.18 mid-grey
    anchor, so Rec.2020 -> AP0 is a matrix, not an exposure change.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PRIMARIES", "WHITE_POINTS", "npm", "bradford_cat",
    "rgb_to_rgb_matrix", "convert", "AP0_CHROMATICITIES",
]

# (rx, ry, gx, gy, bx, by)
PRIMARIES: dict[str, tuple[float, float, float, float, float, float]] = {
    "rec709":  (0.640, 0.330, 0.300, 0.600, 0.150, 0.060),
    "rec2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046),
    "p3d65":   (0.680, 0.320, 0.265, 0.690, 0.150, 0.060),
    "ap0":     (0.7347, 0.2653, 0.0000, 1.0000, 0.0001, -0.0770),
    "ap1":     (0.7130, 0.2930, 0.1650, 0.8300, 0.1280, 0.0440),
}

# (wx, wy)
WHITE_POINTS: dict[str, tuple[float, float]] = {
    "rec709":  (0.3127, 0.3290),   # D65
    "rec2020": (0.3127, 0.3290),   # D65
    "p3d65":   (0.3127, 0.3290),   # D65
    "ap0":     (0.32168, 0.33767),  # ACES white (~D60)
    "ap1":     (0.32168, 0.33767),
}

# ST 2065-4 / ACES container chromaticities attribute, in EXR field order:
# redX, redY, greenX, greenY, blueX, blueY, whiteX, whiteY.
AP0_CHROMATICITIES = (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)
REC2020_CHROMATICITIES = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290)

_BRADFORD = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
], dtype=np.float64)


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    if abs(y) < 1e-12:
        raise ValueError("chromaticity y must be non-zero")
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def npm(space: str) -> np.ndarray:
    """Normalized primary matrix: linear RGB (space) -> CIE XYZ (own white)."""
    if space not in PRIMARIES:
        raise KeyError(f"unknown space {space!r}; known: {sorted(PRIMARIES)}")
    rx, ry, gx, gy, bx, by = PRIMARIES[space]
    prim = np.stack([_xy_to_xyz(rx, ry), _xy_to_xyz(gx, gy), _xy_to_xyz(bx, by)], axis=1)
    white = _xy_to_xyz(*WHITE_POINTS[space])
    scale = np.linalg.solve(prim, white)
    return prim * scale[None, :]


def bradford_cat(src_white_xy: tuple[float, float],
                 dst_white_xy: tuple[float, float]) -> np.ndarray:
    """Bradford chromatic adaptation matrix XYZ(src white) -> XYZ(dst white)."""
    if np.allclose(src_white_xy, dst_white_xy, atol=1e-9):
        return np.eye(3, dtype=np.float64)
    src = _BRADFORD @ _xy_to_xyz(*src_white_xy)
    dst = _BRADFORD @ _xy_to_xyz(*dst_white_xy)
    gain = np.diag(dst / src)
    return np.linalg.inv(_BRADFORD) @ gain @ _BRADFORD


def rgb_to_rgb_matrix(src: str, dst: str) -> np.ndarray:
    """Exact linear RGB src -> dst matrix, Bradford-adapting the white point."""
    if src == dst:
        return np.eye(3, dtype=np.float64)
    cat = bradford_cat(WHITE_POINTS[src], WHITE_POINTS[dst])
    return np.linalg.inv(npm(dst)) @ cat @ npm(src)


def convert(rgb: np.ndarray, src: str, dst: str, clip_negative: bool = False) -> np.ndarray:
    """Convert an (..., 3) linear RGB array between primaries.

    ``clip_negative`` defaults to False: AP0 encodes the full spectral locus so
    Rec.2020 -> AP0 never needs clipping, while the reverse can legitimately
    produce out-of-gamut negatives the downstream grade should decide about.
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError(f"expected (..., 3) RGB, got {arr.shape}")
    out = arr @ rgb_to_rgb_matrix(src, dst).T
    if clip_negative:
        out = np.maximum(out, 0.0)
    return out.astype(np.float32)
