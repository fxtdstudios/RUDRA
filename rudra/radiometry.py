"""Canonical radiometric constants for RUDRA (torch-free, numpy-free).

The single source of truth for every transfer-function and luma constant that
was previously copy-pasted across the codebase (refactor audit 2026-08-22:
the Rec.2020 luma triplet appeared in 8 files, sRGB thresholds in 4, the
ACES fit coefficients in 3). Torch modules, numpy modules, and the delivery
layer all import from here, and ``tests/test_radiometry_2026_08_22.py``
asserts the surviving parallel implementations (torch vs numpy vs storage)
agree numerically — so drift between them is now a test failure, not a
silent supervision bug.

Deliberately NOT unified here:
  - ``pipeline/hdr_io.py`` keeps its own PQ constants: it is the
    self-contained storage codec whose encode/decode must never depend on
    package state. The equivalence test pins it to ``rudra.hdr10`` instead.
  - The full RGB->XYZ matrices in descriptor/hdrvdp/metrics and the
    primaries-derived matrices in ``rudra.delivery.colorspace`` — matrices
    are derived or spec-quoted per use; only scalar luma weights live here.
"""

from __future__ import annotations

# ── Luma weights (BT.709 / BT.2020 / ACEScg Y rows) ─────────────────────────
LUMA_REC709 = (0.2126, 0.7152, 0.0722)
LUMA_REC2020 = (0.2627, 0.6780, 0.0593)
LUMA_ACESCG = (0.272229, 0.674082, 0.053689)

# ── SMPTE ST 2084 (PQ) ──────────────────────────────────────────────────────
PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 32.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 128.0
PQ_C3 = 2392.0 / 128.0
PQ_PEAK_NITS = 10000.0
# BT.2408 graphics/diffuse ("reference") white for PQ containers.
PQ_REF_WHITE_NITS = 203.0

# ── IEC 61966-2-1 sRGB piecewise transfer ───────────────────────────────────
SRGB_CODE_KNEE = 0.04045      # encoded-side linear-segment end
SRGB_LINEAR_KNEE = 0.0031308  # linear-side segment end
SRGB_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_GAMMA = 2.4

# ── ARIB STD-B67 (HLG) OETF constants ───────────────────────────────────────
HLG_A = 0.17883277
HLG_B = 0.28466892
HLG_C = 0.55991073
HLG_REF_WHITE_SIGNAL = 0.75   # BT.2408 reference-white signal level

# ── Narkowicz ACES filmic fit (forward: x(ax+b) / (x(cx+d)+e)) ─────────────
# Used by prepare_training_data (forward, SDR synthesis), sdr2hdr (analytic
# inverse baseline), and infer_sdr2hdr (preview). One drifted coefficient
# here previously meant corrupted supervision — hence one definition.
ACES_A = 2.51
ACES_B = 0.03
ACES_C = 2.43
ACES_D = 0.59
ACES_E = 0.14


def aces_tonemap(x):
    """Forward Narkowicz ACES fit; works on floats, numpy arrays, tensors."""
    return (x * (ACES_A * x + ACES_B)) / (x * (ACES_C * x + ACES_D) + ACES_E)


def luma_cf(x, weights=LUMA_REC2020):
    """Channel-first luminance: x is (B, 3, H, W) torch tensor or ndarray."""
    w0, w1, w2 = weights
    return w0 * x[:, 0:1] + w1 * x[:, 1:2] + w2 * x[:, 2:3]
