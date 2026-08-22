"""Single source of truth for how RUDRA stores HDR targets on disk.

The August 2026 run stored targets as 16-bit *linear* PNG, hard-clipped at
10,000 nits.  Two consequences, both measured:

  * 77.8% of the still corpus peaked above the ceiling, so the highlights the
    model exists to reconstruct were clipped out of its own supervision.
  * Linear quantisation spends codes where the light is: 88.9% of pixels in a
    reference frame fell below code 256, leaving the shadows roughly 8 bits.

This module replaces that with two *explicit* storage modes and makes the
encode and decode sides share one implementation, so they cannot drift.

    STORAGE_MODES
      "pq_10000"       display-referred.  scene-linear -> nits -> clamp at
                       10,000 -> ST-2084 -> uint16.  Matches HDR10 delivery and
                       the ColorVideoVDP convention.  Highlights above 10,000
                       nits are still lost -- deliberately, and recorded per
                       record as ``clipped_fraction``.
      "log2_extended"  scene-referred.  constant *relative* precision from
                       ``floor_nits`` to ``ceiling_nits``.  At the defaults
                       (0.005 .. 1e6 nits, 27.6 stops) that is ~2,380 codes per
                       stop, and an 822,000-nit sun survives intact.

Pick one per dataset, record it in the sentinel, and never mix.

Round-trip accuracy is asserted by ``selftest()``; ``prepare_pairs.py`` runs it
before writing a single file.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np

# Format version. Bump when the on-disk meaning of a code value changes;
# verify_dataset.py refuses to mix versions.
HDR_IO_VERSION = 3

# BT.2408 diffuse white. Scene-linear 1.0 corresponds to this many nits.
# This is the anchor the whole repo already uses (rudra/normalization.py,
# prepare_training_data.py); do not change it without re-preparing everything.
DIFFUSE_WHITE_NITS = 203.0

PQ_PEAK_NITS = 10_000.0

# log2_extended defaults: 0.005 nits is below any display black, 1e6 nits is
# above the brightest sun measured in the Poly Haven corpus (88.2M nits exists
# but is a single specular pixel; see --ceiling-nits to raise it).
LOG2_FLOOR_NITS = 0.005
LOG2_CEILING_NITS = 1_000_000.0

StorageMode = Literal["pq_10000", "log2_extended"]
UINT16_MAX = 65535.0


@dataclass(frozen=True)
class HDRStorage:
    """Everything needed to decode a stored target. Goes in the sentinel."""

    mode: StorageMode = "pq_10000"
    diffuse_white_nits: float = DIFFUSE_WHITE_NITS
    floor_nits: float = LOG2_FLOOR_NITS
    ceiling_nits: float = LOG2_CEILING_NITS
    version: int = HDR_IO_VERSION

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "HDRStorage":
        known = {f: payload[f] for f in cls.__dataclass_fields__ if f in payload}
        return cls(**known)

    def describe(self) -> str:
        if self.mode == "pq_10000":
            return (f"PQ / ST-2084, ceiling {PQ_PEAK_NITS:,.0f} nits, "
                    f"diffuse white {self.diffuse_white_nits:g} nits (display-referred)")
        stops = np.log2(self.ceiling_nits / self.floor_nits)
        return (f"log2, {self.floor_nits:g}..{self.ceiling_nits:,.0f} nits "
                f"({stops:.1f} stops, {UINT16_MAX / stops:,.0f} codes/stop, scene-referred)")


# --------------------------------------------------------------------------
# ST-2084 (PQ)
# --------------------------------------------------------------------------
_M1 = 2610.0 / 16384.0
_M2 = 2523.0 / 32.0
_C1 = 3424.0 / 4096.0
_C2 = 2413.0 / 128.0
_C3 = 2392.0 / 128.0


def pq_oetf(nits: np.ndarray) -> np.ndarray:
    """Absolute nits -> normalised ST-2084 code in [0,1]."""
    y = np.clip(np.asarray(nits, dtype=np.float64) / PQ_PEAK_NITS, 0.0, 1.0)
    p = np.power(y, _M1)
    return np.power((_C1 + _C2 * p) / (1.0 + _C3 * p), _M2)


def pq_eotf(code: np.ndarray) -> np.ndarray:
    """Normalised ST-2084 code in [0,1] -> absolute nits."""
    v = np.clip(np.asarray(code, dtype=np.float64), 0.0, 1.0)
    p = np.power(v, 1.0 / _M2)
    ratio = np.maximum(p - _C1, 0.0) / np.maximum(_C2 - _C3 * p, 1e-12)
    return PQ_PEAK_NITS * np.power(ratio, 1.0 / _M1)


# --------------------------------------------------------------------------
# Encode / decode
# --------------------------------------------------------------------------
def linear_to_nits(linear: np.ndarray, storage: HDRStorage) -> np.ndarray:
    return np.maximum(np.asarray(linear, dtype=np.float64), 0.0) * storage.diffuse_white_nits


def nits_to_linear(nits: np.ndarray, storage: HDRStorage) -> np.ndarray:
    return np.asarray(nits, dtype=np.float64) / storage.diffuse_white_nits


def encode_hdr_u16(linear: np.ndarray, storage: HDRStorage) -> tuple[np.ndarray, dict]:
    """Scene-linear float -> uint16 array plus per-record encode statistics.

    The statistics are not decoration: ``clipped_fraction`` and ``peak_nits``
    are what verify_dataset.py gates on, and what the audit found nobody was
    recording.
    """
    nits = linear_to_nits(linear, storage)
    peak = float(nits.max()) if nits.size else 0.0

    if storage.mode == "pq_10000":
        ceiling = PQ_PEAK_NITS
        clipped = float((nits > ceiling).mean()) if nits.size else 0.0
        code = pq_oetf(nits)
    elif storage.mode == "log2_extended":
        ceiling = storage.ceiling_nits
        clipped = float((nits > ceiling).mean()) if nits.size else 0.0
        span = np.log2(storage.ceiling_nits / storage.floor_nits)
        safe = np.clip(nits, storage.floor_nits, storage.ceiling_nits)
        code = np.log2(safe / storage.floor_nits) / span
    else:
        raise ValueError(f"unknown storage mode {storage.mode!r}")

    if storage.mode == "log2_extended":
        # Code 0 is reserved for true black; the log ramp occupies 1..65535 so
        # that floor_nits itself is representable and distinct from zero.
        ramp = np.rint(np.clip(code, 0.0, 1.0) * (UINT16_MAX - 1.0)) + 1.0
        quantised = np.where(nits <= 0.0, 0.0, ramp).astype(np.uint16)
    else:
        quantised = np.rint(np.clip(code, 0.0, 1.0) * UINT16_MAX).astype(np.uint16)
    stats = {
        "peak_nits": peak,
        "storage_ceiling_nits": float(ceiling),
        "clipped_fraction": clipped,
        "codes_used": int(np.unique(quantised).size),
        "codes_below_diffuse_white": int(
            (quantised < np.rint(_code_for_nits(storage.diffuse_white_nits, storage) * UINT16_MAX)).sum()
        ),
    }
    return quantised, stats


def decode_hdr_u16(code: np.ndarray, storage: HDRStorage) -> np.ndarray:
    """uint16 array -> scene-linear float32. Exact inverse of encode_hdr_u16."""
    raw = np.asarray(code, dtype=np.float64)
    if storage.mode == "pq_10000":
        nits = pq_eotf(raw / UINT16_MAX)
    elif storage.mode == "log2_extended":
        span = np.log2(storage.ceiling_nits / storage.floor_nits)
        norm = np.clip((raw - 1.0) / (UINT16_MAX - 1.0), 0.0, 1.0)
        nits = np.where(raw == 0, 0.0, storage.floor_nits * np.power(2.0, norm * span))
    else:
        raise ValueError(f"unknown storage mode {storage.mode!r}")
    return nits_to_linear(nits, storage).astype(np.float32)


def _code_for_nits(nits: float, storage: HDRStorage) -> float:
    if storage.mode == "pq_10000":
        return float(pq_oetf(np.asarray([nits]))[0])
    span = np.log2(storage.ceiling_nits / storage.floor_nits)
    v = min(max(nits, storage.floor_nits), storage.ceiling_nits)
    ramp = np.log2(v / storage.floor_nits) / span
    return float((ramp * (UINT16_MAX - 1.0) + 1.0) / UINT16_MAX)


def shadow_precision(storage: HDRStorage) -> dict:
    """How many code values the encoding actually gives the shadows.

    The audit's finding, restated as a number you can regression-test: under
    16-bit linear, everything below 39 nits shared 256 codes.
    """
    marks = {
        "0.1_nits": 0.1,
        "1_nit": 1.0,
        "10_nits": 10.0,
        "diffuse_white_203_nits": storage.diffuse_white_nits,
        "1000_nits": 1000.0,
    }
    out = {}
    previous = 0.0
    for name, nits in marks.items():
        code = _code_for_nits(nits, storage) * UINT16_MAX
        out[name] = {"code": round(code, 1), "codes_since_previous": round(code - previous, 1)}
        previous = code
    return out


def selftest(storage: HDRStorage | None = None, tolerance: float = 0.01) -> dict:
    """Assert round-trip accuracy across the full range. Raises on failure."""
    storage = storage or HDRStorage()
    ceiling = PQ_PEAK_NITS if storage.mode == "pq_10000" else storage.ceiling_nits
    probe_nits = np.geomspace(max(storage.floor_nits, 1e-3), ceiling * 0.999, 512)
    linear = nits_to_linear(probe_nits, storage).astype(np.float32)

    code, _ = encode_hdr_u16(linear, storage)
    back = decode_hdr_u16(code, storage)
    back_nits = linear_to_nits(back, storage)

    rel = np.abs(back_nits - probe_nits) / np.maximum(probe_nits, 1e-9)
    worst = float(rel.max())
    if worst > tolerance:
        idx = int(np.argmax(rel))
        raise AssertionError(
            f"hdr_io round-trip failed for {storage.mode}: worst relative error "
            f"{worst:.4%} at {probe_nits[idx]:.4g} nits (tolerance {tolerance:.2%})"
        )
    return {
        "mode": storage.mode,
        "worst_relative_error": worst,
        "median_relative_error": float(np.median(rel)),
        "probes": int(probe_nits.size),
    }


if __name__ == "__main__":
    import json

    for mode in ("pq_10000", "log2_extended"):
        st = HDRStorage(mode=mode)  # type: ignore[arg-type]
        print(f"\n{st.describe()}")
        print("  round-trip:", json.dumps(selftest(st), indent=None))
        print("  shadow precision:")
        for name, info in shadow_precision(st).items():
            print(f"    {name:<26} code {info['code']:>9}   (+{info['codes_since_previous']} since previous)")
