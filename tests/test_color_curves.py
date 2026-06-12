"""Round-trip identity tests for the camera log curves (review §3.1).

These guard the single most damaging bug in the codebase: a forward/inverse
mismatch between target encoding and scene-linear decoding. If any curve's
encode/decode pair stops being an identity, supervision is silently corrupted —
so this test must stay green.

Run:  pytest tests/test_color_curves.py
"""

import math

import pytest

torch = pytest.importorskip("torch")

from rudra.color_curves import (
    LINEAR_TO_LOG,
    LOG_TO_LINEAR,
    encode_linear,
    decode_to_linear,
)


# A linear sweep that spans deep shadows (incl. small negatives), middle grey,
# and bright HDR highlights up to 200x diffuse white.
def _sweep():
    lo = torch.linspace(-0.02, 1.0, 60)
    hi = torch.logspace(math.log10(1e-3), math.log10(200.0), 60)
    return torch.cat([lo, hi])


@pytest.mark.parametrize("key", sorted(LINEAR_TO_LOG.keys()))
def test_round_trip_identity(key):
    x = _sweep().double()
    rt = decode_to_linear(encode_linear(x, key), key)
    # log curves clip below their black floor; compare where decode is defined.
    err = (rt - x).abs().max().item()
    assert err < 1e-4, f"{key} round-trip error {err:.2e} too large"


@pytest.mark.parametrize("key", sorted(LINEAR_TO_LOG.keys()))
def test_middle_grey_is_finite_and_monotonic(key):
    x = torch.linspace(0.0, 10.0, 256)
    code = encode_linear(x, key)
    assert torch.isfinite(code).all(), f"{key} produced non-finite codes"
    # Encoding must be monotonic non-decreasing in linear light.
    assert (code[1:] - code[:-1]).min().item() > -1e-5, f"{key} encode not monotonic"


def test_decode_matches_normalization_dispatch():
    """normalize_to_scene_linear must use the exact curve inverse, not the
    generic approximation, for every named log format."""
    from rudra.config import FORMAT_TO_ID
    from rudra.normalization import normalize_to_scene_linear

    x = torch.rand(2, 3, 16, 16)
    for key in ["logc3", "logc4", "slog3", "vlog", "log3g10"]:
        fid = FORMAT_TO_ID[key]
        code = encode_linear(torch.rand(2, 3, 16, 16).clamp(min=0), key)
        via_norm = normalize_to_scene_linear(code, fid)
        via_curve = decode_to_linear(code, key).clamp(min=0.0)
        assert torch.allclose(via_norm, via_curve, atol=1e-4), f"{key} dispatch mismatch"
