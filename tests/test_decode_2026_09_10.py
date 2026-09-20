"""Reading an SDR frame without throwing away its bit depth.

The defect was silent and total: ``PIL.Image.open(path).convert("RGB")`` on a
true 16-bit PNG reports mode RGB and returns uint8. 512 distinct codes on disk
became 256, in all three of the server's input paths, with no warning. The
thing discarded is exactly what the model is then asked to reconstruct.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.decode import decode_sdr  # noqa: E402


def ramp(bits: int, width: int = 512) -> np.ndarray:
    levels = 2 ** bits - 1
    line = np.linspace(0, levels, width)
    dtype = np.uint8 if bits <= 8 else np.uint16
    return np.dstack([line.astype(dtype)[None, :].repeat(64, 0)] * 3)


def encode(arr: np.ndarray, tmp_path: Path, name: str) -> bytes:
    path = tmp_path / name
    cv2.imwrite(str(path), arr)
    return path.read_bytes()


def test_sixteen_bits_survive(tmp_path):
    data = encode(ramp(16), tmp_path, "a.png")
    d = decode_sdr(data)
    assert d.bits == 16
    assert d.distinct_codes > 256, d.distinct_codes
    assert d.rgb.dtype == np.float32


def test_eight_bits_are_read_as_eight(tmp_path):
    d = decode_sdr(encode(ramp(8), tmp_path, "b.png"))
    assert d.bits == 8
    assert d.truncated_to_8bit


def test_a_sixteen_bit_container_holding_eight_bits_reports_eight(tmp_path):
    """An 8-bit frame padded into a 16-bit file has not gained information, and
    saying '16-bit' about it would let a pipeline believe a problem was fixed."""
    eight = ramp(8).astype(np.uint16) * 257          # 0..255 -> 0..65535
    d = decode_sdr(encode(eight, tmp_path, "c.png"))
    assert d.bits == 8, d.bits


def test_channel_order_is_rgb_not_bgr(tmp_path):
    """OpenCV decodes BGR. Getting this wrong swaps red and blue in every
    reconstruction, which is obvious on a sunset and invisible on a grey."""
    arr = np.zeros((8, 8, 3), np.uint8)
    arr[..., 2] = 255                                # OpenCV's channel 2 is RED
    d = decode_sdr(encode(arr, tmp_path, "d.png"))
    assert d.rgb[0, 0, 0] == pytest.approx(1.0)
    assert d.rgb[0, 0, 2] == pytest.approx(0.0)


def test_values_land_in_zero_to_one(tmp_path):
    for bits in (8, 16):
        d = decode_sdr(encode(ramp(bits), tmp_path, f"e{bits}.png"))
        assert d.rgb.min() >= 0.0 and d.rgb.max() <= 1.0
        assert d.rgb.max() == pytest.approx(1.0, abs=1e-4)


def test_greyscale_becomes_three_channels(tmp_path):
    grey = np.linspace(0, 255, 64).astype(np.uint8)[None, :].repeat(16, 0)
    d = decode_sdr(encode(grey, tmp_path, "f.png"))
    assert d.rgb.shape[2] == 3
    assert np.allclose(d.rgb[..., 0], d.rgb[..., 2])


def test_an_alpha_channel_is_dropped(tmp_path):
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., :3] = 128
    rgba[..., 3] = 64
    d = decode_sdr(encode(rgba, tmp_path, "g.png"))
    assert d.rgb.shape[2] == 3


def test_scene_linear_hdr_is_refused():
    """A float file above 1.0 is HDR already. Clipping it silently would hand
    back a broken SDR and reconstruct it as though it needed reconstructing."""
    hdr = np.full((8, 8, 3), 12.0, np.float32)
    with pytest.raises(ValueError, match="scene-linear HDR"):
        decode_sdr(hdr)


def test_display_encoded_float_is_accepted():
    d = decode_sdr(np.full((8, 8, 3), 0.5, np.float32))
    assert d.bits == 32
    assert d.rgb.max() == pytest.approx(0.5)


def test_sixteen_bit_input_reaches_the_curve_with_more_levels(tmp_path):
    """The point of all of this: more codes in, more levels out."""
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    import torch
    outs = {}
    for bits in (8, 16):
        d = decode_sdr(encode(ramp(bits), tmp_path, f"h{bits}.png"))
        x = torch.from_numpy(d.rgb).permute(2, 0, 1)[None]
        outs[bits] = np.unique(sdr_to_baseline_hdr(x).numpy().round(6)).size
    assert outs[16] > 1.5 * outs[8], outs
