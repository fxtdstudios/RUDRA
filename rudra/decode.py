"""Read an SDR frame WITHOUT throwing away its bit depth.

Every input path used ``PIL.Image.open(...).convert("RGB")``, and PIL's "RGB"
mode is 8 bits per channel. A true 16-bit PNG opens, reports mode RGB, and
comes back as uint8: 512 distinct codes on disk became 256, silently, with no
warning and no error.

That is expensive here in a way it would not be for a viewer, because the
thing being thrown away is exactly what the model is then asked to
reconstruct. Measured 10 Sep 2026 -- a smooth sky with faint texture, the same
baseline curve, only the source's depth changing:

    source     out HF %    chroma fleck
    8-bit          2.36            1.53
    10-bit         1.04            0.67
    12-bit         0.85            0.50
    float          0.83            0.49

The curve amplifies whatever high-frequency content it is handed by about 30x
whatever the depth, so roughly two thirds of the sparkle in an 8-bit sky is the
eight bits, and it is gone by ten. In a generative pipeline the VAE decodes to
float and the frame is quantised on the way to a PNG -- one step before RUDRA
is asked to put back what the quantiser removed.

Integer formats carry the same display encoding at any depth, so they are
simply normalised by their dtype's maximum. Float input is assumed to be
display-encoded in [0, 1]; a float file carrying values above 1 is scene-linear
HDR, which does not need reconstructing, and is refused rather than silently
clipped.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DecodedFrame:
    rgb: np.ndarray          # float32 (H, W, 3) in [0, 1], display-encoded
    bits: int                # bits per channel as stored
    distinct_codes: int      # actually present in the green channel
    source: str              # which decoder read it

    @property
    def truncated_to_8bit(self) -> bool:
        return self.bits <= 8


def decode_sdr(data: bytes | np.ndarray) -> DecodedFrame:
    """Decode an SDR frame at full precision."""
    if isinstance(data, np.ndarray):
        arr, source = data, "array"
    else:
        buf = np.frombuffer(data, dtype=np.uint8)
        # IMREAD_UNCHANGED is the whole point: IMREAD_COLOR converts to 8-bit.
        arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        source = "opencv"
        if arr is None:
            # OpenCV declines a few things Pillow reads (some TIFF flavours,
            # WebP builds). Falling back costs the depth, so say so.
            from PIL import Image
            arr = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
            source = "pillow-8bit-fallback"
        if arr.ndim == 3 and arr.shape[2] >= 3 and source == "opencv":
            arr = arr[..., :3][..., ::-1]                  # BGR -> RGB
    if arr.ndim == 2:
        arr = np.dstack([arr] * 3)
    if arr.shape[2] > 3:
        arr = arr[..., :3]

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        bits = int(np.log2(info.max + 1))
        rgb = arr.astype(np.float32) / float(info.max)
    else:
        bits = 32
        top = float(np.nanmax(arr)) if arr.size else 0.0
        if top > 1.0 + 1e-4:
            raise ValueError(
                f"float input peaks at {top:.3f}: this is scene-linear HDR, not an "
                "SDR frame. RUDRA reconstructs display-encoded SDR; an HDR file "
                "does not need reconstructing.")
        rgb = np.clip(arr.astype(np.float32), 0.0, 1.0)

    distinct = int(np.unique(arr[..., 1]).size)
    # A 16-bit container can hold an 8-bit frame padded up, and calling that
    # "16-bit" lets a pipeline believe a problem was fixed. Detect it EXACTLY
    # -- every value sitting on the 8-bit lattice, n * 257 for uint16 -- rather
    # than by counting distinct codes. Counting misfires twice: a flat float
    # frame has one distinct value and was reported as 8-bit, and a genuinely
    # 16-bit sky with little detail would be too.
    effective = bits
    if np.issubdtype(arr.dtype, np.integer) and bits > 8:
        top = np.iinfo(arr.dtype).max
        # Two ways an 8-bit frame gets padded into a wider container: scaled to
        # the full range (code * 257 for uint16, so 255 -> 65535) or shifted
        # (code * 256, so 255 -> 65280). Test both. Using 256 alone missed the
        # scaled case, which is the one every image library produces.
        for step in (top // 255, (top + 1) // 256):
            if step > 1 and not np.any(arr % step):
                effective = 8
                break
    return DecodedFrame(np.ascontiguousarray(rgb), effective, distinct, source)
