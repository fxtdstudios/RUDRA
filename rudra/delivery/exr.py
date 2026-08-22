"""Minimal, dependency-free OpenEXR 2.0 scanline I/O for RUDRA delivery.

Writes uncompressed HALF/FLOAT RGB EXRs with correct headers — including the
``chromaticities`` attribute that makes a file a legal ACES container image
(SMPTE ST 2065-4) — using nothing but numpy and struct. Replaces the
long-standing ``rudra.exr_io.save_exr_placeholder`` refusal: RUDRA can now
emit EXR masters on any machine, no OpenEXR/OpenImageIO wheel required.

Scope (deliberate): NO compression only, scanline only, RGB(A) only. The
reader exists to round-trip our own files and ingest simple reference EXRs;
it refuses compressed input with a clear error instead of mis-reading it.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

__all__ = ["write_exr", "read_exr"]

_MAGIC = 20000630
_PIXEL_HALF = 1
_PIXEL_FLOAT = 2


def _attr(name: str, type_name: str, payload: bytes) -> bytes:
    return name.encode() + b"\0" + type_name.encode() + b"\0" + struct.pack("<i", len(payload)) + payload


def _chlist(channels: list[str], pixel_type: int) -> bytes:
    out = b""
    for ch in channels:  # must be alphabetically sorted
        out += ch.encode() + b"\0"
        out += struct.pack("<i", pixel_type)
        out += struct.pack("<B3x", 0)          # pLinear + reserved
        out += struct.pack("<ii", 1, 1)        # x/y sampling
    return out + b"\0"


def write_exr(
    path: str | Path,
    rgb: np.ndarray,
    half: bool = True,
    chromaticities: tuple[float, ...] | None = None,
    attributes: dict[str, str] | None = None,
) -> Path:
    """Write an (H, W, 3) or (H, W, 4) linear float image as scanline EXR.

    Args:
        rgb: linear image, float; NaN/inf are sanitized to finite values.
        half: store as 16-bit float (HALF). False stores 32-bit FLOAT —
            use for log2_extended-range data exceeding half's ~65504 max.
        chromaticities: 8 floats (rx,ry,gx,gy,bx,by,wx,wy). Pass
            ``colorspace.AP0_CHROMATICITIES`` for an ACES container file.
        attributes: extra string attributes (e.g. provenance JSON).
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected (H, W, 3|4) image, got {arr.shape}")
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=3.4e38, neginf=0.0)
    height, width, n_ch = arr.shape
    names = ["R", "G", "B", "A"][:n_ch]
    sorted_names = sorted(names)                       # EXR requires sorted chlist
    dtype = np.float16 if half else np.float32
    pixel_type = _PIXEL_HALF if half else _PIXEL_FLOAT
    if half:
        arr = np.clip(arr, -65504.0, 65504.0)

    header = b""
    header += _attr("channels", "chlist", _chlist(sorted_names, pixel_type))
    header += _attr("compression", "compression", struct.pack("<B", 0))
    box = struct.pack("<4i", 0, 0, width - 1, height - 1)
    header += _attr("dataWindow", "box2i", box)
    header += _attr("displayWindow", "box2i", box)
    header += _attr("lineOrder", "lineOrder", struct.pack("<B", 0))
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    if chromaticities is not None:
        if len(chromaticities) != 8:
            raise ValueError("chromaticities needs 8 floats (rx,ry,gx,gy,bx,by,wx,wy)")
        header += _attr("chromaticities", "chromaticities", struct.pack("<8f", *chromaticities))
    for key, value in (attributes or {}).items():
        header += _attr(key, "string", value.encode("utf-8"))
    header += b"\0"

    bytes_per_px = 2 if half else 4
    row_data_size = width * bytes_per_px * n_ch
    chunk_size = 8 + row_data_size                     # y + size prefix per chunk
    table_pos = 8 + len(header)
    data_start = table_pos + 8 * height

    planes = {name: np.ascontiguousarray(arr[:, :, names.index(name)], dtype=dtype)
              for name in sorted_names}

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<ii", _MAGIC, 2))
        fh.write(header)
        offsets = [data_start + y * chunk_size for y in range(height)]
        fh.write(struct.pack(f"<{height}Q", *offsets))
        for y in range(height):
            fh.write(struct.pack("<ii", y, row_data_size))
            for name in sorted_names:
                fh.write(planes[name][y].tobytes())
    return path


def _read_null_str(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.index(b"\0", pos)
    return buf[pos:end].decode("latin-1"), end + 1


def read_exr(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an uncompressed scanline RGB(A) EXR written by this module
    (or any conforming writer). Returns (float32 (H,W,C) image, attributes).

    Raises on compressed, tiled, deep, or multi-part input — by design, so a
    file this reader cannot faithfully decode is never silently mangled.
    """
    raw = Path(path).read_bytes()
    magic, version = struct.unpack_from("<ii", raw, 0)
    if magic != _MAGIC:
        raise ValueError(f"{path}: not an EXR file")
    if version & 0x200 or version & 0x800 or version & 0x1000:
        raise ValueError(f"{path}: tiled/deep/multi-part EXR unsupported")

    pos = 8
    attrs: dict = {}
    channels: list[tuple[str, int]] = []
    while raw[pos] != 0:
        name, pos = _read_null_str(raw, pos)
        type_name, pos = _read_null_str(raw, pos)
        (size,) = struct.unpack_from("<i", raw, pos)
        pos += 4
        payload = raw[pos:pos + size]
        pos += size
        if type_name == "chlist":
            cpos = 0
            while payload[cpos] != 0:
                ch, cpos = _read_null_str(payload, cpos)
                (ptype,) = struct.unpack_from("<i", payload, cpos)
                channels.append((ch, ptype))
                cpos += 16
        elif type_name == "compression":
            attrs["compression"] = payload[0]
        elif type_name == "box2i":
            attrs[name] = struct.unpack("<4i", payload)
        elif type_name == "chromaticities":
            attrs["chromaticities"] = struct.unpack("<8f", payload)
        elif type_name == "string":
            attrs[name] = payload.decode("utf-8", "replace")
        else:
            attrs.setdefault("_raw", {})[name] = (type_name, payload)
    pos += 1  # header terminator

    if attrs.get("compression", 0) != 0:
        raise ValueError(f"{path}: compressed EXR unsupported by this minimal reader")
    x0, y0, x1, y1 = attrs["dataWindow"]
    width, height = x1 - x0 + 1, y1 - y0 + 1
    pos += 8 * height  # skip offset table

    names = [c for c, _ in channels]
    dtypes = {c: (np.float16 if t == _PIXEL_HALF else np.float32) for c, t in channels}
    planes = {c: np.empty((height, width), dtype=np.float32) for c in names}
    for _ in range(height):
        y, size = struct.unpack_from("<ii", raw, pos)
        pos += 8
        for c in names:
            dt = dtypes[c]
            count = width * dt().itemsize
            planes[c][y - y0] = np.frombuffer(raw, dtype=dt, count=width, offset=pos).astype(np.float32)
            pos += count
    order = [c for c in ("R", "G", "B", "A") if c in planes] or names
    image = np.stack([planes[c] for c in order], axis=-1)
    attrs["channel_order"] = order
    return image, attrs
