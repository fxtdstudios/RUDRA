"""Golden files for the native still decoder (native/media still.cpp).

Principle P4: the oracle is rudra/decode.py decode_sdr, run on fixture files
this script writes in every format and depth the Studio accepts. The native
test decodes the same bytes and must produce the same floats, the same
effective bit depth, the same distinct-code count, or the same refusal.

    python tools/emit_decode_golden.py                  # expected results for the committed fixtures
    python tools/emit_decode_golden.py --new-fixtures   # rewrite the fixture files too

The fixture files are inputs: once committed they are reused, so a different
encoder version on another machine cannot change what is being tested. They
are small (37 x 23, odd on purpose so chroma subsampling has an edge to get
wrong).
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.decode import decode_sdr  # noqa: E402

OUT = REPO / "native" / "tests" / "golden" / "decode"
H, W = 23, 37


def scene(rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) float in [0, 1]: a ramp, a clipped patch, texture."""
    y, x = np.mgrid[:H, :W]
    img = np.stack([x / (W - 1), y / (H - 1), (x + y) / (W + H - 2)], -1)
    img[3:9, 25:33] = 1.0
    img = img + rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0.0, 1.0)


def encode(ext: str, arr: np.ndarray, params: list[int] | None = None) -> bytes:
    ok, buf = cv2.imencode(ext, arr, params or [])
    if not ok:
        raise RuntimeError(f"cv2 could not write {ext}")
    return buf.tobytes()


def fixtures() -> list[tuple[str, bytes]]:
    rng = np.random.default_rng(20260923)
    rgb = scene(rng)
    bgr = rgb[..., ::-1]
    grey = rgb.mean(-1)
    f: list[tuple[str, bytes]] = []
    u8, u16 = (bgr * 255).round().astype(np.uint8), (bgr * 65535).round().astype(np.uint16)
    f.append(("png8_rgb.png", encode(".png", u8)))
    f.append(("png16_rgb.png", encode(".png", u16)))
    f.append(("png16_padded_scaled.png", encode(".png", u8.astype(np.uint16) * 257)))
    f.append(("png16_padded_shifted.png", encode(".png", u8.astype(np.uint16) * 256)))
    f.append(("png8_grey.png", encode(".png", (grey * 255).round().astype(np.uint8))))
    f.append(("png16_grey.png", encode(".png", (grey * 65535).round().astype(np.uint16))))
    alpha = (np.linspace(0, 255, W)[None, :].repeat(H, 0)).astype(np.uint8)
    f.append(("png8_rgba.png", encode(".png", np.dstack([u8, alpha]))))
    from PIL import Image
    pal = io.BytesIO()
    Image.fromarray((rgb * 255).round().astype(np.uint8)).convert("P", palette=Image.ADAPTIVE, colors=64).save(pal, "PNG")
    f.append(("png8_palette.png", pal.getvalue()))
    f.append(("jpeg_q92_420.jpg", encode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 92])))
    f.append(("jpeg_q100_444.jpg", encode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, 100,
                                                        cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                                                        cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])))
    f.append(("jpeg_grey.jpg", encode(".jpg", (grey * 255).round().astype(np.uint8))))
    f.append(("tiff8_rgb.tif", encode(".tif", u8)))
    f.append(("tiff16_rgb.tif", encode(".tif", u16)))
    f.append(("tiff32f_rgb.tif", encode(".tif", bgr.astype(np.float32))))
    f.append(("tiff32f_hdr.tif", encode(".tif", (bgr * 4.0).astype(np.float32))))   # refused
    f.append(("bmp24.bmp", encode(".bmp", u8)))
    f.append(("webp_lossless.webp", encode(".webp", u8, [cv2.IMWRITE_WEBP_QUALITY, 101])))
    return f


def main() -> int:
    new = "--new-fixtures" in sys.argv[1:]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in list(OUT.glob("*.npy")):
        old.unlink()
    cases = []
    for name, data in fixtures():
        path = OUT / name
        if new or not path.exists():
            path.write_bytes(data)
        data = path.read_bytes()
        case = {"file": name}
        try:
            d = decode_sdr(data)
        except ValueError as e:
            case["refused"] = str(e)
        else:
            exp = OUT / (Path(name).stem + ".npy")
            np.save(exp, np.ascontiguousarray(np.transpose(d.rgb, (2, 0, 1)).astype(np.float32)), allow_pickle=False)
            case.update({"expected": exp.name, "bits": d.bits, "distinct_codes": d.distinct_codes,
                         "source": d.source})
        cases.append(case)
    (OUT / "index.json").write_text(json.dumps({"oracle": "rudra/decode.py decode_sdr",
                                                "cases": cases}, indent=2), encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} decode fixtures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
