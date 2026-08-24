"""Diagnose exactly why a source file fails to load in the ingest.

    python pipeline\\probe_read.py "\\\\192.168.100.200\\Data\\08_Research\\Source_HDR\\PolyHaven\\abandoned_bakery_2k.exr"
    python pipeline\\probe_read.py E:\\source_hdr\\HdM-HFR-2017_Color-Graded\\...\\some_frame.tif

Prints, for the given path: raw byte access, each available backend
(OpenEXR / imageio / tifffile / cv2), and finally the ingest's own
read_exr / read_tif — with the full traceback of whatever fails.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def attempt(label, fn):
    try:
        result = fn()
        shape = getattr(result, "shape", None)
        print(f"  OK    {label}: {shape if shape is not None else type(result).__name__}")
        return True
    except Exception:
        exc = traceback.format_exc().strip().splitlines()
        print(f"  FAIL  {label}: {exc[-1]}")
        for line in exc[-4:-1]:
            print(f"          {line.strip()}")
        return False


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    print(f"probing: {path}")
    print(f"  exists: {path.exists()}   size: {path.stat().st_size if path.exists() else '-'}")

    attempt("raw read (first 1MB)", lambda: open(path, "rb").read(1 << 20))

    for mod in ("OpenEXR", "imageio", "tifffile", "cv2"):
        try:
            __import__(mod)
            print(f"  OK    import {mod}")
        except Exception as exc:
            print(f"  FAIL  import {mod}: {exc}")

    suffix = path.suffix.lower()
    if suffix == ".exr":
        def via_imageio():
            import imageio.v3 as iio
            return iio.imread(path)
        attempt("imageio.imread", via_imageio)

        def via_cv2():
            import cv2
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("cv2.imread returned None (EXR support is often "
                                   "disabled: set OPENCV_IO_ENABLE_OPENEXR=1)")
            return img
        attempt("cv2.imread", via_cv2)

    if suffix in (".tif", ".tiff"):
        def via_tifffile():
            import tifffile
            return tifffile.imread(path)
        attempt("tifffile.imread", via_tifffile)

    from training.prepare_training_data import read_exr, read_tif  # noqa: E402
    if suffix == ".exr":
        attempt("ingest read_exr", lambda: read_exr(path))
    elif suffix in (".tif", ".tiff"):
        attempt("ingest read_tif", lambda: read_tif(path)[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
