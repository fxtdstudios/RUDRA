"""Enforce the native dependency rule (docs/NATIVE_ARCHITECTURE.md section 2).

A layer may include only headers from layers below it, and third-party
headers appear only where the architecture puts them:

    platform < core < {infer, media, render, deliver} < engine < app
    cli: core, infer, media, deliver (never Qt, never render)

    python tools/check_layers.py           # exit 1 and list violations
"""
from __future__ import annotations

import re
from pathlib import Path

NATIVE = Path(__file__).resolve().parents[1] / "native"
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)

ORDER = {"platform": 0, "core": 1, "infer": 2, "media": 2, "render": 2, "deliver": 2, "engine": 3, "app": 4}
CLI_ALLOWED = {"platform", "core", "infer", "media", "deliver"}

QT = re.compile(r"^(Q[A-Z]\w*|Qt\w*/|rhi/|private/qrhi)")
TORCH = re.compile(r"^(torch/|ATen/|c10/)")
ORT = re.compile(r"^onnxruntime")
GPU = re.compile(r"^(vulkan/|d3d1[12]|dxgi|Metal/|QuartzCore/|GL/|OpenGL/|GLES)")
LIBAV = re.compile(r"^lib(av|sw)\w*/")
OPENCV = re.compile(r"^opencv2?/")


def layer_of(path: Path) -> str:
    return path.relative_to(NATIVE).parts[0]


def violations() -> list[str]:
    out: list[str] = []
    for f in sorted(NATIVE.rglob("*")):
        if f.suffix not in {".hpp", ".h", ".cpp", ".cc", ".mm"} or "build" in f.parts:
            continue
        layer = layer_of(f)
        if layer in {"tests", "cmake"}:
            continue
        rel = f.relative_to(NATIVE)
        is_backend_src = layer == "infer" and f.parent.name == "src" and f.stem.endswith("_backend")
        for inc in INCLUDE.findall(f.read_text(encoding="utf-8", errors="replace")):
            if inc.startswith("rudra/"):
                target = inc.split("/")[1]
                if layer == "cli":
                    if target not in CLI_ALLOWED:
                        out.append(f"{rel}: cli may not include rudra/{target}")
                elif target in ORDER and target != layer and ORDER[target] >= ORDER[layer]:
                    out.append(f"{rel}: {layer} may not include rudra/{target} (same or higher layer)")
                continue
            if QT.match(inc) and layer not in {"render", "app"}:
                out.append(f"{rel}: Qt header <{inc}> outside render/ and app/")
            if inc.startswith(("rhi/", "private/qrhi")) and layer != "render":
                out.append(f"{rel}: QRhi header <{inc}> outside render/")
            if (TORCH.match(inc) or ORT.match(inc)) and not is_backend_src:
                out.append(f"{rel}: inference runtime header <{inc}> outside infer/src/*_backend.cpp")
            if GPU.match(inc) and layer != "render":
                out.append(f"{rel}: GPU API header <{inc}> outside render/")
            if LIBAV.match(inc) and layer not in {"media", "deliver"}:
                out.append(f"{rel}: FFmpeg header <{inc}> outside media/ and deliver/")
            if OPENCV.match(inc) and not (layer in {"media", "deliver"} and f.parent.name == "src"):
                out.append(f"{rel}: OpenCV header <{inc}> outside media/src and deliver/src")
    return out


def main() -> int:
    bad = violations()
    for v in bad:
        print(v)
    print(f"{'FAIL' if bad else 'ok'}: {len(bad)} layering violation(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
