"""RUDRA Studio UI server -- static page plus a real inference backend.

Before 27 Aug 2026 this was ``SimpleHTTPRequestHandler`` and nothing else, and
the page in front of it was a mock: ``app.js`` contained zero network calls, the
metrics moved with the sliders through a comment that said "simulate", and the
header hard-coded "Backbone: Flux 1 Dev". Nothing you clicked touched a model.

Now it serves two endpoints beside the static files:

  GET  /api/model                what checkpoint is actually loaded
  GET  /api/checkpoints          the committed models this clone can load
  POST /api/frame                one forward pass -> raw fields, for the GPU
                                 compositor in ui/compositor.js
  POST /api/infer                run SDR2HDRNet on an uploaded image
  POST /api/sequence/open        open a shot BY PATH: a folder of frames, or
                                 a video file. Nothing is uploaded -- the
                                 server reads the footage where it sits.
  GET  /api/sequence/frame       one frame of an opened shot, in exactly the
                                 format /api/frame returns, so the page's
                                 existing decode and compositor are unchanged.

What the numbers mean, precisely, because the mock's did not:

  * MaxCLL / MaxFALL are REAL, computed by rudra.delivery.metadata on
    max(R,G,B) per CTA-861.3 -- the same code that writes the HDR10 sidecar.
  * peak / P99 nits, and the share of pixels above diffuse white, are measured
    on the prediction.
  * "headroom" is the model's peak over the analytic inverse-ACES baseline's,
    in stops: what the network added that the tone-map inverse could not.
  * There is NO LPIPS, JOD, PSNR or CVVDP here, and there cannot be: every one
    of those needs the ground-truth HDR, which does not exist for a file you
    just dragged in. Use training/sweep_inference.py on the held-out split for
    reference-based numbers.

Run it:  python ui/server.py [--checkpoint PATH] [--device cuda|cpu] [--port 8422]
Without --checkpoint it looks for the newest best.pt/shipped_*.pt under
E:/RUDRA_v3_20260822/checkpoints, then falls back to demo mode -- the page still
loads and says so, rather than lying.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import hashlib
import io
import json
import os
import re
import socketserver
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent
REPO = UI_DIR.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:                                            # run as `python ui/server.py`
    from sequence import Sequence, SequenceError
except ImportError:                             # imported as `ui.server`
    from ui.sequence import Sequence, SequenceError  # type: ignore[no-redef]

DIFFUSE_WHITE_NITS = 203.0
NETWORK_PEAK_NITS = 10_000.0

# Opened shots, by job id. In memory only: this is a local viewer, and a job
# that outlived a restart would point at a frame list nobody asked for.
SEQUENCES: dict[str, Sequence] = {}


def checkpoint_roots() -> tuple[Path, ...]:
    """Where to look for a model, in order of preference.

    Set RUDRA_CHECKPOINT_ROOTS to your own training output, os.pathsep
    separated, to have the newest run there win. This used to be one
    hard-coded absolute path on the machine the models were trained on,
    which meant nothing to anyone else who cloned the repo.

    The repo's own checkpoints/ is always last and always searched, so a
    fresh clone starts with a real model instead of falling back to demo
    mode.
    """
    env = os.environ.get("RUDRA_CHECKPOINT_ROOTS", "")
    roots = [Path(p) for p in env.split(os.pathsep) if p.strip()]
    if not roots:
        roots = [REPO / "hdrdata" / "checkpoints"]
    return (*roots, REPO / "checkpoints")

_state: dict = {"model": None, "info": {"loaded": False}, "lock": threading.Lock()}
# Extra checkpoints loaded on demand, keyed by path. Comparing two models used
# to mean restarting the server, which loses the page state and the terminal --
# and on 28 Aug 2026 meant the A/B of v3b against v4 could not be done at all
# without interrupting the person using it. Pass {"checkpoint": "..."} in
# X-Rudra-Params to score any checkpoint against the same input, in place.
_extra: dict = {}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
REGISTRY = REPO / "checkpoints" / "models.json"


def registry() -> dict:
    """The committed checkpoints and which of them is the default.

    Newest-file is a fine rule for one directory of training runs and a bad one
    for a directory of shipped models: it picks whatever was copied last, and
    one of the files in there is a temporal refiner that is not an SDR2HDRNet
    at all and raises on load. So the repo directory is described, not guessed.
    """
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
    except Exception:                                             # noqa: BLE001
        return {"default": None, "models": []}
    return {"default": data.get("default"),
            "models": [m for m in data.get("models", []) if isinstance(m, dict)]}


def loadable_models() -> list[dict]:
    """Registry entries the viewer can actually load, with resolved paths."""
    out = []
    for m in registry()["models"]:
        if m.get("kind") != "sdr2hdr":
            continue
        path = REPO / "checkpoints" / str(m.get("file", ""))
        if path.is_file():
            out.append(dict(m, path=str(path)))
    return out


def find_checkpoint(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        # A bare name is allowed if it is one of the committed models, so
        # `--checkpoint sdr2hdr_image_v5.pt` works from a fresh clone.
        named = REPO / "checkpoints" / path.name
        return named if named.is_file() else None
    # First root that has anything wins, and the newest file inside it wins
    # from there. Pooling every root and taking the newest mtime, which is
    # what this used to do, made the order above decorative: a fresh clone
    # writes the committed checkpoints with a current mtime, so the repo
    # copy beat the local training run it was supposed to defer to.
    for root in checkpoint_roots():
        if not root.is_dir():
            continue
        if root == REPO / "checkpoints":
            # Described, not guessed. One of the files in here is a temporal
            # refiner that is not an SDR2HDRNet and raises on load, so fall
            # through to the registry listing only if the default is missing.
            default = registry()["default"]
            if default and (root / default).is_file():
                return root / default
            loadable = [Path(m["path"]) for m in loadable_models()]
            if loadable:
                return loadable[0]
            continue
        # A training run writes <root>/<run>/best.pt or <root>/<run>/shipped_*.pt.
        candidates = list(root.glob("*/shipped_*.pt")) + list(root.glob("*/best.pt"))
        if not candidates:
            continue
        # Prefer an explicitly shipped checkpoint, then the most recent.
        shipped = [c for c in candidates if c.name.startswith("shipped_")]
        return max(shipped or candidates, key=lambda p: p.stat().st_mtime)
    return None


def load_model(checkpoint: Path | None, device_name: str):
    import torch

    from rudra.sdr2hdr import SDR2HDRNet

    if checkpoint is None:
        return None, {"loaded": False, "reason": "no checkpoint found",
                      "device": device_name, "demo": True}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {}) or {}
    model = SDR2HDRNet.from_config(config)
    model.load_state_dict(payload.get("model", payload), strict=True)
    device = torch.device(device_name)
    model.to(device).eval()
    info = {
        "loaded": True,
        "demo": False,
        "checkpoint": str(checkpoint),
        "name": checkpoint.parent.name + "/" + checkpoint.name,
        "step": payload.get("step"),
        "base_channels": int(config.get("base_channels", 32)),
        "max_hdr": float(getattr(model, "max_hdr", 4.0)),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "preserve_outside": True,
    }
    return model, info


def model_for(path: str, args):
    """A second (third, ...) checkpoint, loaded once and kept."""
    key = str(Path(path).resolve())
    with _state["lock"]:
        if key in _extra:
            return _extra[key]
        model, info = load_model(Path(key), args.device)
        _extra[key] = (model, info)
        print(f"  loaded extra checkpoint: {json.dumps(info)}")
        return model, info


def ensure_model(args):
    with _state["lock"]:
        if _state["model"] is not None or _state["info"].get("demo"):
            return _state["model"], _state["info"]
        try:
            checkpoint = find_checkpoint(args.checkpoint)
            model, info = load_model(checkpoint, args.device)
        except Exception as exc:  # torch missing, bad checkpoint, no CUDA...
            traceback.print_exc()
            model, info = None, {"loaded": False, "demo": True,
                                 "reason": f"{type(exc).__name__}: {exc}"}
        _state["model"], _state["info"] = model, info
        print(f"  model: {json.dumps(info)}")
        return model, info


# ---------------------------------------------------------------------------
# imaging
# ---------------------------------------------------------------------------
def _srgb_encode(linear):
    import numpy as np

    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def png_data_url(rgb_float) -> str:
    """[0,1] float RGB -> a data: URL the <img> tags can take directly."""
    import numpy as np
    from PIL import Image

    arr = (np.clip(rgb_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def display_map(hdr_chw, display_nits: float):
    """HDR in network units -> an SDR-monitor preview.

    Straight exposure and clip, deliberately: any tone curve here would hide
    the thing the page exists to show. Raising display_nits pulls the clip
    point up, so highlights the network reconstructed stop blowing out while
    the rest of the frame darkens. That reveal IS the comparison.
    """
    import numpy as np

    hwc = np.transpose(hdr_chw, (1, 2, 0))
    scale = NETWORK_PEAK_NITS / max(display_nits, 1e-3)
    return _srgb_encode(np.clip(hwc * scale, 0.0, 1.0))


def measure(hdr_chw, baseline_chw, highlight_mask, shadow_mask) -> dict:
    import numpy as np

    from rudra.delivery import metadata as dm

    hwc = np.transpose(hdr_chw, (1, 2, 0)).astype(np.float64)
    nits = hwc * NETWORK_PEAK_NITS
    base_nits = np.transpose(baseline_chw, (1, 2, 0)).astype(np.float64) * NETWORK_PEAK_NITS

    # analyze_frame takes ABSOLUTE NITS, not scene-linear -- passing
    # nits/203 here reported a 4,000-nit specular as MaxCLL 20.
    stats = dm.analyze_frame(nits, index=0)
    maxcll, maxfall = dm.maxcll_maxfall([stats])

    peak = float(nits.max())
    base_peak = float(base_nits.max())
    # Peak-vs-peak, and it is the WEAKEST number here. Any large blown-white
    # region lands both the model and the baseline on inverse-ACES's ceiling
    # for a saturated pixel (2,551.9 nits), so this reads 0.00 on ordinary
    # graded footage even while the model is working hard elsewhere -- exactly
    # what the 27 Aug 2026 UI test saw on the demo sunset, where the same model
    # lifted a small specular by 2.47 stops on a frame without the plateau.
    headroom = float(np.log2(max(peak, 1e-6) / max(base_peak, 1e-6)))

    # What the model did where it was ALLOWED to act. The residual is gated to
    # the highlight/shadow masks, so this is the honest "is it doing anything".
    hi = np.asarray(highlight_mask) > 0.5
    sh = np.asarray(shadow_mask) > 0.5
    luma = nits.max(axis=2)
    base_luma = base_nits.max(axis=2)

    def _stops_in(region) -> float:
        if not region.any():
            return float("nan")
        return float(np.log2(max(luma[region].mean(), 1e-6)
                             / max(base_luma[region].mean(), 1e-6)))

    # One number that separates "did nothing" from "changed a lot": the RMS
    # deviation from the baseline in stops, over the whole frame.
    ratio = np.log2((luma + 1e-4) / (base_luma + 1e-4))
    highlight, shadow = float(hi.mean()), float(sh.mean())
    return {
        "maxcll": round(maxcll, 1),
        "maxfall": round(maxfall, 1),
        "peak_nits": round(peak, 1),
        "baseline_peak_nits": round(base_peak, 1),
        "headroom_stops": round(headroom, 2),
        "headroom_highlight_stops": round(_stops_in(hi), 2),
        "headroom_shadow_stops": round(_stops_in(sh), 2),
        "departure_rms_stops": round(float(np.sqrt((ratio ** 2).mean())), 3),
        "p99_nits": round(float(np.percentile(nits, 99)), 1),
        "median_nits": round(float(np.median(nits)), 2),
        # 1e-6 relative slack: a pixel stored as exactly diffuse white comes
        # back as 203.0000076 through float32, and counted as "above" it.
        "above_diffuse_white_pct": round(
            100.0 * float((nits > DIFFUSE_WHITE_NITS * (1.0 + 1e-6)).mean()), 2),
        "above_1000_nits_pct": round(100.0 * float((nits > 1000.0).mean()), 3),
        "highlight_mask_pct": round(100.0 * float(highlight), 2),
        "shadow_mask_pct": round(100.0 * float(shadow), 2),
    }


SCOPE_LO_NITS = 0.05
SCOPE_HI_NITS = 4000.0


def scopes(hdr_chw, columns: int = 230, bins: int = 76) -> dict:
    """Waveform envelope and log2 histogram of the actual prediction.

    Sent as numbers, drawn as SVG by the page. A scope that is drawn rather
    than measured is decoration, and this one has to be trustworthy: it is
    what a colourist reads the grade off.
    """
    import numpy as np

    nits = np.transpose(hdr_chw, (1, 2, 0)).astype(np.float64) * NETWORK_PEAK_NITS
    luma = 0.2627 * nits[..., 0] + 0.6780 * nits[..., 1] + 0.0593 * nits[..., 2]
    luma = np.clip(luma, SCOPE_LO_NITS, SCOPE_HI_NITS)

    edges = np.linspace(0, luma.shape[1], columns + 1).astype(int)
    lo, hi = [], []
    mid, q1, q3 = [], [], []
    for i in range(columns):
        col = luma[:, edges[i]:max(edges[i] + 1, edges[i + 1])].ravel()
        a, b, c, d, e = np.percentile(col, [2, 25, 50, 75, 98])
        lo.append(a); q1.append(b); mid.append(c); q3.append(d); hi.append(e)

    span = (np.log10(SCOPE_HI_NITS / SCOPE_LO_NITS))

    def norm(values):
        # 0 at the floor, 1 at the ceiling, on the same log scale as the ladder
        return [round(float(np.log10(v / SCOPE_LO_NITS) / span), 4) for v in values]

    counts, _ = np.histogram(np.log2(luma), bins=bins,
                             range=(np.log2(SCOPE_LO_NITS), np.log2(SCOPE_HI_NITS)))
    peak = max(int(counts.max()), 1)
    return {
        "lo": norm(lo), "q1": norm(q1), "mid": norm(mid), "q3": norm(q3), "hi": norm(hi),
        "histogram": [round(float(c) / peak, 4) for c in counts],
        "floor_nits": SCOPE_LO_NITS, "ceiling_nits": SCOPE_HI_NITS,
    }


def run_inference(model, image_bytes: bytes, params: dict, args) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    from training.infer_sdr2hdr import predict_image

    started = time.time()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_side = int(params.get("max_side", 1600))
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * ratio)),
                              max(1, int(image.height * ratio))), Image.LANCZOS)
    sdr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None].to(next(model.parameters()).device)

    strength = float(params.get("strength", 1.0))
    mode = params.get("recovery_mode", "all")
    preserve = bool(params.get("preserve_outside", True))
    display_nits = float(params.get("display_nits", DIFFUSE_WHITE_NITS))

    hdr = predict_image(model, tensor, preserve_outside=preserve,
                        tile_size=int(params.get("tile_size", 512)),
                        overlap=int(params.get("tile_overlap", 64)),
                        recovery_mode=mode, recovery_strength=strength)
    with torch.inference_mode():
        probe = model(tensor, preserve_outside=preserve, recovery_mode=mode,
                      residual_strength=strength)
        baseline = probe.baseline.float()
        highlight = probe.highlight_mask[0, 0].float().cpu().numpy()
        shadow = probe.shadow_mask[0, 0].float().cpu().numpy()

    hdr_np = hdr[0].cpu().numpy()
    base_np = baseline[0].cpu().numpy()
    metrics = measure(hdr_np, base_np, highlight, shadow)
    metrics["elapsed_s"] = round(time.time() - started, 2)
    metrics["source_resolution"] = f"{image.width}x{image.height}"
    metrics["resolution"] = f"{image.width}x{image.height}"
    metrics["display_nits"] = round(display_nits, 1)
    return {
        "ok": True,
        "sdr_png": png_data_url(sdr),
        "hdr_png": png_data_url(display_map(hdr_np, display_nits)),
        "baseline_png": png_data_url(display_map(base_np, display_nits)),
        "metrics": metrics,
        "scopes": scopes(hdr_np),
    }


MAX_FIELD_MAGNITUDE = 64.0


def run_frame(model, image_bytes: bytes, params: dict, args) -> tuple[dict, bytes]:
    """One forward pass. The client composes; this only ships the fields.

    /api/infer composes server-side and returns two PNGs, which means every
    move of the strength or display-peak slider costs an upload, a forward
    pass and a re-encode. The fields the head produces -- log residual and the
    two masks -- do not depend on recovery_mode, residual_strength or
    preserve_outside, so sending them once lets the page rebuild every
    combination on its own GPU. That is the difference between a control that
    responds in a second and one that responds in a frame.

    Body layout, little-endian, offsets in the header:
        fields   H*W*4 float16   log residual RGB, highlight mask in alpha
        shadow   H*W   float16   shadow mask
        sdr      H*W*3 uint8     exactly the pixels the network was given
    The SDR travels back because the page must compute the analytic baseline
    from the same pixels the network saw, not from its own resize of the file.
    """
    import numpy as np
    import torch
    from PIL import Image

    from training.infer_sdr2hdr import predict_fields

    started = time.time()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    source = f"{image.width}x{image.height}"
    max_side = int(params.get("max_side", 1600))
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize((max(1, int(image.width * ratio)),
                              max(1, int(image.height * ratio))), Image.LANCZOS)

    sdr_u8 = np.asarray(image, dtype=np.uint8)
    sdr = sdr_u8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None].to(next(model.parameters()).device)

    tile_size = int(params.get("tile_size", 0))
    overlap = int(params.get("tile_overlap", 64))
    try:
        fields = predict_fields(model, tensor, tile_size=tile_size, overlap=overlap)
    except Exception as exc:                       # noqa: BLE001 - re-raised below
        if "out of memory" not in str(exc).lower():
            raise
        # An untiled pass is preferred because it makes the viewer and the
        # master agree exactly; falling back beats failing.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fields = predict_fields(model, tensor, tile_size=512, overlap=overlap)

    # The learned shadow weight is a whole-frame judgement, so it is computed
    # once from the whole frame and travels in the header as a scalar. It CANNOT
    # be folded into the residual the way residual_scale is: it scales the
    # shadow prior inside a max(), which is not linear in the residual.
    shadow_weight = 1.0
    if getattr(model, "shadow_gate", None) is not None:
        shadow_weight = float(model.predict_shadow_weight(tensor).flatten()[0])

    residual = fields["residual"][0].permute(1, 2, 0).cpu().numpy()
    highlight = fields["highlight"][0].permute(1, 2, 0).cpu().numpy()
    shadow = fields["shadow"][0, 0].cpu().numpy()

    def _half(a):
        # log residual is added in the log domain and then clamped to
        # log1p(max_hdr * log_scale) = 4.17, so anything past a few units is
        # already saturating; +/-64 keeps every value that can matter inside
        # half float with room to spare.
        return np.clip(np.nan_to_num(a, nan=0.0, posinf=MAX_FIELD_MAGNITUDE,
                                     neginf=-MAX_FIELD_MAGNITUDE),
                       -MAX_FIELD_MAGNITUDE, MAX_FIELD_MAGNITUDE).astype(np.float16)

    packed = _half(np.concatenate([residual, highlight], axis=-1))
    shadow16 = _half(shadow)
    body = packed.tobytes() + shadow16.tobytes() + sdr_u8.tobytes()

    height, width = sdr_u8.shape[:2]
    header = {
        "ok": True,
        "width": width,
        "height": height,
        "tiled": bool(fields["tiled"]),
        "shadow_weight": shadow_weight,
        "log_scale": float(getattr(model, "log_scale", 16.0)),
        "max_hdr": float(getattr(model, "max_hdr", 4.0)),
        "peak_nits": NETWORK_PEAK_NITS,
        "diffuse_white_nits": DIFFUSE_WHITE_NITS,
        "source_resolution": source,
        "resolution": f"{width}x{height}",
        "elapsed_s": round(time.time() - started, 3),
        "offsets": {
            "fields": 0,
            "shadow": width * height * 4 * 2,
            "sdr": width * height * 4 * 2 + width * height * 2,
            "total": len(body),
        },
    }
    return header, body


# ---------------------------------------------------------------------------
# mastering
# ---------------------------------------------------------------------------
MASTER_DIR = UI_DIR / "_masters"
SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")


def run_master(model, image_bytes: bytes, params: dict, args) -> dict:
    """Reconstruct at full resolution and write a real EXR master.

    Scene-linear, diffuse white = 1.0, which is the convention every other
    part of RUDRA already speaks (pipeline/hdr_io.py, rudra/normalization.py).
    The ACES container additionally converts to AP0 and stamps the ST 2065-4
    chromaticities, so the file lands in Resolve or Nuke as an ACES image
    rather than as untagged floats.
    """
    import numpy as np
    import torch
    from PIL import Image

    from rudra.delivery import metadata as dm
    from rudra.delivery.aces import write_aces_exr
    from rudra.delivery.controls import DEFAULT_REGION_BANDS, apply_region_ev
    from rudra.delivery.exr import write_exr
    from training.infer_sdr2hdr import predict_image

    started = time.time()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Full resolution: a master is the one output that must not be downsampled.
    limit = int(params.get("master_max_side", 4096))
    if max(image.size) > limit:
        ratio = limit / max(image.size)
        image = image.resize((max(1, int(image.width * ratio)),
                              max(1, int(image.height * ratio))), Image.LANCZOS)

    sdr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None].to(next(model.parameters()).device)

    def _predict(tile_size: int):
        return predict_image(model, tensor,
                             preserve_outside=bool(params.get("preserve_outside", True)),
                             tile_size=tile_size,
                             overlap=int(params.get("tile_overlap", 64)),
                             recovery_mode=params.get("recovery_mode", "all"),
                             recovery_strength=float(params.get("strength", 1.0)))

    # Untiled by default, exactly like /api/frame. The viewer composes from an
    # untiled pass, and tiling changes the answer inside the overlap bands --
    # on 28 Aug 2026 that had the page reporting MaxCLL 30,254 for a master
    # whose sidecar said 29,809, which is the kind of discrepancy nobody
    # notices until a QC report does.
    tile_size = int(params.get("tile_size", 0))
    try:
        hdr = _predict(tile_size)
    except Exception as exc:                        # noqa: BLE001 - re-raised below
        if "out of memory" not in str(exc).lower():
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tile_size = 512
        hdr = _predict(tile_size)

    network = hdr[0].cpu().numpy()                      # nits / 10,000
    nits = np.transpose(network, (1, 2, 0)).astype(np.float64) * NETWORK_PEAK_NITS

    # Region EV is a grade, not a preview: the picture that was approved in
    # the viewer is the picture that has to land in the file.
    regions = params.get("regions") or list(DEFAULT_REGION_BANDS)
    softness = float(params.get("region_softness_stops", 1.0))
    graded = any(float(b.get("ev", 0.0)) for b in regions)
    if graded:
        ceiling = float(getattr(model, "max_hdr", 4.0)) * NETWORK_PEAK_NITS
        nits = np.clip(apply_region_ev(nits, regions, softness), 0.0, ceiling)

    scene_linear = (nits / DIFFUSE_WHITE_NITS).astype(np.float32)

    stats = dm.analyze_frame(nits, index=0)
    maxcll, maxfall = dm.maxcll_maxfall([stats])

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    stem = SAFE_STEM.sub("_", Path(str(params.get("name", "frame"))).stem)[:60] or "frame"
    container = params.get("container", "aces")
    provenance = {
        "rudra:checkpoint": str(params.get("checkpoint", "")),
        "rudra:maxCLL": f"{maxcll}",
        "rudra:maxFALL": f"{maxfall}",
        "rudra:recoveryMode": str(params.get("recovery_mode", "all")),
        "rudra:preserveOutside": str(bool(params.get("preserve_outside", True))),
        "rudra:regionEV": json.dumps(regions) if graded else "neutral",
        "rudra:tiled": str(bool(tile_size)),
    }
    if container == "aces":
        out = MASTER_DIR / f"{stem}_rudra_aces.exr"
        # HALF tops out near 65,504; scene-linear here is nits/203, so a
        # 1,000,000-nit sun is ~4,926 -- comfortably inside. Keep half.
        write_aces_exr(scene_linear, out, source_space="rec2020", provenance=provenance)
    else:
        out = MASTER_DIR / f"{stem}_rudra_linear.exr"
        write_exr(out, scene_linear, half=True, attributes=provenance)

    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "maxcll_nits": maxcll, "maxfall_nits": maxfall,
        "peak_nits": round(float(nits.max()), 1),
        "resolution": [image.width, image.height],
        "container": "ACES 2065-1 (AP0)" if container == "aces" else "scene-linear Rec.2020",
        "transfer": "linear", "diffuse_white_nits": DIFFUSE_WHITE_NITS,
        "checkpoint": params.get("checkpoint", ""),
        "recovery_mode": params.get("recovery_mode", "all"),
        "residual_strength": float(params.get("strength", 1.0)),
        "preserve_outside": bool(params.get("preserve_outside", True)),
        "region_ev": regions if graded else None,
        "region_softness_stops": softness if graded else None,
        "tiled": bool(tile_size),
    }, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "file": out.name,
        "path": str(out),
        "bytes": out.stat().st_size,
        "sidecar": sidecar.name,
        "maxcll": maxcll,
        "maxfall": maxfall,
        "peak_nits": round(float(nits.max()), 1),
        "resolution": f"{image.width}x{image.height}",
        "container": "ACES 2065-1" if container == "aces" else "Linear Rec.2020",
        "graded": graded,
        "tiled": bool(tile_size),
        "elapsed_s": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# http
# ---------------------------------------------------------------------------
def make_handler(args):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(UI_DIR), **kw)

        def end_headers(self):
            # The page and its scripts are edited live during development, and a
            # cached live.js silently keeps running the previous build -- on
            # 28 Aug 2026 that made a fixed handler look broken for two rounds.
            if (self.path or "").split("?")[0].endswith((".js", ".css", ".html", "/")):
                self.send_header("Cache-Control", "no-store, must-revalidate")
            super().end_headers()

        def log_message(self, fmt, *a):  # quieter than the default
            if "/api/" in (self.path or ""):
                super().log_message(fmt, *a)

        def _binary(self, header: dict, body: bytes) -> None:
            """JSON in a header, floats in the body.

            Base64 in JSON would inflate a 14 MB frame to 19 MB and cost a
            decode on both ends, for data the GPU wants as raw bytes anyway.
            """
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Rudra-Frame", json.dumps(header))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _sequence_bytes(self, params: dict) -> bytes:
            """Frame bytes for a request that names a shot instead of sending one.

            Master and infer POST the File object the user dropped. A frame of
            an opened shot has no File object behind it -- the footage is on
            this machine -- so those requests arrive with an empty body and
            seq_job/seq_index in the params instead. Without this, Master EXR
            on an opened plate failed with "empty body", which is a strange
            way to learn that the plate you are looking at cannot be delivered.
            """
            job = str(params.get("seq_job") or "")
            if not job:
                return b""
            sequence = SEQUENCES.get(job)
            if sequence is None:
                raise SequenceError("that shot is not open any more; open it again")
            return sequence.frame_bytes(int(params.get("seq_index", 0)))

        def _sequence_frame(self):
            """One frame of an opened shot, as /api/frame would return it."""
            from urllib.parse import parse_qs, urlparse

            query = parse_qs(urlparse(self.path).query)
            job = (query.get("job") or [""])[0]
            sequence = SEQUENCES.get(job)
            if sequence is None:
                return self._json({"ok": False,
                                   "error": "that shot is not open any more; "
                                            "open it again"}, status=404)
            try:
                index = int((query.get("i") or ["0"])[0])
            except ValueError:
                return self._json({"ok": False, "error": "bad frame index"}, status=400)

            model, info = ensure_model(args)
            if model is None:
                return self._json({"ok": False, "demo": True,
                                   "error": "No model loaded: "
                                            + str(info.get("reason", "unknown"))},
                                  status=503)
            try:
                params = json.loads(self.headers.get("X-Rudra-Params", "{}") or "{}")
                requested = params.get("checkpoint")
                info_used = info
                if requested and Path(requested).exists() and Path(requested).resolve() \
                        != Path(str(info.get("checkpoint", ""))).resolve():
                    model, info_used = model_for(requested, args)
                params.setdefault("checkpoint_name", info_used.get("name"))
                header, body = run_frame(model, sequence.frame_bytes(index), params, args)
                header["checkpoint"] = (info_used.get("name")
                                        or info_used.get("checkpoint"))
                header["step"] = info_used.get("step")
                header["gpu"] = info_used.get("gpu")
                header["frame_index"] = index
                header["frame_name"] = sequence.name_of(index)
                return self._binary(header, body)
            except SequenceError as exc:
                return self._json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                return self._json({"ok": False,
                                   "error": f"{type(exc).__name__}: {exc}"}, status=500)

        def _sequence_open(self):
            """Open a shot by path. The body is JSON: {"path": "..."}."""
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                wanted = (json.loads(raw.decode("utf-8")) or {}).get("path", "")
                sequence = Sequence.open(wanted)
            except SequenceError as exc:
                # A path the user can fix, not a server fault.
                return self._json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                return self._json({"ok": False,
                                   "error": f"{type(exc).__name__}: {exc}"}, status=500)

            job = f"seq{len(SEQUENCES) + 1}_{int(time.time())}"
            SEQUENCES[job] = sequence
            # One shot at a time is how the page uses this; holding older jobs
            # would keep their extraction caches alive for a session.
            for stale in [k for k in SEQUENCES if k != job]:
                SEQUENCES.pop(stale, None)
            payload = {"ok": True, "job": job}
            payload.update(sequence.describe())
            print(f"   opened {payload['kind']} {payload['name']} "
                  f"-- {payload['count']} frame(s)")
            return self._json(payload)

        def do_GET(self):
            if self.path.startswith("/api/sequence/frame"):
                return self._sequence_frame()
            if self.path.startswith("/api/checkpoints"):
                return self._json({
                    "default": registry()["default"],
                    "models": [{k: v for k, v in m.items() if k != "path"}
                               for m in loadable_models()],
                })
            if self.path.startswith("/api/model"):
                _, info = ensure_model(args)
                return self._json(info)
            if self.path.startswith("/api/master/download"):
                return self._send_master()
            if self.path.split("?")[0] in ("/", "/index.html"):
                return self._send_index()
            return super().do_GET()

        def _send_index(self):
            """index.html with its asset URLs stamped from the files on disk.

            The page used to ask for `style.css?v=4`, a URL that says nothing
            about the file behind it. On 28 Aug 2026 a stylesheet another
            project had left in Chrome's cache for localhost:8080 answered
            that request instead, and RUDRA Studio rendered with someone
            else's CSS -- no error, no failed request, just a page wearing
            the wrong clothes. Cache-Control cannot help there: the browser
            never asks the server for a URL it believes it already has.

            Stamping each asset with its own mtime and size means the URL
            changes whenever the file does, so a stale entry can only ever be
            answered for content that really is identical.
            """
            body = (UI_DIR / "index.html").read_bytes().decode("utf-8")
            for name in ("style.css", "app.js", "compositor.js"):
                stat = (UI_DIR / name).stat()
                token = hashlib.sha1(
                    f"{name}:{int(stat.st_mtime)}:{stat.st_size}".encode()
                ).hexdigest()[:10]
                # Stamp whether or not the markup already carries a query, so
                # a hand-edited index.html cannot quietly opt out of this.
                body = re.sub(rf"{re.escape(name)}(\?[^\"\']*)?",
                              f"{name}?v={token}", body)
            payload = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            # end_headers() adds Cache-Control for html paths; don't duplicate it.
            self.end_headers()
            self.wfile.write(payload)

        def _send_master(self):
            """Hand back one file from _masters, by name only."""
            from urllib.parse import parse_qs, urlparse

            wanted = (parse_qs(urlparse(self.path).query).get("f") or [""])[0]
            # Name only: no separators, no traversal, and it must already exist
            # in the directory we wrote it to.
            if not wanted or "/" in wanted or "\\" in wanted or wanted != Path(wanted).name:
                return self.send_error(400, "bad file")
            target = (MASTER_DIR / wanted).resolve()
            if target.parent != MASTER_DIR.resolve() or not target.is_file():
                return self.send_error(404, "no such master")
            payload = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Disposition",
                             f'attachment; filename="{target.name}"')
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            if self.path.startswith("/api/sequence/open"):
                return self._sequence_open()
            if not self.path.startswith(("/api/infer", "/api/master", "/api/frame")):
                return self.send_error(404, "no such endpoint")
            model, info = ensure_model(args)
            if model is None:
                return self._json(
                    {"ok": False,
                     "error": "No model loaded: " + str(info.get("reason", "unknown")),
                     "demo": True}, status=503)
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length > 0 else b""
                params = json.loads(self.headers.get("X-Rudra-Params", "{}") or "{}")
                if not raw:
                    raw = self._sequence_bytes(params)
                if not raw:
                    return self._json({"ok": False, "error": "empty body"}, status=400)
                requested = params.get("checkpoint")
                info_used = info
                if requested and Path(requested).resolve() != Path(
                        str(info.get("checkpoint", ""))).resolve():
                    if not Path(requested).exists():
                        return self._json({"ok": False,
                                           "error": f"checkpoint not found: {requested}"},
                                          status=400)
                    model, info_used = model_for(requested, args)
                params.setdefault("checkpoint_name", info_used.get("name"))
                if self.path.startswith("/api/frame"):
                    header, body = run_frame(model, raw, params, args)
                    header["checkpoint"] = (info_used.get("name")
                                            or info_used.get("checkpoint"))
                    header["step"] = info_used.get("step")
                    header["gpu"] = info_used.get("gpu")
                    return self._binary(header, body)
                if self.path.startswith("/api/master"):
                    payload = run_master(model, raw, params, args)
                else:
                    payload = run_inference(model, raw, params, args)
                payload["checkpoint"] = info_used.get("name") or info_used.get("checkpoint")
                payload["step"] = info_used.get("step")
                return self._json(payload)
            except SequenceError as exc:
                # Something the user can fix, not a server fault.
                return self._json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                traceback.print_exc()
                return self._json({"ok": False,
                                   "error": f"{type(exc).__name__}: {exc}"}, status=500)

    return Handler


class ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=os.environ.get("RUDRA_CHECKPOINT"))
    parser.add_argument("--device", default=os.environ.get("RUDRA_DEVICE", "cuda"))
    parser.add_argument("--port", type=int, default=8422)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--preload", action="store_true",
                        help="Load the model at startup instead of on the first request")
    args = parser.parse_args()

    print("\n" + "=" * 66)
    print("   RUDRA Studio  --  live inference UI")
    print("=" * 66)
    print(f"   http://localhost:{args.port}")
    found = find_checkpoint(args.checkpoint)
    print(f"   checkpoint : {found if found else 'NONE FOUND -- page will run in demo mode'}")
    print(f"   device     : {args.device}")
    print("=" * 66 + "\n")

    if args.preload:
        ensure_model(args)

    if not args.no_browser:
        threading.Thread(
            target=lambda: (time.sleep(0.6),
                            webbrowser.open(f"http://localhost:{args.port}")),
            daemon=True).start()

    with ThreadedServer(("", args.port), make_handler(args)) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down RUDRA Studio. Goodbye!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
