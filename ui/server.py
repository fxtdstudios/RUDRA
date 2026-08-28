"""RUDRA Studio UI server -- static page plus a real inference backend.

Before 27 Aug 2026 this was ``SimpleHTTPRequestHandler`` and nothing else, and
the page in front of it was a mock: ``app.js`` contained zero network calls, the
metrics moved with the sliders through a comment that said "simulate", and the
header hard-coded "Backbone: Flux 1 Dev". Nothing you clicked touched a model.

Now it serves two endpoints beside the static files:

  GET  /api/model                what checkpoint is actually loaded
  POST /api/infer                run SDR2HDRNet on an uploaded image

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

Run it:  python ui/server.py [--checkpoint PATH] [--device cuda|cpu] [--port 8080]
Without --checkpoint it looks for the newest best.pt/shipped_*.pt under
E:/RUDRA_v3_20260822/checkpoints, then falls back to demo mode -- the page still
loads and says so, rather than lying.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import os
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

DIFFUSE_WHITE_NITS = 203.0
NETWORK_PEAK_NITS = 10_000.0
DEFAULT_CHECKPOINT_ROOTS = (
    Path("E:/RUDRA_v3_20260822/checkpoints"),
    REPO / "hdrdata" / "checkpoints",
)

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
def find_checkpoint(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    candidates: list[Path] = []
    for root in DEFAULT_CHECKPOINT_ROOTS:
        if root.is_dir():
            candidates += list(root.glob("*/shipped_*.pt")) + list(root.glob("*/best.pt"))
    if not candidates:
        return None
    # Prefer an explicitly shipped checkpoint, then the most recent.
    shipped = [c for c in candidates if c.name.startswith("shipped_")]
    pool = shipped or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def load_model(checkpoint: Path | None, device_name: str):
    import torch

    from rudra.sdr2hdr import SDR2HDRNet

    if checkpoint is None:
        return None, {"loaded": False, "reason": "no checkpoint found",
                      "device": device_name, "demo": True}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {}) or {}
    model = SDR2HDRNet(base_channels=int(config.get("base_channels", 32)))
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
    metrics["resolution"] = f"{image.width}x{image.height}"
    metrics["display_nits"] = round(display_nits, 1)
    return {
        "ok": True,
        "sdr_png": png_data_url(sdr),
        "hdr_png": png_data_url(display_map(hdr_np, display_nits)),
        "baseline_png": png_data_url(display_map(base_np, display_nits)),
        "metrics": metrics,
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

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/model"):
                _, info = ensure_model(args)
                return self._json(info)
            return super().do_GET()

        def do_POST(self):
            if not self.path.startswith("/api/infer"):
                return self.send_error(404, "no such endpoint")
            model, info = ensure_model(args)
            if model is None:
                return self._json(
                    {"ok": False,
                     "error": "No model loaded: " + str(info.get("reason", "unknown")),
                     "demo": True}, status=503)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length <= 0:
                    return self._json({"ok": False, "error": "empty body"}, status=400)
                raw = self.rfile.read(length)
                params = json.loads(self.headers.get("X-Rudra-Params", "{}") or "{}")
                requested = params.get("checkpoint")
                info_used = info
                if requested and Path(requested).resolve() != Path(
                        str(info.get("checkpoint", ""))).resolve():
                    if not Path(requested).exists():
                        return self._json({"ok": False,
                                           "error": f"checkpoint not found: {requested}"},
                                          status=400)
                    model, info_used = model_for(requested, args)
                payload = run_inference(model, raw, params, args)
                payload["checkpoint"] = info_used.get("name") or info_used.get("checkpoint")
                payload["step"] = info_used.get("step")
                return self._json(payload)
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
    parser.add_argument("--port", type=int, default=8080)
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
