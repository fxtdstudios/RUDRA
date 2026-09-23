"""Standalone SDR image/video to scene-linear HDR inference.

Outputs float32 RGB TIFF masters (unclipped), optional 16-bit normalized PNG,
and an 8-bit tone-mapped preview.  Video inputs are written as frame sequences
to avoid silently discarding HDR precision through an SDR video codec.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rudra.sdr2hdr import SDR2HDRNet, TemporalHDRRefiner, canonicalize_sdr, linear_to_srgb  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".mxf", ".webm"}


def load_models(image_checkpoint: str, temporal_checkpoint: str | None, device: torch.device):
    checkpoint = torch.load(image_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    image_model = SDR2HDRNet.from_config(config)
    image_model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    image_model.to(device).eval()
    temporal = None
    if temporal_checkpoint:
        temporal_ckpt = torch.load(temporal_checkpoint, map_location="cpu", weights_only=False)
        temporal_config = temporal_ckpt.get("config", {})
        temporal = TemporalHDRRefiner(channels=int(temporal_config.get("temporal_channels", 24)))
        temporal.load_state_dict(temporal_ckpt.get("model", temporal_ckpt), strict=True)
        temporal.to(device).eval()
    return image_model, temporal


def bgr8_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    if frame.shape[2] == 4:
        frame = frame[..., :3]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    if np.issubdtype(frame.dtype, np.integer):
        rgb /= float(np.iinfo(frame.dtype).max)
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_rgb(hdr: torch.Tensor) -> np.ndarray:
    return hdr.detach().float().clamp_min(0).cpu().permute(1, 2, 0).numpy()


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def _tile_weight(height: int, width: int, overlap: int, y: int, x: int,
                 full_h: int, full_w: int, device: torch.device) -> torch.Tensor:
    wy = torch.ones(height, device=device, dtype=torch.float32)
    wx = torch.ones(width, device=device, dtype=torch.float32)
    fy, fx = min(overlap, height // 2), min(overlap, width // 2)
    if y > 0 and fy:
        wy[:fy] = torch.linspace(1e-3, 1.0, fy, device=device)
    if y + height < full_h and fy:
        wy[-fy:] = torch.linspace(1.0, 1e-3, fy, device=device)
    if x > 0 and fx:
        wx[:fx] = torch.linspace(1e-3, 1.0, fx, device=device)
    if x + width < full_w and fx:
        wx[-fx:] = torch.linspace(1.0, 1e-3, fx, device=device)
    return (wy[:, None] * wx[None, :])[None, None]


@torch.inference_mode()
def predict_image(model: SDR2HDRNet, sdr: torch.Tensor, preserve_outside: bool,
                  tile_size: int, overlap: int, recovery_mode: str = "all",
                  recovery_strength: float = 1.0) -> torch.Tensor:
    """Memory-bounded image inference with overlap feathering."""
    if sdr.shape[0] != 1:
        raise ValueError("predict_image expects one image at a time")
    _, _, height, width = sdr.shape
    # The conditioning head reads whole-frame statistics, so its scale is
    # computed once here and handed to every tile. Left to itself each tile
    # would predict from its own window and a patch of sky inside a dim
    # interior would reconstruct as if the whole frame were a sunset.
    scale = model.predict_residual_scale(sdr) if hasattr(model, "predict_residual_scale") else None
    shadow_weight = model.predict_shadow_weight(sdr) if hasattr(model, "predict_shadow_weight") else None
    # Same reason for the curve: one tone-curve estimate per FRAME, or every
    # tile would invert its own guess and the seams would show.
    curve = model.predict_curve(sdr) if hasattr(model, "predict_curve") else None
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if sdr.is_cuda else contextlib.nullcontext()
        with amp:
            return model(sdr, preserve_outside=preserve_outside,
                         recovery_mode=recovery_mode,
                         residual_strength=recovery_strength,
                         residual_scale=scale, shadow_weight=shadow_weight,
                         curve_params=curve).hdr.float()
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must be >= 0 and smaller than tile size")
    result = torch.zeros((1, 3, height, width), device=sdr.device, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=sdr.device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = sdr[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if tile.is_cuda else contextlib.nullcontext()
            with amp:
                prediction = model(tile, preserve_outside=preserve_outside,
                                   recovery_mode=recovery_mode,
                                   residual_strength=recovery_strength,
                                   residual_scale=scale, shadow_weight=shadow_weight,
                         curve_params=curve).hdr.float()
            weight = _tile_weight(tile.shape[-2], tile.shape[-1], overlap, y, x,
                                  height, width, sdr.device)
            result[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += prediction * weight
            weights[..., y:y + tile.shape[-2], x:x + tile.shape[-1]] += weight
    return result / weights.clamp_min(1e-6)


@torch.inference_mode()
def predict_fields(model: SDR2HDRNet, sdr: torch.Tensor, tile_size: int,
                   overlap: int) -> dict[str, torch.Tensor]:
    """The raw head fields, stitched: log residual, highlight mask, shadow mask.

    ``recovery_mode``, ``residual_strength`` and ``preserve_outside`` never
    touch these three -- they only enter the composition that happens after.
    So a client that holds the fields can rebuild any combination of them
    without another forward pass, which is what lets RUDRA Studio's controls
    run at frame rate instead of at one HTTP round trip each.

    Feathering is applied to the fields rather than to the composed
    prediction, so inside an overlap band a tiled frame differs from
    ``predict_image`` by the difference between blending before and after
    ``expm1``. Measured on a 160x160 frame with 64/16 tiles that reaches a few
    percent at the worst pixel of a band -- not a rounding error -- while an
    untiled pass agrees with ``predict_image`` to about 3e-6. That is why the
    viewer asks for an untiled pass and falls back to tiles only when it runs
    out of memory, and why the header it gets back says which happened.
    tests/test_frame_fields_2026_08_28.py pins both numbers.
    """
    if sdr.shape[0] != 1:
        raise ValueError("predict_fields expects one image at a time")
    _, _, height, width = sdr.shape

    scale = model.predict_residual_scale(sdr) if hasattr(model, "predict_residual_scale") else None
    # The three fields do not depend on the shadow weight (it scales the
    # prior in the composite, after them), so this only saves forward() an
    # encode it would otherwise run per tile. The weight itself is what
    # ui/server.py sends the page beside the fields.
    # (Named in full: `shadow` is the stitched mask further down, and a
    # closure reads the name at call time, not at definition.)
    shadow_weight = model.predict_shadow_weight(sdr) \
        if hasattr(model, "predict_shadow_weight") else None
    curve = model.predict_curve(sdr) if hasattr(model, "predict_curve") else None

    def _run(tile: torch.Tensor):
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if tile.is_cuda \
            else contextlib.nullcontext()
        with amp:
            out = model(tile, preserve_outside=False, recovery_mode="all",
                        residual_strength=1.0, residual_scale=scale,
                        shadow_weight=shadow_weight, curve_params=curve)
        residual = out.log_residual.float()
        # The viewer composes from these three fields alone and knows nothing
        # about a conditioning head, so the per-frame scale is folded into the
        # residual here. The GLSL composite then stays a line-for-line port of
        # forward()'s tail, and the Studio's strength slider remains a control
        # on top of the model's judgement rather than a replacement for it.
        if out.residual_scale is not None:
            residual = residual * out.residual_scale.float()
        return (residual, out.highlight_mask.float(), out.shadow_mask.float())

    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        residual, highlight, shadow = _run(sdr)
        return {"residual": residual, "highlight": highlight,
                "shadow": shadow, "tiled": False, "curve": curve}

    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile overlap must be >= 0 and smaller than tile size")
    device = sdr.device
    residual = torch.zeros((1, 3, height, width), device=device, dtype=torch.float32)
    highlight = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    shadow = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    weights = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            tile = sdr[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            r, h, s = _run(tile)
            th, tw = tile.shape[-2], tile.shape[-1]
            weight = _tile_weight(th, tw, overlap, y, x, height, width, device)
            residual[..., y:y + th, x:x + tw] += r * weight
            highlight[..., y:y + th, x:x + tw] += h * weight
            shadow[..., y:y + th, x:x + tw] += s * weight
            weights[..., y:y + th, x:x + tw] += weight
    weights = weights.clamp_min(1e-6)
    return {"residual": residual / weights, "highlight": highlight / weights,
            "shadow": shadow / weights, "tiled": True, "curve": curve}


@torch.inference_mode()
def predict_temporal(model: TemporalHDRRefiner, sdr: torch.Tensor, initial: torch.Tensor,
                     tile_size: int, overlap: int) -> torch.Tensor:
    """Memory-bounded spatial tiling for a complete temporal clip."""
    _, _, _, height, width = sdr.shape
    if tile_size <= 0 or (height <= tile_size and width <= tile_size):
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if sdr.is_cuda else contextlib.nullcontext()
        with amp:
            return model(sdr, initial).float()
    result = torch.zeros_like(initial, dtype=torch.float32)
    weights = torch.zeros((1, 1, 1, height, width), device=sdr.device, dtype=torch.float32)
    for y in _tile_starts(height, tile_size, overlap):
        for x in _tile_starts(width, tile_size, overlap):
            sdr_tile = sdr[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            hdr_tile = initial[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if sdr.is_cuda else contextlib.nullcontext()
            with amp:
                prediction = model(sdr_tile, hdr_tile).float()
            weight = _tile_weight(sdr_tile.shape[-2], sdr_tile.shape[-1], overlap, y, x,
                                  height, width, sdr.device).unsqueeze(1)
            result[..., y:y + sdr_tile.shape[-2], x:x + sdr_tile.shape[-1]] += prediction * weight
            weights[..., y:y + sdr_tile.shape[-2], x:x + sdr_tile.shape[-1]] += weight
    return result / weights.clamp_min(1e-6)


def write_outputs(rgb: np.ndarray, output_stem: Path, write_png16: bool = True) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    from rudra.delivery.exr import write_exr
    write_exr(output_stem.with_suffix(".exr"), rgb * (10000.0 / 203.0), half=False,
              attributes={"rudra:nitsScale": "203"})
    bgr = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_stem.with_suffix(".tif")), bgr):
        raise RuntimeError(f"Failed to write {output_stem.with_suffix('.tif')}")
    if write_png16:
        # Raw normalized scene-linear copy. Most ordinary image viewers will
        # display this as very dark because PNG carries no reliable HDR
        # transfer/peak-luminance contract; use the TIFF in an HDR-aware DCC.
        png = cv2.cvtColor((np.clip(rgb, 0, 1) * 65535.0 + 0.5).astype(np.uint16), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_stem.with_name(output_stem.name + "_16bit").with_suffix(".png")), png)

    # Dataset/model convention is normalized scene-linear with 1.0 = 10000
    # nits and reference white = 203 nits. Undo that normalization, then apply
    # the same -1 EV ACES approximation used to make the SDR training inputs.
    # The old preview applied Reinhard directly to the normalized values and
    # was therefore about log2(10000/203) ~= 5.6 stops too dark.
    scene_linear = np.maximum(rgb, 0) / (203.0 / 10000.0)
    x = scene_linear * 0.5
    preview_linear = np.clip(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        0.0, 1.0,
    )
    preview = torch.from_numpy(preview_linear).permute(2, 0, 1)
    preview = linear_to_srgb(preview).permute(1, 2, 0).numpy()
    preview_bgr = cv2.cvtColor((np.clip(preview, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_stem.with_name(output_stem.name + "_preview").with_suffix(".png")), preview_bgr)
    if write_png16:
        display16 = cv2.cvtColor(
            (np.clip(preview, 0, 1) * 65535.0 + 0.5).astype(np.uint16),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(str(output_stem.with_name(output_stem.name + "_display16").with_suffix(".png")), display16)

    stats = {
        "encoding": "scene-linear RGB; 1.0 = 10000 nits",
        "max_normalized": float(np.max(rgb)),
        "max_nits": float(np.max(rgb) * 10000.0),
        "mean_nits": float(np.mean(rgb) * 10000.0),
        "master": str(output_stem.with_suffix(".tif").name),
        "delivery_master": str(output_stem.with_suffix(".exr").name),
        "delivery_nits_scale": 203.0,
        "raw_linear_16bit": str(output_stem.with_name(output_stem.name + "_16bit").with_suffix(".png").name) if write_png16 else None,
        "display_16bit": str(output_stem.with_name(output_stem.name + "_display16").with_suffix(".png").name) if write_png16 else None,
        "preview": str(output_stem.with_name(output_stem.name + "_preview").with_suffix(".png").name),
    }
    output_stem.with_name(output_stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


@torch.inference_mode()
def infer_image(path: Path, output_dir: Path, model: SDR2HDRNet, device: torch.device,
                preserve_outside: bool, write_png16: bool, transfer: str, value_range: str,
                tile_size: int, tile_overlap: int, recovery_mode: str,
                recovery_strength: float) -> None:
    frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Failed to read {path}")
    sdr = canonicalize_sdr(bgr8_to_tensor(frame, device), transfer, value_range)
    hdr = predict_image(model, sdr, preserve_outside, tile_size, tile_overlap,
                        recovery_mode, recovery_strength)[0]
    write_outputs(tensor_to_rgb(hdr), output_dir / path.stem, write_png16)


@torch.inference_mode()
def infer_video(path: Path, output_dir: Path, model: SDR2HDRNet,
                temporal: TemporalHDRRefiner | None, device: torch.device,
                clip_length: int, write_png16: bool, transfer: str, value_range: str,
                tile_size: int, tile_overlap: int, recovery_mode: str,
                recovery_strength: float, preserve_outside: bool = True) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[torch.Tensor] = []
    index = 0
    sequence_dir = output_dir / path.stem
    sequence_dir.mkdir(parents=True, exist_ok=True)

    def process(chunk: list[torch.Tensor], start: int) -> None:
        sdr = torch.cat(chunk, dim=0)
        initial = torch.cat([
            predict_image(model, frame[None], preserve_outside, tile_size, tile_overlap,
                          recovery_mode, recovery_strength)
            for frame in sdr
        ], dim=0)
        if temporal is not None and len(chunk) >= 2:
            refined = predict_temporal(temporal, sdr.unsqueeze(0), initial.unsqueeze(0),
                                       tile_size, tile_overlap)[0]
        else:
            refined = initial
        for offset, frame_hdr in enumerate(refined):
            write_outputs(tensor_to_rgb(frame_hdr), sequence_dir / f"{start + offset:08d}", write_png16)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(canonicalize_sdr(bgr8_to_tensor(frame, device), transfer, value_range))
        if len(frames) == clip_length:
            process(frames, index)
            index += len(frames)
            frames = []
    if frames:
        process(frames, index)
        index += len(frames)
    capture.release()
    (sequence_dir / "sequence.json").write_text(json.dumps({
        "source": str(path.resolve()), "fps": fps, "frames": index,
        "encoding": "scene-linear RGB; 1.0 = 10000 nits",
        "temporal_refiner": temporal is not None,
        "preserve_outside": preserve_outside,
        "input_transfer": transfer, "input_range": value_range,
        "tile_size": tile_size, "tile_overlap": tile_overlap,
        "recovery_mode": recovery_mode, "recovery_strength": recovery_strength,
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--image-checkpoint", "--checkpoint", required=True)
    parser.add_argument("--temporal-checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sdr2hdr"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-length", type=int, default=9)
    # ON by default since 26 Aug 2026. Measured on step 44,000 of the v3b run,
    # val split, 128 batches (training/sweep_inference.py):
    #
    #                     clean psnr_log   gain     hard psnr_log   gain
    #   inverse-ACES              52.76      --             30.24     --
    #   plain                     51.86   -0.90             31.95  +1.71
    #   preserve_outside          52.77   +0.01             31.79  +1.55
    #
    # Off, the model gives back 0.9 dB on a clean well-graded master for 0.16 dB
    # more on a degraded one -- and nothing at inference says which one arrived.
    parser.add_argument("--preserve-outside", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Blend the prediction back to the analytic inverse-ACES "
                             "baseline wherever the learned highlight/shadow masks are "
                             "cold, so the network only acts where the SDR mapping was "
                             "genuinely non-invertible. --no-preserve-outside disables.")
    parser.add_argument("--input-transfer", choices=("auto", "srgb", "rec709", "gamma22", "gamma24"), default="auto")
    parser.add_argument("--input-range", choices=("full", "limited"), default="full")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Spatial tile size for bounded VRAM use; <=0 disables tiling")
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--recovery-mode", choices=("highlights", "all", "shadows", "off"),
                        default="all",
                        help="Where learned recovery may alter the physical baseline")
    parser.add_argument("--recovery-strength", type=float, default=1.0)
    parser.add_argument("--no-png16", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, temporal = load_models(args.image_checkpoint, args.temporal_checkpoint, device)
    paths = sorted(args.input.rglob("*")) if args.input.is_dir() else [args.input]
    paths = [p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS]
    destinations = set()
    for path in paths:
        relative = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        destination = args.output_dir / relative.with_suffix("")
        key = str(destination.resolve()).casefold()
        if key in destinations or destination.exists() or any(destination.parent.glob(destination.name + ".*")):
            raise ValueError(f"Output collision: {destination}")
        destinations.add(key)
    for path in paths:
        relative = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        output_dir = args.output_dir / relative.parent
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            transfer = "srgb" if args.input_transfer == "auto" else args.input_transfer
            infer_image(path, output_dir, model, device, args.preserve_outside,
                        not args.no_png16, transfer, args.input_range,
                        args.tile_size, args.tile_overlap,
                        args.recovery_mode, args.recovery_strength)
        elif suffix in VIDEO_EXTENSIONS:
            transfer = "rec709" if args.input_transfer == "auto" else args.input_transfer
            infer_video(path, output_dir, model, temporal, device, args.clip_length,
                        not args.no_png16, transfer, args.input_range,
                        args.tile_size, args.tile_overlap,
                        args.recovery_mode, args.recovery_strength, args.preserve_outside)


if __name__ == "__main__":
    main()
