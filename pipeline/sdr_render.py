"""How a corpus turns scene-linear HDR into the SDR a model learns to invert.

Up to corpus_v4b every SDR frame was rendered one way: exposure, then the
Narkowicz ACES fit, then sRGB, then 8 bits. A model trained on that learns to
invert that one curve. Scored on 23 Sep 2026 against the same 429 references
re-rendered with a Hable curve and a real H.264 round trip, the shipped model
was worse than the analytic baseline it sits on (-0.41 dB PU21, -0.024 JOD,
worse on 270 of 429 frames). Real footage and generated video come through
neither our curve nor our synthetic degradation.

``render_mix`` draws the render per pair from a family of curves real SDR is
actually made with -- filmic, AgX-like, Reinhard, a camera log-to-709 look, a
plain clip -- plus an exposure and contrast jitter and, sometimes, a real codec
round trip. Everything drawn is returned as a dict and written into the pair's
metadata so the bench can be scored per curve and per codec.

Torch-free (numpy + OpenCV + optional ffmpeg), so it runs in the ingest.
Deterministic for a given ``np.random.Generator``.

The nominal exposure (``tonemap_ev``) stays constant across a corpus so
``corpus_ev_of`` still finds one value; the per-pair offset lives in
``render_ev`` and is what a curve-estimating model has to discover.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

RENDER_VERSION = "mix:v1"

# Diffuse white is 1.0 scene-linear throughout (203 nits).


def _srgb_oetf(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055).astype(np.float32)


def _rec709_oetf(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin < 0.018, lin * 4.5,
                    1.099 * np.power(lin, 0.45) - 0.099).astype(np.float32)


# --------------------------------------------------------------------- curves
# Each maps scene-linear (>= 0, diffuse white 1.0) to display-linear [0, 1].

def curve_aces(x: np.ndarray, **_) -> np.ndarray:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def curve_hable(x: np.ndarray, white: float = 11.2, **_) -> np.ndarray:
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30

    def h(v):
        return (v * (a * v + c * b) + d * e) / (v * (a * v + b) + d * f) - e / f

    return np.clip(h(x * 2.0) / h(np.float32(white)), 0.0, 1.0)


def curve_reinhard(x: np.ndarray, white: float = 6.0, **_) -> np.ndarray:
    return np.clip(x * (1.0 + x / (white * white)) / (1.0 + x), 0.0, 1.0)


def curve_agx(x: np.ndarray, contrast: float = 1.0, **_) -> np.ndarray:
    """AgX-like: a sigmoid in log2 exposure. Wide shoulder, soft toe."""
    lo, hi = -12.47, 4.03                       # AgX's published log2 range
    t = (np.log2(np.maximum(x, 2.0 ** lo)) - lo) / (hi - lo)
    t = np.clip(t, 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-(t - 0.6) * 10.0 * contrast))
    s0 = 1.0 / (1.0 + np.exp(0.6 * 10.0 * contrast))
    s1 = 1.0 / (1.0 + np.exp(-0.4 * 10.0 * contrast))
    return np.clip((s - s0) / (s1 - s0), 0.0, 1.0) ** 2.2


def curve_camera_log(x: np.ndarray, contrast: float = 1.0, **_) -> np.ndarray:
    """A camera 'log to Rec.709' look: log encode, S-curve, display.

    Stands in for the manufacturer LUTs most delivered camera footage went
    through. Not any one vendor's curve, on purpose.
    """
    log = np.log2(np.maximum(x, 1e-6) / 0.18) / 14.0 + 0.5    # ~14 stops, grey 0.5
    log = np.clip(log, 0.0, 1.0)
    k = 7.0 * contrast
    s = 1.0 / (1.0 + np.exp(-(log - 0.5) * k))
    s0, s1 = 1.0 / (1.0 + np.exp(0.5 * k)), 1.0 / (1.0 + np.exp(-0.5 * k))
    return np.clip((s - s0) / (s1 - s0), 0.0, 1.0) ** 2.4


def curve_clip(x: np.ndarray, **_) -> np.ndarray:
    """No curve: scale, clip. What a phone's 'natural' mode or a naive
    renderer produces, and the hardest case -- everything above white is gone."""
    return np.clip(x, 0.0, 1.0)


CURVES = {
    "aces": curve_aces,
    "hable": curve_hable,
    "reinhard": curve_reinhard,
    "agx": curve_agx,
    "camera_log": curve_camera_log,
    "clip": curve_clip,
}
# Weights: ACES stays the most common so the bench's clean/hard conditions,
# which render with ACES, stay in distribution.
CURVE_WEIGHTS = {"aces": 0.25, "hable": 0.15, "reinhard": 0.12, "agx": 0.16,
                 "camera_log": 0.20, "clip": 0.12}


# --------------------------------------------------------------------- codecs

def _jpeg(rgb8: np.ndarray, quality: int) -> np.ndarray:
    import cv2
    ok, buf = cv2.imencode(".jpg", rgb8[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return rgb8
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].copy()


_FFMPEG_ENCODERS: set[str] | None = None


def available_video_codecs() -> list[str]:
    """libx264 / libx265 / libaom-av1 / libsvtav1 that this ffmpeg can encode."""
    global _FFMPEG_ENCODERS
    if _FFMPEG_ENCODERS is None:
        _FFMPEG_ENCODERS = set()
        if shutil.which("ffmpeg"):
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True).stdout
            for name in ("libx264", "libx265", "libsvtav1", "libaom-av1"):
                if f" {name} " in out:
                    _FFMPEG_ENCODERS.add(name)
    return sorted(_FFMPEG_ENCODERS)


def _video_codec(rgb8: np.ndarray, encoder: str, crf: int) -> np.ndarray:
    """One frame through a real 4:2:0 video encoder and back."""
    import cv2
    h, w = rgb8.shape[:2]
    pad_h, pad_w = h % 2, w % 2
    frame = np.pad(rgb8, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge") if (pad_h or pad_w) else rgb8
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.png", Path(tmp) / "out.mkv"
        cv2.imwrite(str(src), frame[..., ::-1])
        extra = ["-cpu-used", "8", "-row-mt", "1"] if encoder == "libaom-av1" else []
        extra += ["-preset", "12"] if encoder == "libsvtav1" else []
        extra += ["-x265-params", "log-level=error"] if encoder == "libx265" else []
        enc = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", encoder,
               "-crf", str(int(crf)), *extra, "-pix_fmt", "yuv420p", "-frames:v", "1", str(dst)]
        dec = ["ffmpeg", "-y", "-v", "error", "-i", str(dst), "-frames:v", "1",
               str(Path(tmp) / "out.png")]
        for cmd in (enc, dec):
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                return rgb8
        out = cv2.imread(str(Path(tmp) / "out.png"), cv2.IMREAD_COLOR)
        if out is None:
            return rgb8
        return out[:h, :w, ::-1].copy()


# --------------------------------------------------------------------- render

def render_aces(linear: np.ndarray, ev: float) -> np.ndarray:
    """The legacy render, bit-for-bit with training.prepare_training_data.make_sdr."""
    display = curve_aces(np.maximum(linear, 0.0) * (2.0 ** ev))
    return np.clip(np.rint(_srgb_oetf(display) * 255.0), 0, 255).astype(np.uint8)


def draw_render(rng: np.random.Generator, ev_jitter: float = 1.5,
                codec_probability: float = 0.5, codecs: list[str] | None = None) -> dict:
    """Draw one render recipe. Separate from applying it so a whole shot can
    share one recipe (a clip is graded once) while crops and frames vary."""
    names = list(CURVE_WEIGHTS)
    weights = np.array([CURVE_WEIGHTS[n] for n in names], dtype=np.float64)
    curve = str(rng.choice(names, p=weights / weights.sum()))
    recipe = {
        "render_version": RENDER_VERSION,
        "curve": curve,
        "render_ev": round(float(rng.uniform(-ev_jitter, ev_jitter)), 4),
        "contrast": round(float(rng.uniform(0.8, 1.25)), 4),
        "saturation": round(float(rng.uniform(0.85, 1.15)), 4),
        "white": round(float(rng.uniform(4.0, 16.0)), 3),
        "oetf": str(rng.choice(["srgb", "rec709"], p=[0.7, 0.3])),
        "codec": "none",
        "quality": None,
    }
    if rng.random() < codec_probability:
        pool = ["jpeg"] + list(codecs if codecs is not None else available_video_codecs())
        recipe["codec"] = str(rng.choice(pool))
        recipe["quality"] = (int(rng.integers(40, 93)) if recipe["codec"] == "jpeg"
                             else int(rng.integers(18, 36)))
    return recipe


def apply_render(linear: np.ndarray, recipe: dict, nominal_ev: float = 0.0) -> np.ndarray:
    """Scene-linear (H,W,3) -> uint8 sRGB-ish SDR, by the recipe."""
    x = np.maximum(linear.astype(np.float32), 0.0) * (2.0 ** (nominal_ev + recipe["render_ev"]))
    display = CURVES[recipe["curve"]](x, white=recipe["white"], contrast=recipe["contrast"])
    grey = display.mean(axis=-1, keepdims=True)
    display = np.clip(grey + recipe["saturation"] * (display - grey), 0.0, 1.0)
    encoded = _srgb_oetf(display) if recipe["oetf"] == "srgb" else _rec709_oetf(display)
    rgb8 = np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)
    codec = recipe.get("codec", "none")
    if codec == "jpeg":
        rgb8 = _jpeg(rgb8, recipe["quality"])
    elif codec != "none":
        rgb8 = _video_codec(rgb8, codec, recipe["quality"])
    return rgb8


def render_mix(linear: np.ndarray, rng: np.random.Generator, nominal_ev: float = 0.0,
               **draw_kwargs) -> tuple[np.ndarray, dict]:
    recipe = draw_render(rng, **draw_kwargs)
    return apply_render(linear, recipe, nominal_ev), recipe
