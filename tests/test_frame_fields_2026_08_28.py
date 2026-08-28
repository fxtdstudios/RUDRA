"""The Studio composes on the viewer's GPU. This is what keeps that honest.

`/api/frame` ships the three raw head fields and lets ui/compositor.js build
the prediction from them, so every control moves at frame rate instead of at
one forward pass each. That only works if composing from the fields lands on
exactly what `SDR2HDRNet.forward` would have produced -- otherwise the picture
on screen and the EXR that Master writes are two different images with the
same label.

The chain is pinned in two places:

    here                        torch model   <-> tests/compose_reference.py
    tests/webgl_parity/         GLSL shader   <-> tests/compose_reference.py

so the shader can never quietly drift away from the network.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "ui"), str(REPO / "tests")):
    if path not in sys.path:
        sys.path.insert(0, path)

torch = pytest.importorskip("torch")

from compose_reference import baseline_of, compose            # noqa: E402
from rudra.sdr2hdr import SDR2HDRNet                          # noqa: E402
from training.infer_sdr2hdr import predict_fields             # noqa: E402

CASES = [
    (1.0, "all", True),
    (1.0, "all", False),
    (0.0, "all", True),
    (2.0, "highlights", True),
    (1.35, "shadows", False),
    (1.0, "off", True),
]


def _net(seed: int = 7) -> SDR2HDRNet:
    """The head is zero-initialised on purpose; give it something to say."""
    torch.manual_seed(seed)
    net = SDR2HDRNet(base_channels=8)
    torch.nn.init.normal_(net.head.weight, std=0.05)
    torch.nn.init.normal_(net.head.bias, std=0.2)
    return net.eval()


def _sdr(height: int = 48, width: int = 61, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.random((height, width, 3)).astype(np.float32)
    frame[0] = 1.0      # clipped white: the inverse-ACES ceiling
    frame[1] = 0.0      # crushed black: the other clamp
    return frame


def _fields_np(net, tensor, tile_size=0, overlap=0):
    fields = predict_fields(net, tensor, tile_size=tile_size, overlap=overlap)
    return (fields["residual"][0].permute(1, 2, 0).numpy().astype(np.float64),
            fields["highlight"][0].permute(1, 2, 0).numpy().astype(np.float64),
            fields["shadow"][0].permute(1, 2, 0).numpy().astype(np.float64),
            fields)


@pytest.mark.parametrize("strength,mode,preserve", CASES)
def test_fields_compose_to_the_model_prediction(strength, mode, preserve):
    net, sdr = _net(), _sdr()
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None]
    with torch.inference_mode():
        out = net(tensor, preserve_outside=preserve, recovery_mode=mode,
                  residual_strength=strength)
    residual, highlight, shadow, _ = _fields_np(net, tensor)
    reference = compose(sdr.astype(np.float64), residual, highlight, shadow,
                        strength=strength, mode=mode, preserve=preserve)
    got = out.hdr[0].permute(1, 2, 0).numpy().astype(np.float64)
    assert np.abs(reference - got).max() < 1e-5, (
        f"composing from the fields drifts from forward() at "
        f"strength={strength} mode={mode} preserve={preserve}")


def test_analytic_baseline_matches_the_model():
    """The page computes the baseline itself, from the SDR it was handed."""
    net, sdr = _net(), _sdr()
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None]
    with torch.inference_mode():
        out = net(tensor)
    got = out.baseline[0].permute(1, 2, 0).numpy().astype(np.float64)
    assert np.abs(baseline_of(sdr.astype(np.float64)) - got).max() < 1e-6


def test_fields_do_not_depend_on_the_composition_controls():
    """The premise of the whole design: strength, mode and preserve act only
    after the head, so one forward pass covers every setting of them."""
    net, sdr = _net(), _sdr()
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None]
    with torch.inference_mode():
        a = net(tensor, preserve_outside=False, recovery_mode="all", residual_strength=1.0)
        b = net(tensor, preserve_outside=True, recovery_mode="shadows", residual_strength=0.25)
    for name in ("log_residual", "highlight_mask", "shadow_mask"):
        assert torch.equal(getattr(a, name), getattr(b, name)), f"{name} moved"


def test_half_float_transport_costs_less_than_a_thousandth():
    """The fields cross the wire as float16. That has to be free in practice."""
    net, sdr = _net(), _sdr()
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None]
    residual, highlight, shadow, _ = _fields_np(net, tensor)
    exact = compose(sdr.astype(np.float64), residual, highlight, shadow)
    quantised = compose(sdr.astype(np.float64),
                        residual.astype(np.float16).astype(np.float64),
                        highlight.astype(np.float16).astype(np.float64),
                        shadow.astype(np.float16).astype(np.float64))
    relative = np.abs(quantised - exact).max() / max(float(np.abs(exact).max()), 1e-9)
    assert relative < 1e-3, f"half float costs {relative:.2e} relative"


def test_tiling_agrees_with_a_single_pass_away_from_the_seams():
    net, sdr = _net(), _sdr(96, 96)
    tensor = torch.from_numpy(sdr).permute(2, 0, 1)[None]
    whole = predict_fields(net, tensor, tile_size=0, overlap=0)
    tiled = predict_fields(net, tensor, tile_size=64, overlap=16)
    assert whole["tiled"] is False and tiled["tiled"] is True
    # Convolutions see different context at a tile edge, so compare the
    # interior, which is what feathering is supposed to protect.
    a = whole["residual"][..., 24:72, 24:72]
    b = tiled["residual"][..., 24:72, 24:72]
    assert torch.allclose(a, b, atol=5e-3), "tiled fields drift in the interior"


def test_frame_body_is_laid_out_the_way_the_header_says():
    """The client indexes the body by the header's offsets and nothing else."""
    pytest.importorskip("PIL", reason="run_frame decodes the upload with Pillow")
    from PIL import Image

    server = pytest.importorskip("server", reason="ui/server.py needs PIL")

    net = _net()
    sdr = _sdr(40, 53)
    buffer = io.BytesIO()
    Image.fromarray((sdr * 255.0 + 0.5).astype(np.uint8)).save(buffer, format="PNG")

    header, body = server.run_frame(net, buffer.getvalue(),
                                    {"max_side": 4096, "tile_size": 0}, None)
    json.dumps(header)                      # it travels as an HTTP header
    width, height = header["width"], header["height"]
    n = width * height
    offsets = header["offsets"]
    assert (width, height) == (53, 40)
    assert offsets == {"fields": 0, "shadow": n * 8, "sdr": n * 10, "total": n * 11}
    assert len(body) == n * 11

    fields = np.frombuffer(body, dtype=np.float16, count=n * 4,
                           offset=offsets["fields"]).reshape(height, width, 4)
    shadow = np.frombuffer(body, dtype=np.float16, count=n,
                           offset=offsets["shadow"]).reshape(height, width, 1)
    returned_sdr = np.frombuffer(body, dtype=np.uint8, count=n * 3,
                                 offset=offsets["sdr"]).reshape(height, width, 3)

    # The page must compute its baseline from the pixels the network saw.
    assert np.array_equal(returned_sdr, (sdr * 255.0 + 0.5).astype(np.uint8))

    tensor = torch.from_numpy(returned_sdr.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    residual, highlight, shadow_ref, _ = _fields_np(net, tensor)
    assert np.abs(fields[..., :3].astype(np.float64) - residual).max() < 1e-2
    assert np.abs(fields[..., 3:].astype(np.float64) - highlight).max() < 1e-3
    assert np.abs(shadow.astype(np.float64) - shadow_ref).max() < 1e-3
