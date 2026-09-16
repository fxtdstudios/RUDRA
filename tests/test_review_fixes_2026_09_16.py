"""What the 16 Sep 2026 review found, pinned so it stays fixed.

Three reads of the tree -- engineer, colourist, researcher -- and every
finding here was reproduced before it was fixed. Each test names the failure
it guards against in its docstring, because a test that only says what the
code does now is no defence against someone putting it back.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO, REPO / "ui"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from rudra.delivery import video as video_mod                          # noqa: E402
from rudra.delivery.colorspace import REC2020_CHROMATICITIES, convert  # noqa: E402
from rudra.delivery.exr import read_exr                                # noqa: E402
from rudra.delivery.video import (encode_sequence, hlg_inverse_ootf,   # noqa: E402
                                  hlg_oetf, shoulder_to_peak)
from rudra.hdr10 import pq_oetf                                        # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg and ffprobe")
torch = pytest.importorskip("torch")
needs_torch = pytest.mark.skipif(torch is None, reason="needs torch")


# ---------------------------------------------------------------------------
# delivery
# ---------------------------------------------------------------------------
def _decode_ycbcr(path: Path) -> np.ndarray:
    """First frame as Y'CbCr 16-bit planes, straight from the file."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "yuv444p16le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint16)


@needs_ffmpeg
def test_the_chroma_matrix_is_the_one_the_tag_claims(tmp_path):
    """bt2020nc was written into the tags; BT.601 was written into the pixels.

    -colorspace on the output only labels it. The RGB -> Y'CbCr conversion is
    swscale's, inserted for the pix_fmt change, and it used its default
    matrix. Measured on ffmpeg 6.1 before the fix: pure red at PQ' 0.75 came
    out Y' = 1044 (12-bit) where BT.2020 says 946 and BT.601 says 1042. A
    decoder honouring the tag then renders foliage 13% dark in green. The tag
    check could not catch it because the tag was right.
    """
    # A frame whose encoded red is PQ' 0.75 exactly, so the expected Y' is
    # arithmetic: Y' = Kr * R' with Kr = 0.2627 (BT.2020) or 0.299 (BT.601).
    red_nits = float(np.interp(0.75, pq_oetf(np.linspace(0, 10000, 100001)),
                               np.linspace(0, 10000, 100001)))
    frame = np.zeros((16, 16, 3)); frame[..., 0] = red_nits

    out = encode_sequence(iter([frame, frame]), tmp_path / "red", target="prores4444",
                          fps=24.0, peak_nits=10000.0, shoulder=False)
    y16 = int(_decode_ycbcr(out)[0])
    y_2020 = (16 + 219 * 0.2627 * 0.75) / 255.0 * 65535.0
    y_601 = (16 + 219 * 0.299 * 0.75) / 255.0 * 65535.0
    assert abs(y16 - y_2020) < abs(y16 - y_601), (y16, y_2020, y_601)
    assert abs(y16 - y_2020) < 0.01 * 65535, (y16, y_2020)


def test_hlg_inverse_ootf_puts_diffuse_white_where_bt2408_says():
    """HLG's OETF takes scene light; RUDRA has display light.

    Feeding nits/peak straight to the OETF -- what deliver did -- skipped the
    display's OOTF (gamma 1.2 at 1,000 nits), so 203 nits encoded to signal
    0.697 and displayed at 148 nits: half a stop dark, a stop dark at 18%
    grey. With the inverse OOTF, 203 nits on a 1,000-nit HLG display encodes
    to the 75% signal BT.2408 specifies, within the tolerance of the
    reference-white convention (0.75 +- 0.01).
    """
    white = np.array([[[203.0, 203.0, 203.0]]]) / 1000.0
    signal = hlg_oetf(hlg_inverse_ootf(white, 1000.0))
    assert signal[0, 0, 0] == pytest.approx(0.75, abs=0.01), signal
    # And the OOTF round-trips: applying the forward OOTF to the inverse
    # gives the display light back.
    scene = hlg_inverse_ootf(white, 1000.0)
    y_s = float(np.sum(scene[0, 0] * video_mod.HLG_LUMA))
    y_d = y_s ** 1.2
    assert y_d == pytest.approx(0.203, rel=1e-6)
    # Hue is preserved: the inverse is applied on luminance, so a coloured
    # pixel keeps its RGB ratios.
    colour = np.array([[[0.4, 0.2, 0.1]]])
    ratio = hlg_inverse_ootf(colour, 1000.0)[0, 0] / colour[0, 0]
    assert np.allclose(ratio, ratio[0])


def test_deliver_never_exceeds_the_mastering_peak_it_declares():
    """A PQ stream whose pixels top its mastering display is a QC reject.

    The analytic baseline puts SDR white at 2,552 nits and deliver clipped
    at 10,000, so any plain white shirt produced MaxCLL > MaxMDL. The
    shoulder is hue-preserving (rudra.hdr10.master_to_peak), rolls off from
    75% of peak, and leaves everything below the knee alone.
    """
    frame = np.zeros((4, 4, 3))
    frame[0, 0] = (2552.0, 2552.0, 2552.0)   # the baseline's SDR white
    frame[0, 1] = (500.0, 500.0, 500.0)      # below the 750-nit knee
    frame[0, 2] = (3000.0, 600.0, 200.0)     # a saturated highlight
    out = shoulder_to_peak(frame, 1000.0)
    assert out.max() <= 1000.0 + 1e-6
    assert np.allclose(out[0, 1], frame[0, 1])
    ratio = out[0, 2] / frame[0, 2]
    assert np.allclose(ratio, ratio[0], rtol=1e-6), "the shoulder changed hue"


@needs_ffmpeg
def test_encode_shoulders_by_default_and_can_be_told_not_to(tmp_path):
    frame = np.full((16, 16, 3), 3600.0)
    shouldered = encode_sequence(iter([frame, frame]), tmp_path / "a", target="hdr10",
                                 peak_nits=1000.0)
    raw = encode_sequence(iter([frame, frame]), tmp_path / "b", target="hdr10",
                          peak_nits=1000.0, shoulder=False)

    def peak(path):
        px = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1",
                             "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                            capture_output=True, check=True).stdout
        return float(np.frombuffer(px, dtype=np.uint16).max()) / 65535.0
    assert peak(shouldered) == pytest.approx(float(pq_oetf(np.array(1000.0))), abs=0.02)
    assert peak(raw) == pytest.approx(float(pq_oetf(np.array(3600.0))), abs=0.02)


def test_deliver_defaults_to_rec709_because_that_is_what_an_sdr_plate_is():
    """RUDRA never converts primaries, so an sRGB plate's master is Rec.709.

    Both `deliver` and `aces` defaulted to rec2020, which handed 709
    primaries to the 2020->AP0 matrix: skin (0.8, 0.5, 0.4) came back on a
    709 monitor as (1.005, 0.463, 0.383) and pure red left the gamut.
    """
    from rudra.delivery import cli
    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:
        pytest.skip("cli has no build_parser")
    for sub in ("deliver", "aces"):
        args = parser.parse_args([sub, "x", "--output", "y"])
        assert args.source_space == "rec709", sub


def test_the_linear_master_says_it_is_rec2020_and_is(tmp_path):
    """A Rec.2020 linear EXR with no chromaticities reads as 709 in Resolve.

    The Studio's linear container wrote 709 pixels with no header, labelled
    Rec.2020 in the sidecar. It now converts 709 -> 2020 in linear light and
    writes the chromaticities, so the file is what it says.
    """
    from rudra.delivery.exr import write_exr
    rgb709 = np.array([[[0.8, 0.5, 0.4]]], dtype=np.float32)
    rgb2020 = convert(rgb709, "rec709", "rec2020").astype(np.float32)
    path = tmp_path / "m.exr"
    write_exr(path, rgb2020, half=True, chromaticities=REC2020_CHROMATICITIES)
    back, attrs = read_exr(path)
    assert tuple(round(v, 3) for v in attrs["chromaticities"]) == \
        tuple(round(v, 3) for v in REC2020_CHROMATICITIES)
    # Pure 709 red is inside 2020, so nothing clips, and the round trip holds.
    assert np.allclose(convert(back, "rec2020", "rec709"), rgb709, atol=2e-3)


# ---------------------------------------------------------------------------
# Studio server
# ---------------------------------------------------------------------------
server = pytest.importorskip("server", reason="ui/server.py needs PIL")


def test_studio_binds_loopback_unless_told_otherwise():
    """0.0.0.0 made every endpoint a LAN-facing file reader and pickle loader."""
    import argparse
    import inspect
    src = inspect.getsource(server.main)
    assert 'ThreadedServer((args.host, args.port)' in src
    assert 'ThreadedServer(("", args.port)' not in src
    parser = argparse.ArgumentParser()
    # The default must be loopback with no environment override in play.
    import os
    env = os.environ.pop("RUDRA_HOST", None)
    try:
        assert '"127.0.0.1"' in src
    finally:
        if env is not None:
            os.environ["RUDRA_HOST"] = env


def test_a_checkpoint_override_is_not_an_authorisation(tmp_path, monkeypatch):
    """X-Rudra-Params {"checkpoint": <path>} went to torch.load(weights_only=False).

    Any file on disk could be named, and a payload dict is a pickle: a .pt
    in Downloads was code execution. The override now resolves only to the
    committed registry or to a .pt under the configured checkpoint roots.
    """
    stray = tmp_path / "evil.pt"
    stray.write_bytes(b"not a checkpoint")
    with pytest.raises(server.RequestRefused):
        server.allowed_checkpoint(str(stray))
    with pytest.raises(server.RequestRefused):
        server.allowed_checkpoint(str(tmp_path / "missing.pt"))
    with pytest.raises(server.RequestRefused):
        server.allowed_checkpoint(str(tmp_path / "notes.txt"))
    # A committed model by bare name is fine, wherever the request spells it.
    loadable = server.loadable_models()
    if loadable:
        name = Path(loadable[0]["path"]).name
        assert server.allowed_checkpoint(name) == Path(loadable[0]["path"])
        assert server.allowed_checkpoint(str(tmp_path / "anywhere" / name)) == \
            Path(loadable[0]["path"])
    # A .pt under a configured root is fine too.
    root = tmp_path / "runs"; (root / "exp").mkdir(parents=True)
    ok = root / "exp" / "best.pt"; ok.write_bytes(b"")
    monkeypatch.setenv("RUDRA_CHECKPOINT_ROOTS", str(root))
    assert server.allowed_checkpoint(str(ok)) == ok.resolve()


def test_a_request_body_has_a_ceiling():
    class FakeHandler:
        def __init__(self, length):
            self.headers = {"Content-Length": str(length)}
            self.rfile = None
    with pytest.raises(server.RequestRefused):
        server.read_body(FakeHandler(server.MAX_BODY_BYTES + 1))
    with pytest.raises(server.RequestRefused):
        server.read_body(FakeHandler("-5"))
    assert server.read_body(FakeHandler(0)) == b""


def test_the_frame_header_carries_what_the_compositor_needs():
    """corpus_ev was a constant in the shader; fps was never sent at all.

    A 0 EV checkpoint would have previewed one stop brighter than its own
    master, and every shot played at 24 fps whatever its rate.
    """
    import inspect
    src = inspect.getsource(server.run_frame)
    assert '"corpus_ev"' in src
    seq = inspect.getsource(server.make_handler)
    assert 'header["fps"]' in seq
    js = (REPO / "ui" / "compositor.js").read_text(encoding="utf-8")
    assert "uniform float uBaselineScale;" in js
    assert "const float LOG_SCALE" not in js
    assert "2.0 * 203.0 / 10000.0" not in js
    app = (REPO / "ui" / "app.js").read_text(encoding="utf-8")
    assert "corpus_ev: head.corpus_ev" in app


def test_every_stylesheet_and_script_the_page_loads_is_cache_stamped():
    """theme.css and shell.js shipped with ?v=1 -- the stale-cache failure
    the server's own docstring describes -- while only three files were
    stamped."""
    import inspect, re
    html = (REPO / "ui" / "index.html").read_text(encoding="utf-8")
    loaded = set(re.findall(r'(?:href|src)="([^"?]+\.(?:css|js))', html))
    stamped = set(re.findall(r'"([a-z_]+\.(?:css|js))"',
                             inspect.getsource(server.make_handler)))
    assert loaded <= stamped, loaded - stamped


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------
@needs_torch
def test_tiled_inference_uses_one_shadow_weight_for_the_whole_frame():
    """forward() computed the gate from whatever it was handed.

    Tiled, that was one weight per tile -- seams in the shadows of a master
    the viewer never showed, because the viewer sends one whole-frame
    weight. Untiled at 1280x720, it was a weight from full-resolution pooled
    features the gate was never trained on. predict_image now computes it
    once, the way predict_shadow_weight and the Studio do.
    """
    sys.path.insert(0, str(REPO))
    from rudra.sdr2hdr import SDR2HDRNet
    from training.infer_sdr2hdr import predict_image

    torch.manual_seed(0)
    net = SDR2HDRNet(base_channels=8, shadow_conditioning=True).eval()
    torch.nn.init.normal_(net.head.weight, std=0.05)
    torch.nn.init.normal_(net.shadow_gate.mlp[-1].weight, std=2.0)
    torch.nn.init.normal_(net.shadow_gate.mlp[-1].bias, std=2.0)
    torch.manual_seed(3)
    # Bright left half, dark right half: the tiles disagree about the frame.
    sdr = torch.rand(1, 3, 64, 128)
    sdr[..., 64:] *= 0.05

    weight = net.predict_shadow_weight(sdr)
    with torch.no_grad():
        tiled = predict_image(net, sdr, preserve_outside=False, tile_size=64, overlap=0,
                              recovery_mode="shadows")
        ref_r = net(sdr[..., 64:], preserve_outside=False, recovery_mode="shadows",
                    shadow_weight=weight).hdr
        per_tile = net(sdr[..., 64:], preserve_outside=False, recovery_mode="shadows").hdr
    # The dark tile composed with the whole-frame weight, not its own.
    assert torch.allclose(tiled[..., 64:], ref_r, atol=1e-5)
    assert not torch.allclose(per_tile, ref_r, atol=1e-3), \
        "the fixture does not separate the two behaviours; strengthen the gate init"


@needs_torch
def test_corpus_ev_travels_from_the_manifest_into_the_model(tmp_path):
    """prepare_training_data wrote tonemap_ev; nothing read it.

    The model was always built with the legacy -1 EV baseline, so the first
    checkpoint trained on a 0 EV corpus would have learned against a
    baseline one stop too bright.
    """
    from training.sdr2hdr_dataset import corpus_ev_of
    from rudra.sdr2hdr import SDR2HDRNet, LEGACY_CORPUS_EV

    def manifest(rows):
        p = tmp_path / f"m{len(list(tmp_path.iterdir()))}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return p

    # Legacy corpus: no sidecar, no field -> legacy exposure.
    assert corpus_ev_of(manifest([{"sdr_path": "a"}, {"sdr_path": "b"}])) == LEGACY_CORPUS_EV
    # A v4 corpus says so in every row.
    assert corpus_ev_of(manifest([{"tonemap_ev": 0.0}] * 3)) == 0.0
    # Or in its sidecars.
    side = tmp_path / "meta.json"; side.write_text(json.dumps({"tonemap_ev": 0.0}))
    assert corpus_ev_of(manifest([{"metadata_path": str(side)}])) == 0.0
    # Two exposures is two corpora.
    with pytest.raises(ValueError):
        corpus_ev_of(manifest([{"tonemap_ev": 0.0}, {"tonemap_ev": -1.0}]))
    # And a partly-labelled manifest is refused rather than guessed.
    with pytest.raises(ValueError):
        corpus_ev_of(manifest([{"tonemap_ev": 0.0}, {"sdr_path": "x"}]))
    # from_config reads what train_sdr2hdr.py now writes.
    assert SDR2HDRNet.from_config({"corpus_ev": 0.0, "base_channels": 8}).corpus_ev == 0.0
    src = (REPO / "training" / "train_sdr2hdr.py").read_text(encoding="utf-8")
    assert '"corpus_ev": corpus_ev' in src and "corpus_ev=corpus_ev" in src


def test_infer_defaults_to_the_mode_the_shipped_model_was_scored_in():
    src = (REPO / "training" / "infer_sdr2hdr.py").read_text(encoding="utf-8")
    assert 'default="all",' in src.split('"--recovery-mode"')[1][:200]
