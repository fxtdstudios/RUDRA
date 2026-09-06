"""The temporal gate's hard condition, and the assumption hiding inside it.

The v02 gate asks whether neighbouring frames carry information a single frame
does not. On 4 Sep 2026 it answered +9.308 JOD -- enormous -- and the answer
was an artefact of how the degradation was seeded, not a fact about video.

`degrade_like_eval` reseeds from the frame index. Applied down a clip that
redraws the exposure, the tone curve, the white balance, the saturation, the
chroma subsampling, the bit depth and the JPEG quality on EVERY FRAME. Real
footage does none of that: a shot has one grade and one encoder, and the only
term that genuinely varies frame to frame is sensor noise. Averaging nine
independent draws of a corruption is worth sqrt(9) whether or not the frames
carry any information at all, so the per-frame condition hands a temporal
model a win it has not earned.

`apply_hard` makes the assumption a flag instead of an accident. These tests
hold the three modes to what they claim, and hold the gate to scoring RUDRA in
the configuration RUDRA ships in -- the other half of the 4 Sep discrepancy,
where a bare `model(x)` call scored -2.396 JOD against the image benchmark's
7.805 on the same checkpoint and the same 1280x720 frames.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

torch = pytest.importorskip("torch")

from training.gate_temporal_oracle import (          # noqa: E402
    NETWORK_PEAK_NITS, STORAGE_WHITE_NITS, apply_hard, codec_round_trip)


def frame(seed: int = 7) -> "torch.Tensor":
    """A CHW 8-bit-ish SDR frame with structure, not noise.

    Band-limited on purpose: a white-noise fixture would make every mode look
    different from every other for reasons that have nothing to do with the
    degradation.
    """
    generator = torch.Generator().manual_seed(seed)
    small = torch.rand((3, 12, 20), generator=generator)
    return torch.nn.functional.interpolate(
        small[None], size=(96, 160), mode="bilinear", align_corners=False)[0]


NOISY_CLIP = "hall_of_fire_4k_c0"      # rolls below 0.60: sensor noise on
QUIET_CLIP = "aarfontein_dirt_road_4k_c0"   # rolls 0.734: sensor noise off


def test_per_clip_is_identical_on_every_frame():
    """A frozen grade must not move down the clip. This is the lower bound."""
    base = frame()
    first = apply_hard(base, 0, 1234, "per-clip")
    last = apply_hard(base, 8, 1234, "per-clip")
    assert torch.equal(first, last)


def test_per_frame_moves_on_every_frame():
    """The 4 Sep condition. Kept reproducible, not kept as the default."""
    base = frame()
    first = apply_hard(base, 0, 1234, "per-frame")
    last = apply_hard(base, 8, 1234, "per-frame")
    assert not torch.equal(first, last)
    # Not a rounding difference -- a whole different grade.
    assert (first - last).abs().mean().item() > 1e-3


def test_realistic_moves_far_less_than_per_frame():
    """The point of the calibration, in one number.

    Both modes vary down the clip; the question is by how much. Under
    `realistic` the only thing that changes is sensor noise, bounded by
    sigma <= 3/255. Under `per-frame` the entire grade changes. If these two
    were ever the same order of magnitude, the mode would not be worth having.
    """
    base = frame()
    realistic = [apply_hard(base, i, 1234, "realistic") for i in range(9)]
    per_frame = [apply_hard(base, i, 1234, "per-frame") for i in range(9)]

    def spread(clip):
        stack = torch.stack(clip)
        return (stack - stack.mean(0, keepdim=True)).abs().mean().item()

    assert spread(realistic) < spread(per_frame) / 5.0


def test_realistic_keeps_the_grade_of_per_clip():
    """Same seed, same shot: the grade underneath must be the per-clip one.

    Sensor noise is added ON TOP of that grade, so with the noise branch off
    the two modes have to agree exactly, and with it on they may differ only
    by something small.
    """
    base = frame()
    for clip_seed in (1234, 99, 20260905):
        quiet = apply_hard(base, 3, clip_seed, "per-clip")
        noisy = apply_hard(base, 3, clip_seed, "realistic")
        assert (quiet - noisy).abs().max().item() < 6.0 * (3.0 / 255.0)


def test_realistic_noise_is_independent_across_frames():
    """Whatever varies must vary INDEPENDENTLY, or averaging cannot help.

    Correlated noise would make the gate pessimistic in the same way per-frame
    seeding made it optimistic, so the direction of this check matters.
    """
    base = frame()
    seed = next(s for s in range(1, 400)
                if not torch.equal(apply_hard(base, 0, s, "realistic"),
                                   apply_hard(base, 1, s, "realistic")))
    a = (apply_hard(base, 0, seed, "realistic") - apply_hard(base, 0, seed, "per-clip"))
    b = (apply_hard(base, 1, seed, "realistic") - apply_hard(base, 1, seed, "per-clip"))
    correlation = float((a * b).mean() / (a.std() * b.std() + 1e-12))
    assert abs(correlation) < 0.05


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        apply_hard(frame(), 0, 1234, "whatever")


def test_reference_ceiling_matches_the_image_benchmark():
    """The gate and the benchmark must clamp the reference at the same place.

    An unclamped rendered panorama carries the sun at 1e8 nits, and those few
    pixels dominate PU21-PSNR. The two numbers are only comparable while this
    conversion holds.
    """
    from training.export_bench_pairs import REFERENCE_CEILING
    ceiling = REFERENCE_CEILING * (NETWORK_PEAK_NITS / STORAGE_WHITE_NITS)
    assert np.isclose(ceiling * STORAGE_WHITE_NITS,
                      REFERENCE_CEILING * NETWORK_PEAK_NITS)
    assert ceiling > 1000.0        # far above diffuse white, as it must be


def test_gate_scores_rudra_as_deployed():
    """`preserve_outside=True` is not a detail -- it is most of the model.

    Outside the learned highlight/shadow masks RUDRA as deployed returns the
    analytic baseline UNTOUCHED. A bare forward pass replaces it everywhere,
    including the midtones the baseline already inverts almost exactly. The
    gate scored the bare pass on 4 Sep and produced a floor 9.8 JOD below the
    image benchmark's on the same checkpoint.

    A freshly initialised model predicts almost no residual, so comparing the
    two paths on one would compare the baseline with itself and pass whatever
    the code did. The head's biases are set here to make the residual large
    and the learned masks closed -- the configuration in which the difference
    between the two paths is the whole picture.
    """
    from rudra.sdr2hdr import SDR2HDRNet
    from training.infer_sdr2hdr import predict_image

    torch.manual_seed(0)
    model = SDR2HDRNet(base_channels=8).eval()
    with torch.no_grad():
        model.head.bias[:3] = 2.0      # a residual worth several stops
        # sigmoid(-20) is 2e-9: shut, not merely small. At -8 the masks
        # still leak 3.3e-4 of a multi-stop residual, which is above
        # the tolerance this test is worth stating.
        model.head.bias[3:] = -20.0    # highlight and shadow masks shut
    x = frame()[None]

    deployed = predict_image(model, x, preserve_outside=True, tile_size=0,
                             overlap=64, recovery_mode="all",
                             recovery_strength=1.0)
    bare = model(x).hdr
    out = model(x, preserve_outside=True, recovery_mode="all")

    # Shut masks: as deployed, the output IS the analytic baseline.
    assert torch.allclose(deployed, out.baseline, atol=1e-5)
    # And the bare pass is a different image entirely, which is the point.
    assert (bare - out.baseline).abs().max().item() > 1e-2
    assert not torch.allclose(deployed, bare, atol=1e-3)


def test_preserve_outside_is_a_composite_not_a_switch():
    """The contract the gate now depends on, stated as an identity.

    ``pred = baseline + max(highlight, shadow) * (pred_open - baseline)``.
    Anything that changes this changes what every v02 measurement means, so
    it is pinned here rather than left to the two call sites to agree by
    habit.
    """
    from rudra.sdr2hdr import SDR2HDRNet

    torch.manual_seed(1)
    model = SDR2HDRNet(base_channels=8).eval()
    with torch.no_grad():
        model.head.bias[:3] = 1.5
    x = frame(11)[None]
    with torch.no_grad():
        opened = model(x, preserve_outside=False, recovery_mode="all")
        closed = model(x, preserve_outside=True, recovery_mode="all")
    recovery = torch.maximum(opened.highlight_mask, opened.shadow_mask)
    expected = opened.baseline + recovery * (opened.hdr - opened.baseline)
    assert torch.allclose(closed.hdr, expected, atol=1e-6)


# --------------------------------------------------------------------------
# The codec mode. The three synthetic modes argue about how a per-frame
# degradation should move down a clip; this one declines to be a per-frame
# degradation at all.
# --------------------------------------------------------------------------

def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


ffmpeg_only = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not installed")


@ffmpeg_only
def test_codec_round_trip_preserves_the_clip_shape():
    clip = [frame(i).permute(1, 2, 0).numpy() for i in range(6)]
    out = codec_round_trip(clip, crf=28, fps=24.0)
    assert len(out) == len(clip)
    for original, decoded in zip(clip, out):
        assert decoded.shape == original.shape
        assert decoded.dtype == np.float32
        assert 0.0 <= float(decoded.min()) and float(decoded.max()) <= 1.0


@ffmpeg_only
def test_codec_actually_damages_the_frames():
    """A round trip that changes nothing would make the mode a no-op."""
    clip = [frame(i).permute(1, 2, 0).numpy() for i in range(6)]
    out = codec_round_trip(clip, crf=40, fps=24.0)
    delta = float(np.abs(np.stack(out) - np.stack(clip)).mean())
    assert delta > 1e-3


@ffmpeg_only
def test_codec_error_is_correlated_along_a_static_clip():
    """THE REASON THIS MODE EXISTS.

    A codec predicts each frame from its neighbours, so on a shot that does
    not change it spends almost nothing after the first frame and every frame
    carries the SAME error. Averaging aligned neighbours cannot remove an
    error they all share -- which is the case the per-frame seeding hid, and
    the case most real footage is closer to.

    Contrast with independent per-frame noise, where the same average is worth
    sqrt(n) whether or not anything was learned.
    """
    still = frame(3).permute(1, 2, 0).numpy()
    out = codec_round_trip([still] * 6, crf=32, fps=24.0)
    errors = np.stack([o - still for o in out])
    shared = errors.mean(axis=0)
    residual = errors - shared
    # The part every frame shares must dominate the part that varies.
    assert np.abs(shared).mean() > 4.0 * np.abs(residual).mean()
