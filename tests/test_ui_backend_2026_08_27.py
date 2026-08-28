"""The UI's measurement path, which is the only part of it that can lie quietly.

The page it serves used to be a mock: no network calls, metrics computed from
slider positions. Now it reports MaxCLL/MaxFALL from the same code that writes
the HDR10 sidecar, so a units slip here would put a plausible wrong number on
screen with a real label over it. That already happened once during the build:
``analyze_frame`` takes ABSOLUTE NITS, and feeding it nits/203 reported a
4,000-nit specular as MaxCLL 20.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for path in (str(REPO), str(REPO / "ui")):
    if path not in sys.path:
        sys.path.insert(0, path)

server = pytest.importorskip("server", reason="ui/server.py needs PIL")

NETWORK_PEAK_NITS = 10_000.0


def _frame(background_nits: float, specular_nits: float, size: int = 64):
    """(3, H, W) in network units, with a 4x4 specular patch."""
    frame = np.full((3, size, size), background_nits / NETWORK_PEAK_NITS, dtype=np.float32)
    frame[:, 10:14, 10:14] = specular_nits / NETWORK_PEAK_NITS
    return frame


def _masks(size: int, hi_slice, sh_slice):
    hi = np.zeros((size, size), dtype=np.float32)
    sh = np.zeros((size, size), dtype=np.float32)
    hi[hi_slice] = 1.0
    sh[sh_slice] = 1.0
    return hi, sh


def test_maxcll_is_absolute_nits_not_scene_linear():
    hdr = _frame(203.0, 4000.0)
    base = _frame(203.0, 900.0)

    m = server.measure(hdr, base, *_masks(64, (slice(10, 14), slice(10, 14)),
                                          (slice(60, 62), slice(None))))

    # ceil() over a float32 round-trip can add one nit; 20 vs 4000 is the bug.
    assert 4000 <= m["maxcll"] <= 4001, m["maxcll"]
    expected_fall = (203.0 * (64 * 64 - 16) + 4000.0 * 16) / (64 * 64)
    assert abs(m["maxfall"] - expected_fall) <= 2, m["maxfall"]


def test_headroom_reports_stops_gained_over_the_analytic_baseline():
    m = server.measure(_frame(203.0, 4000.0), _frame(203.0, 1000.0),
                       *_masks(64, (slice(10, 14), slice(10, 14)),
                               (slice(60, 62), slice(None))))
    assert abs(m["headroom_stops"] - 2.0) < 0.01
    assert m["baseline_peak_nits"] == pytest.approx(1000.0, rel=1e-3)


def test_diffuse_white_share_is_not_fooled_by_float_round_trip():
    """A frame stored at exactly 203 nits has none of its pixels ABOVE 203."""
    flat = np.full((3, 32, 32), 203.0 / NETWORK_PEAK_NITS, dtype=np.float32)
    m = server.measure(flat, flat, *_masks(32, (slice(0, 2), slice(None)),
                                           (slice(30, 32), slice(None))))
    assert m["above_diffuse_white_pct"] == 0.0


def test_display_map_clips_at_the_chosen_peak_and_reveals_above_it():
    hdr = _frame(203.0, 4000.0)

    at_sdr = server.display_map(hdr, 203.0)
    at_hdr = server.display_map(hdr, 4000.0)

    # At an SDR peak the specular is blown; the midtone sits near mid-grey.
    assert at_sdr[11, 11, 0] == pytest.approx(1.0, abs=1e-3)
    assert at_sdr[0, 0, 0] == pytest.approx(1.0, abs=1e-3)
    # Raising the display peak darkens the frame so the highlight has somewhere
    # to go -- that reveal is the whole point of the compare slider.
    assert at_hdr[0, 0, 0] < 0.5
    assert at_hdr[11, 11, 0] > at_hdr[0, 0, 0]


def test_png_data_url_is_a_real_png():
    import base64

    url = server.png_data_url(np.zeros((4, 4, 3), dtype=np.float32))
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_peak_headroom_goes_blind_on_a_clipped_plateau():
    """Why headroom_stops alone is not enough, stated as a test.

    Inverse-ACES maps a saturated SDR pixel to a fixed ceiling, so any large
    blown-white region puts the model AND the baseline on the same peak. On
    27 Aug 2026 the demo sunset read +0.00 stops while the same checkpoint
    lifted a small specular by 2.47 stops on a frame without the plateau.
    """
    size = 64
    plateau = (slice(0, 32), slice(None))
    model = _frame(203.0, 2551.9, size)
    base = _frame(203.0, 2551.9, size)
    # Both pinned at the ceiling across a big region...
    model[:, 0:32, :] = 2551.9 / NETWORK_PEAK_NITS
    base[:, 0:32, :] = 2551.9 / NETWORK_PEAK_NITS
    # ...while the model genuinely lifts the midtones it was gated onto.
    model[:, 40:48, :] = 800.0 / NETWORK_PEAK_NITS
    base[:, 40:48, :] = 200.0 / NETWORK_PEAK_NITS
    hi, sh = _masks(size, (slice(40, 48), slice(None)), (slice(56, 60), slice(None)))

    m = server.measure(model, base, hi, sh)

    assert m["headroom_stops"] == 0.0, "peak-vs-peak is blind here -- that is the point"
    assert m["headroom_highlight_stops"] == pytest.approx(2.0, abs=0.05)
    assert m["departure_rms_stops"] > 0.3, "the frame plainly changed"


def test_departure_is_zero_when_the_model_reproduces_the_baseline():
    """recovery_mode 'off' must be indistinguishable from doing nothing."""
    frame = _frame(203.0, 4000.0)
    hi, sh = _masks(64, (slice(10, 14), slice(10, 14)), (slice(60, 62), slice(None)))

    m = server.measure(frame, frame.copy(), hi, sh)

    assert m["departure_rms_stops"] == 0.0
    assert m["headroom_stops"] == 0.0
    assert m["headroom_highlight_stops"] == 0.0


def test_masked_headroom_is_nan_not_zero_when_no_mask_fired():
    """An empty mask means "no measurement", which is not the same as "no gain"."""
    import math

    frame = _frame(203.0, 4000.0)
    hi, sh = _masks(64, (slice(0, 0), slice(None)), (slice(0, 0), slice(None)))

    m = server.measure(frame, frame.copy(), hi, sh)

    assert math.isnan(m["headroom_highlight_stops"])
