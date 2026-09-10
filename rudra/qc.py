"""QC for an SDR-to-HDR reconstruction: PASS, FAIL, or UNMEASURED.

The rule this module exists to enforce is that a metric which could not be
measured FAILS. A null that reads as a pass is how a folder of PNGs once got
called a multipart EXR, and it is how a single-frame spread once reported a
coefficient of variation of zero and looked like proof.

Every threshold comes from ``configs/qc_reconstruction.json`` -- with the
reason it holds that value written beside it -- and none is hard-coded here.
Widening one to make a shot pass is not a QC decision; it is a decision for a
person, taken with the number in front of them.

The metrics are the ones that caught real defects on 10 Sep 2026:

  do-no-harm     unclipped picture came back a median 1.98x brighter
  chroma shift   unclipped hue moved about 10 JND
  hf excess      flat sky carried 1.9x the noise the source justified
  clip gain      the reason to run the model at all
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

LUMA_REC2020 = np.array([0.2627, 0.6780, 0.0593])
DIFFUSE_WHITE_NITS = 203.0
DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[1] / "configs" / "qc_reconstruction.json"

PASS, FAIL, UNMEASURED = "PASS", "FAIL", "UNMEASURED"


@dataclass
class Check:
    name: str
    status: str
    value: float | None
    threshold: float | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        # UNMEASURED is not a pass. That is the whole point of the class.
        return self.status == PASS


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    verdict: str = PASS

    def add(self, check: Check) -> None:
        self.checks.append(check)
        if not check.ok:
            self.verdict = FAIL

    def first_failure(self) -> Check | None:
        return next((c for c in self.checks if not c.ok), None)

    def to_dict(self) -> dict:
        return {"verdict": self.verdict,
                "checks": [vars(c) for c in self.checks]}


def load_thresholds(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DEFAULT_THRESHOLDS
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v["value"] for k, v in raw.items() if not k.startswith("_")}


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigmaX=sigma,
                            borderType=cv2.BORDER_REPLICATE).astype(np.float64)


def _uv(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :2] / np.maximum(rgb.sum(axis=-1, keepdims=True), 1e-12)


def check_frame(hdr_nits: np.ndarray, sdr_srgb: np.ndarray,
                thresholds: dict | None = None, ceiling_nits: float = 40_000.0) -> Report:
    """Score one reconstruction against its own source."""
    if hdr_nits.shape != sdr_srgb.shape:
        raise ValueError(f"shape mismatch: {hdr_nits.shape} vs {sdr_srgb.shape}")
    t = thresholds if thresholds is not None else load_thresholds()
    r = Report()

    hdr = np.asarray(hdr_nits, dtype=np.float64)
    sdr = np.asarray(sdr_srgb, dtype=np.float64)
    source = srgb_to_linear(sdr) * DIFFUSE_WHITE_NITS
    y_out, y_in = hdr @ LUMA_REC2020, source @ LUMA_REC2020

    # -- data integrity ---------------------------------------------------
    for key, count, label in (("nan_max", int(np.isnan(hdr).sum()), "NaN"),
                              ("inf_max", int(np.isinf(hdr).sum()), "Inf"),
                              ("negative_max", int((hdr < 0).sum()), "negative")):
        r.add(Check(label.lower(), PASS if count <= t[key] else FAIL, count, t[key],
                    f"{count} {label} sample(s)"))
    pinned = 100.0 * float((hdr >= ceiling_nits * 0.999).sum()) / hdr.size
    r.add(Check("at_ceiling_pct", PASS if pinned <= t["at_ceiling_max_pct"] else FAIL,
                pinned, t["at_ceiling_max_pct"],
                f"{pinned:.4f}% pinned at {ceiling_nits:.0f} nits"))

    # -- do no harm -------------------------------------------------------
    safe = sdr.max(axis=-1) < 250 / 255
    if safe.sum() < t["flat_region_min_px"]:
        r.add(Check("do_no_harm_median", UNMEASURED, None, t["do_no_harm_median_max"],
                    f"only {int(safe.sum())} unclipped px -- nothing to compare"))
        r.add(Check("do_no_harm_within_1pct", UNMEASURED, None,
                    t["do_no_harm_within_1pct_min"], "no unclipped sample"))
        r.add(Check("chroma_shift_uv", UNMEASURED, None,
                    t["chroma_shift_median_max_uv"], "no unclipped sample"))
    else:
        ratio = y_out[safe] / np.maximum(y_in[safe], 1e-9)
        med = float(np.median(ratio))
        within = float(np.mean(np.abs(ratio - 1.0) < 0.01))
        r.add(Check("do_no_harm_median", PASS if med <= t["do_no_harm_median_max"] else FAIL,
                    med, t["do_no_harm_median_max"], f"{med:.3f}x on unclipped picture"))
        r.add(Check("do_no_harm_within_1pct",
                    PASS if within >= t["do_no_harm_within_1pct_min"] else FAIL,
                    within, t["do_no_harm_within_1pct_min"], f"{100 * within:.1f}% within 1%"))
        duv = float(np.median(np.linalg.norm(_uv(hdr)[safe] - _uv(source)[safe], axis=-1)))
        r.add(Check("chroma_shift_uv",
                    PASS if duv <= t["chroma_shift_median_max_uv"] else FAIL,
                    duv, t["chroma_shift_median_max_uv"], f"median duv {duv:.4f}"))

    # -- did clipping actually get recovered ------------------------------
    clipped = sdr.max(axis=-1) >= 254 / 255
    if clipped.sum() < t["clipped_min_px"]:
        why = f"only {int(clipped.sum())} clipped px -- recovery cannot be judged"
        r.add(Check("clip_gain", UNMEASURED, None, t["clip_gain_min"], why))
        r.add(Check("clip_structure_pct", UNMEASURED, None, t["clip_structure_min_pct"], why))
    else:
        gain = float(y_out[clipped].mean() / max(y_in[clipped].mean(), 1e-9))
        r.add(Check("clip_gain", PASS if gain >= t["clip_gain_min"] else FAIL,
                    gain, t["clip_gain_min"], f"{gain:.2f}x on {100 * clipped.mean():.2f}% of px"))
        # Erode before measuring. The blur that suppresses pixel noise also
        # drags the surrounding picture into the edge of the clipped region, so
        # measuring over the raw mask reports the boundary's variation as if it
        # were recovered detail -- a uniformly filled blown patch scored 12.7%
        # against a 5% floor and passed.
        interior = cv2.erode(clipped.astype(np.uint8), np.ones((5, 5), np.uint8))
        interior = interior.astype(bool) if interior.sum() >= t["clipped_min_px"] else clipped
        low = _blur(y_out, 3.0)
        struct = float(np.std(low[interior]) / max(np.mean(low[interior]), 1e-9) * 100)
        r.add(Check("clip_structure_pct",
                    PASS if struct >= t["clip_structure_min_pct"] else FAIL,
                    struct, t["clip_structure_min_pct"], f"{struct:.1f}% relative variation"))

    # -- manufactured noise ------------------------------------------------
    grad = np.hypot(*np.gradient(_blur(y_in, 2.0)))
    # The flattest quarter OF THE USABLE PICTURE, and the percentile is taken
    # over that same set. Taken over the whole frame, a large blown area -- 39%
    # of a test frame, a sky on a real one -- sits at gradient exactly zero and
    # owns the entire bottom quarter, so excluding clipped pixels afterwards
    # left nothing at all and the noise check reported "only 0 flat px". A
    # clipped pixel is excluded because this measures how the expansion treats
    # SOURCE detail, and a clipped pixel has none left to treat.
    usable = (~clipped) & (y_in > 40.0)
    if usable.any():
        flat = usable & (grad <= np.percentile(grad[usable], 25))
    else:
        flat = usable
    if flat.sum() < t["flat_region_min_px"]:
        r.add(Check("hf_excess", UNMEASURED, None, t["hf_excess_max"],
                    f"only {int(flat.sum())} flat px -- noise cannot be measured"))
    else:
        xi, yo = y_in[flat], y_out[flat]
        # Bin count scales with the sample. Fixed at 60, a 5,000-pixel flat
        # region leaves most bins under the occupancy floor and the curve comes
        # back UNMEASURED -- which fails closed, correctly, but fails a frame
        # that was fine. At least 8 bins, never more than 60.
        bins = int(np.clip(xi.size // 400, 8, 60))
        edges = np.linspace(xi.min(), xi.max(), bins + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        med_curve = np.array([np.median(yo[(xi >= a) & (xi < b)])
                              if ((xi >= a) & (xi < b)).sum() > 20 else np.nan
                              for a, b in zip(edges[:-1], edges[1:])])
        good = ~np.isnan(med_curve)
        if good.sum() < 4:
            r.add(Check("hf_excess", UNMEASURED, None, t["hf_excess_max"],
                        "transfer curve could not be estimated"))
        else:
            # Amplification is POINTWISE. A spatially smoothed gain cannot
            # express a steep curve, and using one reported 7.3x where the
            # honest figure was 1.9x.
            slope = np.interp(xi, mid[good], np.gradient(med_curve[good], mid[good]))
            level = float(np.mean(_blur(y_out, 1.6)[flat]))
            predicted = float(np.std((y_in - _blur(y_in, 1.6))[flat] * slope) / level * 100)
            actual = float(np.std((y_out - _blur(y_out, 1.6))[flat]) / level * 100)
            excess = actual / max(predicted, 1e-9)
            r.add(Check("hf_excess", PASS if excess <= t["hf_excess_max"] else FAIL,
                        excess, t["hf_excess_max"],
                        f"{actual:.1f}% actual vs {predicted:.1f}% predicted"))
    return r


def format_report(report: Report, name: str, resolution: str) -> str:
    lines = [f"QC — {name}", f"  {resolution}", f"  VERDICT: {report.verdict}"]
    for c in report.checks:
        v = "—" if c.value is None else f"{c.value:.4g}"
        lines.append(f"    {c.status:<10} {c.name:<24} {v:>10}   {c.detail}")
    first = report.first_failure()
    if first:
        lines.append(f"  First failure: {first.name} ({first.status}) — {first.detail}")
    return "\n".join(lines)
