"""Dynamic HDR metadata analysis for RUDRA output (torch-free).

Generates per-shot Dolby Vision L1 trim analysis and HDR10+ (ST 2094-40
style) scene statistics directly from RUDRA's linear frames — the step every
competitor (Runway Ruby, Topaz Hyperion, Beeble SwitchHDR) leaves to a
separate mastering pass or skips entirely. Static CTA-861.3 MaxCLL/MaxFALL
are computed here too, on max(R,G,B) as the spec requires (the legacy
``training/export_hdr10.py`` computed them on Rec.2020 luminance, which
understates both).

Outputs:
  - a canonical RUDRA sidecar JSON with everything in explicit nits AND
    12-bit PQ codes (the honest, self-describing record);
  - a `dovi_tool generate` config JSON (profile 8.1) carrying per-shot L1
    min/mid/max — pipe straight into dovi_tool to produce an RPU;
  - an HDR10+-style scene JSON (percentile distribution + MaxSCL per shot).
    Field units are documented in-file; verify against your injector's
    version before muxing (schemas in the wild vary by tool release).

All statistics use max(R,G,B) per pixel ("MaxRGB"), matching Dolby L1 and
CTA-861.3 practice, not luma — a specular red highlight is as bright to the
tone-mapping display as a white one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..hdr10 import pq_oetf

__all__ = [
    "FrameStats", "analyze_frame", "detect_shots", "maxcll_maxfall",
    "l1_per_shot", "to_dovi_generate_json", "to_hdr10plus_json",
    "to_rudra_sidecar", "write_all_sidecars",
]

_PERCENTILES = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.98)


@dataclass
class FrameStats:
    """MaxRGB statistics of one linear frame, in nits."""
    index: int
    min_nits: float
    avg_nits: float
    max_nits: float
    maxscl_nits: tuple[float, float, float]      # per-channel max
    percentiles_nits: tuple[float, ...]           # at _PERCENTILES
    log_hist: np.ndarray = field(repr=False, default=None)  # scene-cut feature


def _pq12(nits) -> int | list[int]:
    code = np.round(pq_oetf(np.asarray(nits, dtype=np.float64)) * 4095.0)
    out = np.clip(code, 0, 4095).astype(int)
    return out.tolist() if out.ndim else int(out)


def analyze_frame(rgb_nits: np.ndarray, index: int = 0, hist_bins: int = 64) -> FrameStats:
    """Analyze one (H, W, 3) linear RGB frame in absolute nits."""
    arr = np.asarray(rgb_nits, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr.shape}")
    arr = np.maximum(np.nan_to_num(arr, nan=0.0, posinf=10000.0), 0.0)
    max_rgb = arr.max(axis=2)
    log_l = np.log2(np.maximum(max_rgb, 1e-4))
    hist, _ = np.histogram(log_l, bins=hist_bins, range=(-14.0, 21.0))
    hist = hist.astype(np.float64) / max(1, max_rgb.size)
    return FrameStats(
        index=index,
        min_nits=float(max_rgb.min()),
        avg_nits=float(max_rgb.mean()),
        max_nits=float(max_rgb.max()),
        maxscl_nits=tuple(float(v) for v in arr.reshape(-1, 3).max(axis=0)),
        percentiles_nits=tuple(float(v) for v in np.percentile(max_rgb, _PERCENTILES)),
        log_hist=hist,
    )


def detect_shots(stats: list[FrameStats], threshold: float = 0.35) -> list[tuple[int, int]]:
    """Split a frame sequence into shots via log-luminance histogram distance.

    Returns [(start_index, length), ...] covering all frames. ``threshold``
    is the chi-square-like histogram distance that declares a cut; 0.35 is
    conservative (favors fewer, longer shots — safer for L1 trims, which
    must never change mid-shot or the grade visibly pumps).
    """
    if not stats:
        return []
    cuts = [0]
    for prev, cur in zip(stats, stats[1:]):
        a, b = prev.log_hist, cur.log_hist
        if a is None or b is None:
            continue
        dist = float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-9)))
        if dist > threshold:
            cuts.append(cur.index - stats[0].index)
    cuts.append(len(stats))
    return [(cuts[i], cuts[i + 1] - cuts[i]) for i in range(len(cuts) - 1)]


def maxcll_maxfall(stats: list[FrameStats]) -> tuple[int, int]:
    """CTA-861.3 static metadata from MaxRGB stats (ceil to integer nits)."""
    if not stats:
        raise ValueError("no frames analyzed")
    max_cll = int(np.ceil(max(s.max_nits for s in stats)))
    max_fall = int(np.ceil(max(s.avg_nits for s in stats)))
    return max_cll, max_fall


def l1_per_shot(stats: list[FrameStats], shots: list[tuple[int, int]]) -> list[dict]:
    """Dolby Vision L1 analysis per shot: min/mid/max as 12-bit PQ codes."""
    out = []
    for start, length in shots:
        window = stats[start:start + length]
        out.append({
            "start": start,
            "duration": length,
            "min_pq": _pq12(min(s.min_nits for s in window)),
            "avg_pq": _pq12(float(np.mean([s.avg_nits for s in window]))),
            "max_pq": _pq12(max(s.max_nits for s in window)),
            "min_nits": min(s.min_nits for s in window),
            "avg_nits": float(np.mean([s.avg_nits for s in window])),
            "max_nits": max(s.max_nits for s in window),
        })
    return out


def to_dovi_generate_json(
    stats: list[FrameStats],
    shots: list[tuple[int, int]] | None = None,
    mastering_peak_nits: float = 1000.0,
    mastering_min_nits: float = 0.0001,
) -> dict:
    """Config consumable by ``dovi_tool generate --json`` (profile 8.1)."""
    shots = shots or detect_shots(stats)
    max_cll, max_fall = maxcll_maxfall(stats)
    return {
        "cm_version": "V40",
        "profile": "8.1",
        "length": len(stats),
        "level6": {
            "max_display_mastering_luminance": int(round(mastering_peak_nits)),
            "min_display_mastering_luminance": int(round(mastering_min_nits * 10000)),
            "max_content_light_level": max_cll,
            "max_frame_average_light_level": max_fall,
        },
        "shots": [
            {
                "start": shot["start"],
                "duration": shot["duration"],
                "metadata_blocks": [{
                    "Level": 1,
                    "min_pq": shot["min_pq"],
                    "max_pq": shot["max_pq"],
                    "avg_pq": shot["avg_pq"],
                }],
            }
            for shot in l1_per_shot(stats, shots)
        ],
    }


def to_hdr10plus_json(
    stats: list[FrameStats],
    shots: list[tuple[int, int]] | None = None,
    target_display_nits: int = 400,
) -> dict:
    """ST 2094-40-style per-scene statistics (HDR10+ analysis).

    Luminance fields are in cd/m² (explicitly labeled). Injector JSON schemas
    differ across tool versions, so this is the analysis payload — remap the
    field names to your injector's schema if it disagrees rather than
    re-measuring the frames.
    """
    shots = shots or detect_shots(stats)
    scenes = []
    for scene_id, (start, length) in enumerate(shots):
        window = stats[start:start + length]
        scenes.append({
            "SceneId": scene_id,
            "SceneFirstFrameIndex": start,
            "SceneFrameNumbers": length,
            "TargetedSystemDisplayMaximumLuminance": target_display_nits,
            "LuminanceParameters": {
                "Units": "cd/m2",
                "AverageMaxRGB": float(np.mean([s.avg_nits for s in window])),
                "MaxSCL": [float(max(s.maxscl_nits[c] for s in window)) for c in range(3)],
                "LuminanceDistributions": {
                    "DistributionIndex": list(_PERCENTILES),
                    "DistributionValues": [
                        float(np.max([s.percentiles_nits[i] for s in window]))
                        for i in range(len(_PERCENTILES))
                    ],
                },
            },
        })
    return {
        "JSONInfo": {"Generator": "RUDRA delivery", "Version": "1.0", "Units": "cd/m2"},
        "SceneInfoSummary": {"SceneFirstFrameIndex": [s["SceneFirstFrameIndex"] for s in scenes],
                             "SceneFrameNumbers": [s["SceneFrameNumbers"] for s in scenes]},
        "SceneInfo": scenes,
    }


def to_rudra_sidecar(stats: list[FrameStats], shots: list[tuple[int, int]] | None = None) -> dict:
    """Canonical self-describing sidecar: everything in nits AND 12-bit PQ."""
    shots = shots or detect_shots(stats)
    max_cll, max_fall = maxcll_maxfall(stats)
    return {
        "generator": "rudra.delivery.metadata",
        "convention": "MaxRGB per pixel; nits absolute; pq codes 12-bit (0-4095)",
        "frame_count": len(stats),
        "max_cll_nits": max_cll,
        "max_fall_nits": max_fall,
        "shots": l1_per_shot(stats, shots),
        "frames": [
            {
                "index": s.index,
                "min_nits": s.min_nits, "avg_nits": s.avg_nits, "max_nits": s.max_nits,
                "min_pq": _pq12(s.min_nits), "avg_pq": _pq12(s.avg_nits), "max_pq": _pq12(s.max_nits),
                "maxscl_nits": list(s.maxscl_nits),
                "percentiles": {str(p): v for p, v in zip(_PERCENTILES, s.percentiles_nits)},
            }
            for s in stats
        ],
    }


def write_all_sidecars(
    stats: list[FrameStats],
    output_stem: str | Path,
    mastering_peak_nits: float = 1000.0,
    shot_threshold: float = 0.35,
) -> dict[str, Path]:
    """Write the three sidecars next to ``output_stem`` and return their paths."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    shots = detect_shots(stats, threshold=shot_threshold)
    paths = {
        "rudra": stem.with_name(stem.name + "_hdr_analysis.json"),
        "dovi": stem.with_name(stem.name + "_dovi_generate.json"),
        "hdr10plus": stem.with_name(stem.name + "_hdr10plus_scenes.json"),
    }
    paths["rudra"].write_text(json.dumps(to_rudra_sidecar(stats, shots), indent=2), encoding="utf-8", newline="\n")
    paths["dovi"].write_text(
        json.dumps(to_dovi_generate_json(stats, shots, mastering_peak_nits), indent=2), encoding="utf-8", newline="\n")
    paths["hdr10plus"].write_text(json.dumps(to_hdr10plus_json(stats, shots), indent=2), encoding="utf-8", newline="\n")
    return paths
