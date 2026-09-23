"""Which source a pair came from, under what licence, and whether it may train
weights that are sold.

The research weights are non-commercial because HdM-HDR-2014 and HdM-HFR-2017
(75.6% of v3) are academic-use only. ``rudra-studio`` is the same recipe
trained without them. This module is the one place that decides, from a
record's own paths, which side of that line it falls on -- so the filter and
the audit trail cannot disagree.

Matching is on lower-cased path text (scene_id, sdr/hdr/metadata paths,
source_path), first rule wins. Anything unmatched is ``unknown`` and NOT
commercial: a source nobody has classified does not go into sold weights.
Licence facts are those recorded in training/fetch_corpus.py.
"""
from __future__ import annotations

from typing import Iterable

# (source id, match keywords, licence, commercial_ok)
RULES: tuple[tuple[str, tuple[str, ...], str, bool], ...] = (
    ("hdm", ("stuttgart", "hdm-hdr", "hdm_hdr", "hdm-hfr", "hdm_hfr", "/hdm/", "hdm_commercial"),
     "HdM Stuttgart academic licence", False),
    ("netflix_sparks", ("sparks",), "CC BY 4.0 (Netflix Open Content)", True),
    ("netflix_sol_levante", ("sollevante", "sol_levante"), "CC BY 4.0 (Netflix Open Content)", True),
    ("netflix_chimera", ("chimera", "netflix"), "CC BY 4.0 (Netflix Open Content)", True),
    ("polyhaven_moves", ("rudra_v02", "pairs_moves", "hdri_moves"), "CC0 (rendered from Poly Haven)", True),
    ("polyhaven", ("polyhaven", "poly haven", "poly_haven"), "CC0", True),
    ("live_tmhdr", ("live_tmhdr", "live-tmhdr"), "LIVE-TMHDR, any purpose", True),
    ("liu_hdrv", ("hdrv", "liu_hdrv", "linkoping"), "CC BY-SA 4.0", True),
    ("pandora", ("pandora",), "see fetch_corpus.py", True),
    ("dvb_hdr", ("dvb_hdr", "dvb-hdr"), "see fetch_corpus.py", True),
    ("fxtd", ("fxtd",), "FXTD Studios own footage", True),
    ("sjtu_hdr", ("sjtu",), "academic only", False),
    ("fraunhofer_8k", ("fraunhofer",), "academic only", False),
    ("red_samples", ("red_samples", "/red/"), "no redistribution / training grant", False),
)


def _text(record: dict) -> str:
    keys = ("scene_id", "source_path", "sdr_path", "hdr_path", "metadata_path")
    return " ".join(str(record.get(k) or "") for k in keys).replace("\\", "/").lower()


def classify(record: dict) -> dict:
    """{'source': id, 'licence': text, 'commercial_ok': bool} for one record."""
    text = _text(record)
    for sid, words, licence, ok in RULES:
        if any(w in text for w in words):
            return {"source": sid, "licence": licence, "commercial_ok": ok}
    return {"source": "unknown", "licence": "unclassified", "commercial_ok": False}


def annotate(records: Iterable[dict]) -> list[dict]:
    """Write source / licence / commercial_ok onto every record (in place)."""
    out = []
    for record in records:
        record.update({f"licence_{k}" if k != "commercial_ok" else k: v
                       for k, v in classify(record).items()})
        out.append(record)
    return out
