"""The weights are non-commercial, and nothing may quietly say otherwise.

Until 5 September 2026 six checkpoints shipped under the repository's blanket
Apache 2.0 while HdM-HDR-2014 -- one of their training sources -- is free for
academic use only and requires a separate agreement with HdM Stuttgart for
commercial use. That told every reader commercial use was fine, which FXTD
Studios was not in a position to say.

The failure mode this guards is somebody tidying `license: other` back to
`license: apache-2.0` because it looks inconsistent with the repository. It
is supposed to look inconsistent. These are two different things.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "docs" / "HUB_MODEL_CARD.md"


def frontmatter(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---"), f"{path} has no YAML frontmatter"
    return yaml.safe_load(text.split("---")[1])


def test_the_weights_carry_their_own_licence():
    licence = REPO / "checkpoints" / "LICENSE"
    assert licence.is_file(), "checkpoints/LICENSE is gone"
    body = licence.read_text(encoding="utf-8")
    for needle in ("NON-COMMERCIAL", "HdM", "CC BY 4.0", "Poly Haven"):
        assert needle in body, f"checkpoints/LICENSE no longer mentions {needle}"


def test_the_weights_licence_actually_restricts_commercial_use():
    body = (REPO / "checkpoints" / "LICENSE").read_text(encoding="utf-8")
    assert "FOR NON-COMMERCIAL PURPOSES ONLY" in body, (
        "the grant no longer says non-commercial. If that was deliberate, the "
        "HdM term has to have been settled first -- it is a contract accepted "
        "at download, not a term FXTD Studios can waive.")
    # Research use must stay explicitly allowed, or the licence quietly blocks
    # the reproduction the paper asks readers to attempt.
    for allowed in ("research", "teaching", "evaluation", "benchmarking"):
        assert allowed in body.lower(), f"non-commercial use '{allowed}' is no longer named"


def test_the_notice_carries_the_required_attribution():
    notice = REPO / "NOTICE"
    assert notice.is_file(), "NOTICE is gone; it carries the licence's Required Notice"
    body = notice.read_text(encoding="utf-8")
    assert "Netflix" in body, "the Chimera CC BY 4.0 attribution is required"
    assert "checkpoints/LICENSE" in body, "NOTICE must point at the weights licence"


def test_the_model_card_does_not_claim_apache():
    meta = frontmatter(CARD)
    assert meta["license"] != "apache-2.0", (
        "the HuggingFace card claims Apache 2.0 for the weights again. They "
        "carry an HdM restriction FXTD Studios cannot waive, and since 24 Sep "
        "2026 the code is non-commercial too. See checkpoints/LICENSE.")
    assert meta["license"] == "other"
    assert "noncommercial" in meta.get("license_name", ""), (
        f"license_name is {meta.get('license_name')!r}; it should say "
        f"noncommercial so the HuggingFace listing is not misleading")
    assert meta.get("license_link", "").endswith("checkpoints/LICENSE")


def test_the_readme_separates_the_two():
    body = (REPO / "README.md").read_text(encoding="utf-8")
    assert "checkpoints/LICENSE" in body, (
        "the README's licence section no longer points at the weights licence")
    assert "non-commercial" in body.lower(), (
        "the README no longer says the weights are non-commercial")


def test_the_code_is_non_commercial():
    """24 Sep 2026: the code moved from Apache 2.0 to PolyForm Noncommercial 1.0.0."""
    body = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert body.startswith("# PolyForm Noncommercial License 1.0.0"), "LICENSE is no longer PolyForm Noncommercial 1.0.0"
    assert "https://polyformproject.org/licenses/noncommercial/1.0.0" in body
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "Required Notice: Copyright 2026 FXTD Studios" in notice, "the licence's Required Notice line is gone"
    for path in ("README.md", "docs/HUB_MODEL_CARD.md", "checkpoints/LICENSE"):
        text = (REPO / path).read_text(encoding="utf-8")
        assert "PolyForm Noncommercial" in text, f"{path} does not name the code's licence"
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "Code is Apache 2.0" not in readme
