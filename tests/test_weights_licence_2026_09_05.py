"""The weights are not Apache 2.0, and nothing may quietly say they are.

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
    for needle in ("HdM", "COMMERCIAL", "CC BY 4.0", "Poly Haven"):
        assert needle in body, f"checkpoints/LICENSE no longer mentions {needle}"


def test_the_notice_carries_the_required_attribution():
    notice = REPO / "NOTICE"
    assert notice.is_file(), "NOTICE is gone; Apache 2.0 section 4(d) wants it"
    body = notice.read_text(encoding="utf-8")
    assert "Netflix" in body, "the Chimera CC BY 4.0 attribution is required"
    assert "checkpoints/LICENSE" in body, "NOTICE must point at the weights licence"


def test_the_model_card_does_not_claim_apache():
    meta = frontmatter(CARD)
    assert meta["license"] != "apache-2.0", (
        "the HuggingFace card claims Apache 2.0 for the weights again. It is "
        "the code that is Apache 2.0; the weights carry an HdM restriction "
        "FXTD Studios cannot waive. See checkpoints/LICENSE.")
    assert meta["license"] == "other"
    assert meta.get("license_link", "").endswith("checkpoints/LICENSE")


def test_the_readme_separates_the_two():
    body = (REPO / "README.md").read_text(encoding="utf-8")
    assert "checkpoints/LICENSE" in body, (
        "the README's licence section no longer points at the weights licence")
