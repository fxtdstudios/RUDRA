"""The checkpoints committed under checkpoints/ load, and the docs match them.

This exists because the load snippet in `checkpoints/README.md` was wrong the
first time it was written: the file is a payload dict, not a bare state dict,
so `load_state_dict(torch.load(path))` raises with ~100 missing keys and four
unexpected ones. Documentation that does not run is documentation that drifts,
so the snippet is executed here rather than trusted.

The registry checks came later and for the same reason. `checkpoints/models.json`
told the viewer v6 had 4,772,485 parameters; the file has 4,770,117, and the
paper had copied the wrong figure too. Nothing compared either claim to the
tensor it describes until this ran.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"
SUMS = REPO / "checkpoints" / "SHA256SUMS"
DOC = REPO / "checkpoints" / "README.md"

pytestmark = pytest.mark.skipif(
    not CKPT.exists(), reason="committed checkpoint not present")

# The documentation checks below need no torch, so they run everywhere --
# including the environments where the drift they guard against goes unnoticed.
needs_torch = pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed")


@pytest.fixture(scope="module")
def payload():
    import torch
    return torch.load(CKPT, map_location="cpu", weights_only=False)


@needs_torch
def test_it_is_a_payload_dict_not_a_bare_state_dict(payload):
    # The shape the README's snippet has to match.
    assert isinstance(payload, dict)
    assert "model" in payload and "config" in payload
    assert not any(k.endswith(".weight") for k in payload)


@needs_torch
def test_the_readme_snippet_loads_it(payload):
    from rudra.sdr2hdr import SDR2HDRNet
    model = SDR2HDRNet.from_config(payload.get("config", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()


@needs_torch
def test_the_gate_is_actually_in_there(payload):
    from rudra.sdr2hdr import SDR2HDRNet
    assert any(k.startswith("shadow_gate.") for k in payload["model"]), \
        "this is the backbone, not the shipped model"
    model = SDR2HDRNet.from_config(payload.get("config", {}))
    model.load_state_dict(payload["model"], strict=True)
    gate = sum(p.numel() for p in model.shadow_gate.parameters())
    total = sum(p.numel() for p in model.parameters())
    # The figures the paper and both READMEs quote.
    assert gate == 21121, gate
    assert total - gate == 1196197, total - gate


@needs_torch
def test_it_runs_a_frame_and_emits_a_weight_in_range(payload):
    import torch
    from rudra.sdr2hdr import SDR2HDRNet
    model = SDR2HDRNet.from_config(payload.get("config", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    frame = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        weight = model.predict_shadow_weight(frame)
        out = model(frame)
    assert torch.isfinite(weight).all()
    assert float(weight.min()) >= 0.0 and float(weight.max()) <= 1.0
    assert torch.isfinite(out.hdr).all()


@needs_torch
def test_checksums_match(payload):
    listed = dict(
        (name, digest)
        for digest, name in (line.split() for line in
                             SUMS.read_text().split("\n") if line.strip()))
    got = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert got == listed["sdr2hdr_shadow_v1.pt"]


def test_the_readme_quotes_the_real_digest():
    got = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert got in DOC.read_text(encoding="utf-8"), \
        "checkpoints/README.md quotes a stale sha256"


def test_the_readme_snippet_is_the_one_that_is_tested():
    # If someone edits the snippet, they have to keep these lines in it.
    doc = DOC.read_text(encoding="utf-8")
    for fragment in ('weights_only=False',
                     'SDR2HDRNet.from_config(payload.get("config", {}))',
                     'load_state_dict(payload["model"], strict=True)'):
        assert fragment in doc, f"README snippet no longer contains: {fragment}"


# ---------------------------------------------------------------------------
# the registry: every committed model, and the claims made about it
# ---------------------------------------------------------------------------
REGISTRY = REPO / "checkpoints" / "models.json"


def _registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_lists_files_that_exist():
    for entry in _registry()["models"]:
        assert (REPO / "checkpoints" / entry["file"]).is_file(), entry["file"]


def test_registry_default_is_a_loadable_model():
    reg = _registry()
    default = reg["default"]
    match = [m for m in reg["models"] if m["file"] == default]
    assert match, f"default {default!r} is not in the registry"
    # A default the viewer cannot load is worse than no default: it starts in
    # demo mode and says nothing about why.
    assert match[0]["kind"] == "sdr2hdr", match[0]


@needs_torch
@pytest.mark.parametrize("entry", [m for m in _registry()["models"]
                                   if m.get("kind") == "sdr2hdr"],
                         ids=lambda m: m["file"])
def test_every_sdr2hdr_model_loads_and_runs(entry):
    import torch
    from rudra.sdr2hdr import SDR2HDRNet
    payload = torch.load(REPO / "checkpoints" / entry["file"],
                         map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(payload.get("config", {}))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 64, 64))
    assert torch.isfinite(out.hdr).all()


@needs_torch
@pytest.mark.parametrize("entry", [m for m in _registry()["models"]
                                   if m.get("kind") == "sdr2hdr"
                                   and m.get("params") is not None],
                         ids=lambda m: m["file"])
def test_parameter_counts_match_the_registry(entry):
    import torch
    from rudra.sdr2hdr import SDR2HDRNet
    payload = torch.load(REPO / "checkpoints" / entry["file"],
                         map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(payload.get("config", {}))
    model.load_state_dict(payload["model"], strict=True)
    got = sum(p.numel() for p in model.parameters())
    assert got == entry["params"], (
        f"{entry['file']}: models.json claims {entry['params']}, "
        f"the tensors say {got}")


def test_the_paper_quotes_the_measured_parameter_counts():
    # v5 and v6 are quoted in §3.4 of the paper; the gate's 21,121 appears in
    # the abstract, §6.2 and both READMEs.
    paper = (REPO / "PAPER_DRAFT_2026-08-29.md").read_text(encoding="utf-8")
    by_file = {m["file"]: m for m in _registry()["models"]}
    for name, key in (("sdr2hdr_image_v5.pt", "1,196,197"),
                      ("sdr2hdr_image_v6.pt", "4,770,117")):
        assert f"{by_file[name]['params']:,}" == key
        assert key in paper, f"the paper no longer quotes {key} for {name}"


@needs_torch
def test_the_temporal_refiner_is_not_offered_to_the_viewer():
    # It is not an SDR2HDRNet. The registry marks it so, and the viewer filters
    # on that mark; if this ever loads, the filter has become optional.
    import torch
    from rudra.sdr2hdr import SDR2HDRNet
    entry = [m for m in _registry()["models"] if m["kind"] == "temporal"][0]
    payload = torch.load(REPO / "checkpoints" / entry["file"],
                         map_location="cpu", weights_only=False)
    model = SDR2HDRNet.from_config(payload.get("config", {}))
    with pytest.raises(RuntimeError):
        model.load_state_dict(payload["model"], strict=True)
