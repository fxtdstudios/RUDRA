"""The native model package: every exported graph computes what eager computes.

tools/export_model.py re-expresses SDR2HDRNet's orchestration for TorchScript
and ONNX (the dataclass return, the curve head's Python-sized resize, and
area/quantile/median have no direct export). These pin that the re-expression
is exact for LibTorch and within tolerance for ONNX Runtime, for the shipped
checkpoint and for a model with every head enabled, and that the big-frame
branches (quantile subsampling above 1e6 pixels, non-integer area resize) are
exercised rather than assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from rudra.sdr2hdr import SDR2HDRNet  # noqa: E402
from tools import export_model as em  # noqa: E402

SHIPPED = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"


def _all_heads_checkpoint(path: Path) -> Path:
    torch.manual_seed(7)
    net = SDR2HDRNet(gate_conditioning=True, shadow_conditioning=True, curve_head=True)
    # Every head is zero-initialised so a fresh one is a no-op; give them real
    # weights or the parity check would compare constants.
    with torch.no_grad():
        for layer in (net.head, net.gate.mlp[-1], net.shadow_gate.mlp[-1], net.curve.mlp[-1]):
            layer.weight.normal_(0.0, 0.05)
            layer.bias.normal_(0.0, 0.05)
    config = {"base_channels": 32, "gate_conditioning": True, "shadow_conditioning": True,
              "curve_head": True, "corpus_ev": 0.0}
    torch.save({"config": config, "model": net.state_dict()}, path)
    return path


@pytest.mark.skipif(not SHIPPED.exists(), reason="shipped checkpoint not in this clone")
def test_shipped_checkpoint_package_is_complete_and_passes(tmp_path):
    manifest = em.build_package(SHIPPED, tmp_path)
    out = tmp_path / SHIPPED.stem
    for f in ("model.ts", "model.frame.onnx", "model.tile.onnx", "manifest.json",
              "golden/golden.json"):
        assert (out / f).is_file(), f
    assert manifest["contract"] == em.CONTRACT
    assert manifest["network"]["heads"] == {"residual_gate": False, "shadow_gate": True, "curve": False}
    assert manifest["parity"]["passed"] == {"torchscript": True, "onnx": True}
    # LibTorch runs the same kernels as eager: bit-exact, not merely close.
    assert max(manifest["parity"]["max_abs"]["torchscript"].values()) == 0.0
    golden = json.loads((out / "golden" / "golden.json").read_text())
    assert len(golden["frames"]) == 16
    first = golden["frames"][0]
    sdr = np.load(out / "golden" / first["sdr"]["file"])
    res = np.load(out / "golden" / first["residual"]["file"])
    assert sdr.shape[2] == 3 and res.shape == (1, 3, sdr.shape[0], sdr.shape[1])
    assert golden["stitch"]["tile_size"] == 128


def test_every_head_exports_and_matches(tmp_path):
    ckpt = _all_heads_checkpoint(tmp_path / "all_heads.pt")
    manifest = em.build_package(ckpt, tmp_path / "pkg", tile_golden=False)
    assert manifest["network"]["heads"] == {"residual_gate": True, "shadow_gate": True, "curve": True}
    assert manifest["network"]["curve_params"] == 9
    assert manifest["parity"]["passed"] == {"torchscript": True, "onnx": True}
    ts = manifest["parity"]["max_abs"]["torchscript"]
    assert max(ts.values()) <= em.TOLERANCE["torchscript"]["atol"]
    # The curve and the residual scale must actually be non-trivial, or this
    # test would pass on a model that exports constants.
    sample = em._to_tensor(em.golden_inputs()[0][1])
    ref = em.eager_reference(em.load_network(ckpt)[0], sample)
    assert np.abs(ref["curve_params"]).max() > 1e-3
    assert abs(float(ref["residual_scale"].ravel()[0]) - 1.0) > 1e-3


def test_big_frame_branches_match_eager(tmp_path):
    """Above 1e6 pixels frame_conditioning_stats subsamples before quantile, and
    1100x1000 -> 512 is a non-integer area resize. Frame pass only: the tile
    pass on a megapixel frame adds nothing the golden frames do not cover."""
    ckpt = _all_heads_checkpoint(tmp_path / "all_heads.pt")
    net, _ = em.load_network(ckpt)
    torch.manual_seed(3)
    sdr = torch.rand(1, 3, 1000, 1100) ** 2.0
    with torch.inference_mode():
        ref = (net.predict_residual_scale(sdr), net.predict_shadow_weight(sdr), net.predict_curve(sdr))
        ts = em.script_model(net)
        got = ts.frame_pass(sdr)
    for a, b in zip(ref, got):
        assert torch.equal(a.float(), b.float())

    em.export_onnx(net, tmp_path / "f.onnx", tmp_path / "t.onnx", 17)
    sess = ort.InferenceSession(str(tmp_path / "f.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"sdr": sdr.numpy()})
    tol = em.TOLERANCE["onnx"]
    for a, b in zip(ref, out):
        a = a.float().numpy()
        assert np.all(np.abs(a - b) <= tol["atol"] + tol["rtol"] * np.abs(a))


def test_area_matrix_is_adaptive_avg_pool():
    """The ONNX resize is two matmuls; it must bin exactly like area interpolate."""
    net = SDR2HDRNet()
    m = em.NativeModel(net, onnx_safe=True)
    torch.manual_seed(1)
    x = torch.rand(1, 3, 331, 517)
    got = m._area_onnx(x, 128)
    size = m._target_size(331, 517, 128)
    ref = torch.nn.functional.interpolate(x, size=size, mode="area")
    assert got.shape == ref.shape
    assert torch.max(torch.abs(got - ref)) < 1e-6


def test_failed_parity_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setitem(em.TOLERANCE, "onnx", {"atol": 0.0, "rtol": 0.0})
    ckpt = _all_heads_checkpoint(tmp_path / "all_heads.pt")
    with pytest.raises(SystemExit):
        em.build_package(ckpt, tmp_path / "pkg", tile_golden=False)
    assert not (tmp_path / "pkg" / "all_heads").exists()
