"""Tests for the direct pixel SDR-to-HDR path."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Keep collection alive on torch-free delivery machines (matches the other
# suites, which importorskip torch): rudra.delivery tests must still run there.
torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from rudra.sdr2hdr import SDR2HDRNet, TemporalHDRRefiner, canonicalize_sdr, sdr2hdr_loss
from training.build_sdr_hdr_manifest import build_manifest, scene_identity
from training.infer_sdr2hdr import predict_image
from training.sdr2hdr_dataset import SDRHDRDataset, degrade_sdr


def test_scene_identity_keeps_video_frames_together():
    a = scene_identity("tif_0000001_Chimera_DCI4k_HDR_00120")
    b = scene_identity("tif_0009999_Chimera_DCI4k_HDR_00121")
    assert a[0] == b[0]
    assert a[1] == 120 and b[1] == 121
    assert a[2] and b[2]


def test_manifest_is_paired_and_scene_safe():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdr, hdr, meta = root / "sdr", root / "hdr", root / "meta"
        for folder in (sdr, hdr, meta):
            folder.mkdir()
        names = [
            "tif_0000001_clip_A_00001.png",
            "tif_0000002_clip_A_00002.png",
            "exr_0000003_still_one.png",
            "exr_0000004_still_two.png",
        ]
        for name in names:
            cv2.imwrite(str(sdr / name), np.full((8, 8, 3), 128, np.uint8))
            cv2.imwrite(str(hdr / name), np.full((8, 8, 3), 32000, np.uint16))
        output = root / "manifest.jsonl"
        stats = build_manifest(sdr, hdr, output, meta, 0.25, 0.25, seed=7)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert stats["pairs"] == 4
        assert len({r["split"] for r in records if r["scene_id"] == "tif:clip_A"}) == 1
        assert all(Path(r["sdr_path"]).exists() and Path(r["hdr_path"]).exists() for r in records)


def test_dataset_reads_uint8_sdr_and_uint16_hdr():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sdr_path, hdr_path = root / "sdr.png", root / "hdr.png"
        cv2.imwrite(str(sdr_path), np.full((20, 30, 3), 128, np.uint8))
        cv2.imwrite(str(hdr_path), np.full((20, 30, 3), 32768, np.uint16))
        manifest = root / "manifest.jsonl"
        manifest.write_text(json.dumps({
            "asset_id": "x", "scene_id": "x", "split": "train",
            "sdr_path": str(sdr_path), "hdr_path": str(hdr_path),
        }) + "\n")
        sample = SDRHDRDataset(manifest, crop_size=16, augment=False)[0]
        assert sample["sdr"].shape == (3, 16, 16)
        assert sample["hdr"].shape == (3, 16, 16)
        assert abs(float(sample["sdr"].mean()) - 128 / 255) < 1e-4
        assert abs(float(sample["hdr"].mean()) - 32768 / 65535) < 1e-4


def test_image_model_loss_is_finite_and_backpropagates():
    model = SDR2HDRNet(base_channels=8)
    sdr = torch.rand(2, 3, 33, 47)
    target = torch.rand(2, 3, 33, 47)
    output = model(sdr)
    assert output.hdr.shape == target.shape
    losses = sdr2hdr_loss(output, sdr, target)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert model.head.weight.grad is not None


def test_zero_initialized_image_model_matches_physical_baseline():
    model = SDR2HDRNet(base_channels=8).eval()
    output = model(torch.rand(1, 3, 31, 45))
    assert torch.allclose(output.hdr, output.baseline, atol=1e-6)


def test_temporal_refiner_shape_and_zero_initial_correction():
    model = TemporalHDRRefiner(channels=8)
    sdr = torch.rand(1, 3, 3, 16, 20)
    initial = torch.rand_like(sdr)
    output = model(sdr, initial)
    assert output.shape == initial.shape
    assert torch.allclose(output, initial, atol=1e-5)


def test_degradation_remains_valid_sdr():
    output = degrade_sdr(torch.rand(3, 32, 32), strength=1.0)
    assert output.shape == (3, 32, 32)
    assert torch.isfinite(output).all()
    assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_sdr_transfer_canonicalization_is_bounded():
    code = torch.linspace(0, 1, 32).view(1, 1, 1, 32).repeat(1, 3, 1, 1)
    for transfer in ("srgb", "rec709", "gamma22", "gamma24"):
        output = canonicalize_sdr(code, transfer=transfer)
        assert output.shape == code.shape
        assert torch.isfinite(output).all()
        assert 0.0 <= float(output.min()) <= float(output.max()) <= 1.0


def test_tiled_inference_preserves_full_image_shape():
    model = SDR2HDRNet(base_channels=8).eval()
    sdr = torch.rand(1, 3, 65, 73)
    output = predict_image(model, sdr, preserve_outside=False, tile_size=32, overlap=8)
    assert output.shape == sdr.shape
    assert torch.isfinite(output).all()


def test_highlight_only_mode_preserves_deep_shadows():
    model = SDR2HDRNet(base_channels=8).eval()
    with torch.no_grad():
        model.head.bias[:3].fill_(1.0)
        sdr = torch.zeros(1, 3, 16, 16)
        all_recovery = model(sdr, recovery_mode="all").hdr
        highlight_only = model(sdr, recovery_mode="highlights").hdr
        baseline = model(sdr, recovery_mode="off").hdr
    assert float(all_recovery.max()) > float(highlight_only.max())
    assert torch.allclose(highlight_only, baseline, atol=1e-6)
