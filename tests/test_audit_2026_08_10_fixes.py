"""Regression tests pinning the AUDIT_2026-08-10 fixes.

Each test targets a specific defect from AUDIT_2026-07-15 / AUDIT_2026-08-10
that was live in the codebase despite the earlier audit documenting it.
These are the tests §4 of the 2026-08-10 report said were missing.
"""

import json
import os
import re
import sys
import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "training"))


# ── P0-1 (2026-07-15): offline flip augmentation ─────────────────────────────

def test_offline_flip_aug_flips_target_width_not_channels():
    """dataset_hdr.HDRPairDataset._load_pair must flip the (H,W,3) target on
    the WIDTH axis when the (C,h,w) latent is width-flipped — not the channel
    axis (the corpus-corrupting bug)."""
    from dataset_hdr import HDRPairDataset

    d = tempfile.mkdtemp()
    latent = np.zeros((16, 32, 32), np.float32)
    latent[:, :, :16] = 1.0                      # left half = 1
    target = np.zeros((256, 256, 3), np.float32)
    target[:, :128, 1] = 0.7                     # G on the LEFT half only
    np.savez_compressed(os.path.join(d, "pair.npz"), latent=latent, log_coded=target)

    ds = HDRPairDataset(pair_dir=d, augment=True)
    torch.manual_seed(0)
    flipped = None
    for _ in range(100):
        l, t = ds[0]
        if l[0, 0, 0].item() != 1.0:             # latent got width-flipped
            flipped = (l, t)
            break
    assert flipped is not None, "augmentation never triggered in 100 draws"
    l, t = flipped
    # After a correct width flip, G moves to the RIGHT half.
    assert t[0, 0, 1].item() == pytest.approx(0.0), \
        "target left edge still has G: target was not spatially flipped"
    assert t[0, 200, 1].item() == pytest.approx(0.7), \
        "target right half missing G: wrong flip axis"
    # And the channel order must be intact (no RGB reversal).
    assert t[0, 200, 0].item() == pytest.approx(0.0)
    assert t[0, 200, 2].item() == pytest.approx(0.0)


# ── P0-3 (2026-07-15): deploy filename contract ──────────────────────────────

def test_deploy_names_carry_rudra_prefix_everywhere():
    """Every deploy filename constructed by the orchestrators must match
    rudra_{size}_decoder_{tag}_ema.safetensors — the ComfyUI node contract."""
    sources = {
        "train_all_decoders.py": None,
        "build_all_decoders.py": None,
        "train_full_decoders.py": None,
    }
    pat = re.compile(r'f?"[^"\n]*decoder_[^"\n]*_ema\.safetensors"')
    offenders = []
    for fname in sources:
        path = os.path.join(REPO, "training", fname)
        text = open(path, encoding="utf-8").read()
        for m in pat.finditer(text):
            s = m.group(0)
            # deploy-name templates must include the rudra_ prefix unless they
            # are the trainer's internal checkpoint names ({size}_decoder_ema_best)
            if "rudra_" in s:
                continue
            if "_ema_best" in s or "_ema_step" in s:
                continue  # internal checkpoint naming, not the deploy contract
            offenders.append(f"{fname}: {s}")
    assert not offenders, f"un-prefixed deploy names: {offenders}"


# ── P0-4 (2026-07-15): resume must persist best tracking ─────────────────────

def test_trainers_persist_and_restore_best_psnr():
    """Both trainers must save best_psnr/stall_count in checkpoints and read
    them back on resume (so a resumed run can't clobber a better *_ema_best)."""
    for fname in ("train_turbo_decoder.py", "train_rudra.py"):
        text = open(os.path.join(REPO, "training", fname), encoding="utf-8").read()
        assert '"best_psnr"' in text, f"{fname}: best_psnr not saved in checkpoints"
        assert 'ckpt.get("best_psnr"' in text, f"{fname}: best_psnr not restored on resume"
        assert re.search(r"best_psnr\s*=\s*-1\.0", text) is None or \
               "resume_best_psnr" in text, f"{fname}: best_psnr still reset after resume"


# ── P0-5 (2026-07-15): wrappers must pass validation flags ───────────────────

def test_wrappers_pass_val_split_and_patience():
    for fname in ("build_all_decoders.py", "train_full_decoders.py"):
        text = open(os.path.join(REPO, "training", fname), encoding="utf-8").read()
        assert "--val_split" in text and "--patience" in text, \
            f"{fname}: does not pass --val_split/--patience (best ckpt would be train-batch noise)"


# ── P0-6 (2026-07-15): cleanup classifier ────────────────────────────────────

def test_cleanup_recognizes_all_trainer_dir_patterns(tmp_path):
    import cleanup_checkpoints as cc

    live_names = [
        "turbo_flux", "full_ltxvideo",            # build_all_decoders
        "flux_turbo_decoder", "sdxl_full_decoder", # train_all_decoders
        "flux1_turbo", "wan21_full", "flux1_lora", # retrain_all
        "sdr2hdr_image_v2",                        # train_sdr2hdr
        "flux_rudra_stage1",                       # train_rudra
    ]
    for name in live_names:
        d = tmp_path / name
        d.mkdir()
        assert cc.is_node_run(d), f"live run misclassified as dead: {name}"
    # A truly unknown dir with no artifacts stays dead.
    dead = tmp_path / "old_experiment_junk"
    dead.mkdir()
    assert not cc.is_node_run(dead)
    # ...unless it contains a best-EMA artifact.
    (dead / "x_ema_best.safetensors").write_bytes(b"0")
    assert cc.is_node_run(dead)


# ── NEW-2 (2026-08-10): ingest EOTF spec anchors ─────────────────────────────

def test_ingest_slog3_matches_sony_spec():
    from prepare_training_data import eotf_slog3

    grey = eotf_slog3(np.array([420.0 / 1023.0], np.float32))[0]
    assert grey == pytest.approx(0.18, abs=1e-4)
    top = eotf_slog3(np.array([1.0], np.float32))[0]
    assert top == pytest.approx(38.42, rel=1e-3)
    cut = 171.2102946929 / 1023.0
    lo = eotf_slog3(np.array([cut - 1e-6], np.float32))[0]
    hi = eotf_slog3(np.array([cut + 1e-6], np.float32))[0]
    assert abs(hi - lo) < 1e-4, "discontinuity at S-Log3 breakpoint"


def test_ingest_pq_is_absolute_10000_nits_ref_203():
    from prepare_training_data import eotf_pq

    # PQ(100 nits) = 0.50807; with 203-nit diffuse white -> 100/203
    v = eotf_pq(np.array([0.50807], np.float32))[0]
    assert v == pytest.approx(100.0 / 203.0, rel=1e-3)
    # PQ(1000 nits) = 0.7518 -> 1000/203
    v = eotf_pq(np.array([0.7518], np.float32))[0]
    assert v == pytest.approx(1000.0 / 203.0, rel=1e-3)


def test_ingest_hlg_diffuse_white_is_one():
    from prepare_training_data import eotf_hlg

    assert eotf_hlg(np.array([0.75], np.float32))[0] == pytest.approx(1.0, rel=1e-4)


# ── NEW-4 (2026-08-10): normalization cross-format radiometric consistency ──

def test_normalization_same_scene_same_radiometry_across_formats():
    from rudra.normalization import (
        normalize_to_scene_linear, encode_scene_linear_to_format,
    )
    from rudra.config import FORMAT_TO_ID

    torch.manual_seed(0)
    scene = torch.rand(1, 3, 32, 32) * 0.5 + 0.05
    scene[:, :, :8, :] = 16.0   # strong highlight band
    ref = None
    for fmt in ("slog3", "pq", "hlg", "logc4"):
        fid = FORMAT_TO_ID[fmt]
        coded = encode_scene_linear_to_format(scene, fid)
        dec = normalize_to_scene_linear(coded, fid)
        med = dec.median().item()
        if ref is None:
            ref = med
        assert med == pytest.approx(ref, rel=0.02), \
            f"{fmt} decodes the same scene to a different radiometric scale"
        assert dec.max().item() == pytest.approx(16.0, rel=0.05), \
            f"{fmt} lost the 16x highlight"


# ── NEW-5 (2026-08-10): global descriptor highlight block must fire ──────────

def test_global_descriptor_highlight_block_fires():
    from rudra.descriptor import RUDRADescriptor
    from rudra.config import FORMAT_TO_ID

    d = RUDRADescriptor()
    img = torch.full((1, 3, 64, 64), 0.18)
    img[:, :, :32, :] = 16.0
    desc = d(img, format_id=FORMAT_TO_ID["linear"], input_is_scene_linear=True)
    assert desc.shape[-1] == d.DR_RAW_DIM
    hl = desc[0, 12:18]
    assert hl.abs().max().item() > 0.1, \
        "highlight block near-zero on a 16x-white highlight (structurally dead)"
    flat = torch.full((1, 3, 64, 64), 0.18)
    hl0 = d(flat, format_id=FORMAT_TO_ID["linear"], input_is_scene_linear=True)[0, 12:18]
    assert hl0.abs().max().item() < 0.05, "highlight block fires on flat grey"


# ── NEW-6 (2026-08-10): manifest regex must match real MXF ingest naming ─────

def test_sequence_regex_matches_mxf_frame_naming():
    import build_sdr_hdr_manifest as m1
    import build_video_manifest as m2

    for regex in (m1.SEQUENCE_RE, m2.SEQUENCE_RE):
        mm = regex.match("mxf_0000042_clipA_f0003")
        assert mm, "mxf_*_f#### frame naming does not match SEQUENCE_RE"
        assert mm.group("sequence") == "clipA"
        assert int(mm.group("frame")) == 3
        mt = regex.match("tif_0000001_shot_0001")
        assert mt and mt.group("sequence") == "shot" and int(mt.group("frame")) == 1

    # Scene identity must not include the per-file ingest counter.
    sid_a, fr_a, vid_a = m1.scene_identity("mxf_0000042_clipA_f0003")
    sid_b, fr_b, vid_b = m1.scene_identity("mxf_0000043_clipA_f0004")
    assert sid_a == sid_b, "adjacent frames of one clip got different scene ids"
    assert vid_a and vid_b and fr_a == 3 and fr_b == 4


def test_split_by_scene_groups_frames_of_one_clip(tmp_path):
    """dataset_hdr.HDRPairDataset.split_by_scene must place all frames of a
    sequence in the same split (no per-frame leakage)."""
    from dataset_hdr import HDRPairDataset

    d = tmp_path / "pairs"
    d.mkdir()
    latent = np.zeros((4, 8, 8), np.float32)
    target = np.zeros((64, 64, 3), np.float32)
    # two clips x 6 frames, mxf naming with per-file unique ingest idx
    idx = 0
    for clip in ("clipA", "clipB"):
        for f in range(6):
            meta = json.dumps({"source": f"G:/data/mxf_{idx:07d}_{clip}_f{f:04d}.png"})
            np.savez_compressed(
                str(d / f"{clip}_{f}.npz"),
                latent=latent, log_coded=target, meta=meta,
            )
            idx += 1
    ds = HDRPairDataset(pair_dir=str(d), augment=False)
    train, val = ds.split_by_scene(val_split=0.5, seed=0)
    names = {p.name: ("train" if p in train else "val") for p in train + val}
    for clip in ("clipA", "clipB"):
        splits = {names[f"{clip}_{f}.npz"] for f in range(6)}
        assert len(splits) == 1, f"frames of {clip} leaked across splits: {splits}"


# ── NEW-3 (2026-08-10): pair-dir cross-VAE sentinel ──────────────────────────

def test_pair_dir_refuses_mismatched_vae(tmp_path, monkeypatch):
    from dataset_hdr import HDRPairDataset

    out = tmp_path / "pairs"

    class FakeVAE:
        def encode(self, t):
            return torch.zeros(1, 4, 8, 8)

    # First generation writes the sentinel.
    HDRPairDataset.generate_pairs(
        exr_dirs=[str(tmp_path / "none")], output_dir=str(out),
        vae=FakeVAE(), device="cpu", vae_id="flux:ae.safetensors",
    )
    assert (out / "_pair_config.json").exists()
    # Re-running with a different VAE id must refuse.
    with pytest.raises(RuntimeError, match="different"):
        HDRPairDataset.generate_pairs(
            exr_dirs=[str(tmp_path / "none")], output_dir=str(out),
            vae=FakeVAE(), device="cpu", vae_id="flux2:flux2-vae.safetensors",
        )


# ── LPIPS input range (losses.py) ────────────────────────────────────────────

def test_perceptual_loss_feeds_lpips_in_minus_one_one():
    from rudra import losses as L

    captured = {}

    class SpyLPIPS:
        def __call__(self, a, b):
            captured["min"] = min(a.min().item(), b.min().item())
            captured["max"] = max(a.max().item(), b.max().item())
            return torch.zeros(a.shape[0])

    pred = torch.zeros(2, 3, 16, 16)            # tonemaps to 0.0
    target = torch.full((2, 3, 16, 16), 100.0)  # tonemaps to ~0.99
    L.perceptual_loss(pred, target, lpips_net=SpyLPIPS())
    assert captured["min"] < -0.5, \
        f"LPIPS input min {captured['min']} — still being fed [0,1]"
    assert captured["max"] <= 1.0 + 1e-5


# ── Stage 2/3 apply_model target selection (source-level pin) ────────────────

def test_apply_model_paths_use_x0_target():
    """Every comfy_model.apply_model call site must set a used_apply_model
    flag and select the clean/x0 target when it is set."""
    for fname, needle in (
        ("train_hdr_lora.py", "used_apply_model"),
        ("train_rudra.py", "used_apply_model_s3"),
    ):
        text = open(os.path.join(REPO, "training", fname), encoding="utf-8").read()
        assert needle in text, f"{fname}: apply_model x0-target handling missing"
    # train_hdr_lora: the training loss must switch target on the flag.
    t = open(os.path.join(REPO, "training", "train_hdr_lora.py"), encoding="utf-8").read()
    assert "clean if used_apply_model else target" in t
    # train_rudra stage 3: mse must not compare pred to target_noise unconditionally.
    t2 = open(os.path.join(REPO, "training", "train_rudra.py"), encoding="utf-8").read()
    assert "target_x0_s3 if used_apply_model_s3 else target_noise" in t2
