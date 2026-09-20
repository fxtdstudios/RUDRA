"""The gate, end to end, and the resume that keeps a long run from being wasted.

Every v02 number came out of `gate_temporal_oracle.py` and none of them had a
test that ran the script. This builds a small corpus the way the renderer
builds one -- sdr/, hdr/, meta/ and the `_ingest_config.json` sentinel -- runs
the gate over it, and checks the shape of what comes back.

The resume half exists because of two lost runs. On 6 Sep 2026 a 12-clip RAFT
pass was OOM-killed with no traceback; on 7 Sep another was lost when the
sandbox holding it was reclaimed. Both had scored clips at the time and both
wrote nothing, because the output was dumped once at the end. Rows are now
flushed after every clip, and `--resume` picks up from them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from pipeline.hdr_io import HDR_IO_VERSION, HDRStorage, encode_hdr_u16  # noqa: E402


FRAMES = 3
SIZE = (160, 256)          # >= 128 a side, so any flow backend would fit too


def build_clip(root: Path, name: str, seed: int) -> None:
    """One clip directory in the renderer's layout."""
    rng = np.random.default_rng(seed)
    storage = HDRStorage(mode="log2_extended", ceiling_nits=1_000_000.0)
    for sub in ("sdr", "hdr", "meta"):
        (root / name / sub).mkdir(parents=True, exist_ok=True)

    # Band-limited, so the frames look like a scene rather than like noise.
    small = rng.random((SIZE[0] // 16, SIZE[1] // 16, 3)).astype(np.float32)
    base = cv2.resize(small, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_CUBIC)

    for index in range(FRAMES):
        stem = f"tif_0000000_{name}_{index:05d}"
        # A pan, so consecutive frames genuinely differ.
        frame = np.roll(base, index * 3, axis=1)
        sdr = np.clip(frame, 0.0, 1.0)
        cv2.imwrite(str(root / name / "sdr" / f"{stem}.png"),
                    (sdr[..., ::-1] * 255.0 + 0.5).astype(np.uint8))
        code, _ = encode_hdr_u16(sdr.astype(np.float32) * 4.0, storage)
        cv2.imwrite(str(root / name / "hdr" / f"{stem}.png"), code[..., ::-1])
        (root / name / "meta" / f"{stem}.json").write_text(json.dumps({
            "stem": stem, "source": f"{name}.exr", "clip": 0, "frame": index,
            "pose": {"yaw": 10.0 + index * 0.8, "pitch": -2.0,
                     "roll": 0.5, "hfov": 75.0},
        }), encoding="utf-8")

    (root / "_ingest_config.json").write_text(json.dumps({
        "hdr_io_version": HDR_IO_VERSION,
        "storage": storage.as_dict(),
    }), encoding="utf-8")


def run_gate(clips: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "training" / "gate_temporal_oracle.py"),
         "--clips", str(clips),
         "--checkpoint", str(REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"),
         "--condition", "clean", "--no-cvvdp", "--out", str(out), *extra],
        capture_output=True, text=True, cwd=str(REPO))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("clips")
    for index, name in enumerate(("alpha_4k_c0", "beta_4k_c0")):
        build_clip(root, name, seed=index)
    return root


checkpoint_only = pytest.mark.skipif(
    not (REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt").is_file(),
    reason="shipped checkpoint not present")


@checkpoint_only
@pytest.mark.slow
def test_the_gate_runs_and_scores_every_clip(corpus, tmp_path):
    out = tmp_path / "gate.json"
    done = run_gate(corpus, out)
    assert done.returncode == 0, done.stderr[-2000:]
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [row["clip"] for row in rows] == ["alpha_4k_c0", "beta_4k_c0"]
    for row in rows:
        for arm in ("per_frame", "control", "aligned_mean", "oracle"):
            assert np.isfinite(row[arm]["pu21_db"])
        # The oracle consults ground truth; it cannot lose to the per-frame
        # reconstruction it is choosing among.
        assert row["oracle"]["pu21_db"] >= row["per_frame"]["pu21_db"] - 1e-6


@checkpoint_only
@pytest.mark.slow
def test_rows_are_written_before_the_run_ends(corpus, tmp_path):
    """The property both lost runs needed and did not have."""
    out = tmp_path / "gate.json"
    assert run_gate(corpus, out).returncode == 0
    partial = json.loads(out.read_text(encoding="utf-8"))[:1]
    out.write_text(json.dumps(partial), encoding="utf-8")

    done = run_gate(corpus, out, "--resume")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "1 clip(s) already scored" in done.stdout
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [row["clip"] for row in rows] == ["alpha_4k_c0", "beta_4k_c0"]


@checkpoint_only
@pytest.mark.slow
def test_a_truncated_output_starts_over_rather_than_scoring_a_subset(corpus, tmp_path):
    """Half a JSON file is the normal remains of a process that was killed.

    Reading it as 'already done' would silently report a gate number over
    whichever clips happened to finish, which is worse than losing the run.
    """
    out = tmp_path / "gate.json"
    out.write_text('[{"clip": "alpha_4k_c0", "per_f', encoding="utf-8")
    done = run_gate(corpus, out, "--resume")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "unreadable, starting fresh" in done.stdout
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 2


@checkpoint_only
@pytest.mark.slow
def test_the_confidence_combiner_reduces_to_the_mean_under_exact_poses(corpus, tmp_path):
    """THE PROPERTY THAT KEEPS THE POSE ARM A FIXED REFERENCE.

    The v02 decision compares estimated alignment against the renderer's exact
    poses. If a new combiner moved the pose arm too, every earlier pose number
    would be stale and the comparison would be between two moving things.

    Exact correspondence has no forward-backward residual, so every weight is
    1 inside the valid mask and the weighted sum is the mean. That has to hold
    numerically, not just in principle.
    """
    plain = tmp_path / "mean.json"
    weighted = tmp_path / "confidence.json"
    assert run_gate(corpus, plain, "--alignment", "pose").returncode == 0
    done = run_gate(corpus, weighted, "--alignment", "pose",
                    "--combiner", "confidence")
    assert done.returncode == 0, done.stderr[-2000:]

    a = {r["clip"]: r["aligned_mean"]["pu21_db"]
         for r in json.loads(plain.read_text(encoding="utf-8"))}
    b = {r["clip"]: r["aligned_mean"]["pu21_db"]
         for r in json.loads(weighted.read_text(encoding="utf-8"))}
    assert a.keys() == b.keys()
    for clip in a:
        assert a[clip] == pytest.approx(b[clip], abs=1e-4), clip
