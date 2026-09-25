"""24 Sep 2026: CP7 called paired_gate.py with no gate flags, so every
comparison printed GATE PASS and cp_results.json said "passed": true, including
shadow_v1's 0 of 537 on the ACES bench. The gates now live in
training/cp7_verdicts.py; paired_gate says "n/a" when no gate is asked for."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.paired_gate import compare, load, verdict

REPO = Path(__file__).resolve().parents[1]
HEADER = "clip,frame,pu_psnr_db,cvvdp_jod\n"


def _csv(path: Path, psnr: float, jod: float, n: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "".join(f"c,f{i},{psnr + (i % 5) * 0.01},{jod + (i % 3) * 0.001}\n"
                                     for i in range(n)), encoding="utf-8")
    return path


def test_no_gate_flag_is_a_report_not_a_pass(tmp_path):
    worse = _csv(tmp_path / "a.csv", 20.0, 7.0)
    base = _csv(tmp_path / "b.csv", 30.0, 8.0)
    r = compare(load(worse), load(base))
    assert verdict(r) is None
    out = tmp_path / "g.json"
    p = subprocess.run([sys.executable, str(REPO / "training" / "paired_gate.py"), "--a", str(worse),
                        "--b", str(base), "--out", str(out)], capture_output=True, text=True)
    assert p.returncode == 0 and "GATE n/a" in p.stdout and "PASS" not in p.stdout
    assert json.loads(out.read_text())["passed"] is None


def test_require_positive_fails_a_loss_and_exits_1(tmp_path):
    worse = _csv(tmp_path / "a.csv", 20.0, 7.0)
    base = _csv(tmp_path / "b.csv", 30.0, 8.0)
    assert verdict(compare(load(worse), load(base)), require_positive=True) is False
    p = subprocess.run([sys.executable, str(REPO / "training" / "paired_gate.py"), "--a", str(worse),
                        "--b", str(base), "--require-positive"], capture_output=True, text=True)
    assert p.returncode == 1 and "GATE FAIL" in p.stdout


def test_one_metric_down_fails_require_positive(tmp_path):
    # the N3 shape: PU21 up, CVVDP down
    a = _csv(tmp_path / "a.csv", 31.0, 7.9)
    b = _csv(tmp_path / "b.csv", 30.0, 8.0)
    assert verdict(compare(load(a), load(b)), require_positive=True) is False


def test_per_metric_regression_limits(tmp_path):
    a = _csv(tmp_path / "a.csv", 29.95, 7.99)       # -0.05 dB, -0.01 JOD
    b = _csv(tmp_path / "b.csv", 30.0, 8.0)
    r = compare(load(a), load(b))
    assert verdict(r, max_regression={"pu_psnr_db": 0.1, "cvvdp_jod": 0.02}) is True
    assert verdict(r, max_regression={"pu_psnr_db": 0.1, "cvvdp_jod": 0.005}) is False


def test_cp7_verdicts_scores_the_named_gates(tmp_path):
    from training.cp7_verdicts import score
    res = tmp_path / "cp_aces" / "results"
    _csv(res / "baseline.csv", 47.0, 9.5)
    _csv(res / "v4b.csv", 39.0, 9.0)
    _csv(res / "v4b_gate.csv", 39.0, 9.0)
    _csv(res / "shadow_v1.csv", 23.0, 8.8)
    oog = tmp_path / "cp_oog" / "results"
    _csv(oog / "baseline.csv", 26.0, 8.1)
    _csv(oog / "v4c.csv", 27.0, 8.2)
    s = score(tmp_path)
    g = s["gates"]
    assert g["N1/step4"]["status"] == "FAIL"
    assert g["N1/step5"]["status"] == "FAIL"          # identical outputs add nothing
    assert g["N3"]["status"] == "PASS"
    assert g["N7/oog"]["status"] == "not run"
    assert s["comparisons"]["aces/shadow_v1 vs baseline"]["passed"] is False
    assert s["comparisons"]["aces/v4b vs shadow_v1"]["passed"] is None   # report only


def test_export_and_predict_take_fp32():
    pytest.importorskip("torch")
    from training.infer_sdr2hdr import predict_image
    assert inspect.signature(predict_image).parameters["bf16"].default is True
    src = (REPO / "training" / "export_bench_pairs.py").read_text(encoding="utf-8")
    assert '"--precision"' in src and 'bf16=args.precision == "bf16"' in src
