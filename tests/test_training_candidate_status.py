import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ui'))
import server


def test_training_candidate_reports_complete_but_not_promoted(tmp_path, monkeypatch):
    run = tmp_path / 'outputs' / 'finetune_views_20260920'
    run.mkdir(parents=True)
    (run / 'assessment.json').write_text(json.dumps({'selected_seed':'seed20260920','promoted':False,'license':'Non-commercial v5 derivative'}))
    (run / 'gate_seed20260920').mkdir()
    (run / 'gate_seed20260920' / 'best.pt').write_bytes(b'x')
    monkeypatch.setattr(server, 'REPO', tmp_path)
    info = server.training_candidate()
    assert info['available'] is True
    assert info['promoted'] is False
    assert info['selected_seed'] == 'seed20260920'


def test_training_candidate_missing_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'REPO', tmp_path)
    assert server.training_candidate() == {'available':False,'promoted':False}
