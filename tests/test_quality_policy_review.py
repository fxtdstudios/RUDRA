import json

import pytest
import torch

from rudra.batch import digest
from rudra.recovery_policy import MODES, RecoveryPolicy
from training.review_quality_policy import compare, review
from training.train_quality_policy import summarize


def records():
    return [dict(scene_id=str(i), asset_id=str(i), condition=condition, split='val',
                 pu21_db=[30., 42. if i == 0 else 39., 35., 40.],
                 cvvdp_jod=[7., 9.2 if i == 0 else 8.9, 8., 9.])
            for condition in ('clean', 'hard') for i in range(2)]


def test_scene_regressions_not_hidden_by_positive_average():
    report = compare(records(), [1] * 4)
    assert report['clean']['pu21_db']['mean_delta'] == .5
    assert report['clean']['pu21_db']['regressed'] == 1
    assert report['clean']['pu21_db']['worst_scenes'][0]['scene_id'] == '1'
    assert report['clean']['mode_counts'] == {'highlights': 2}
    assert report == compare(records(), [1] * 4)
    duplicated = records() + [records()[0]]
    with pytest.raises(ValueError, match='one frame per scene'):
        compare(duplicated, [1] * 5)


def test_saved_checkpoint_replay_and_mismatch_detection(tmp_path):
    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data))
    backbone = tmp_path / 'backbone.pt'
    backbone.write_bytes(b'fixture; image model is not loaded for report')
    policy = RecoveryPolicy(2)
    with torch.no_grad():
        for p in policy.parameters():
            p.zero_()
        policy.net[-1].bias[1] = 10
    path = tmp_path / 'policy.pt'
    torch.save(dict(model=policy.state_dict(), dimensions=2, backbone_sha256=digest(backbone)), path)
    torch.save(dict(features=torch.zeros(4, 2), val=torch.arange(4)), tmp_path / 'feature_cache.pt')
    write('status.json', dict(state='complete'))
    write('protocol.json', dict(checkpoint=str(backbone), backbone_sha256=digest(backbone), modes=MODES))
    selected = dict(path=str(path), validation=summarize(records(), [1] * 4))
    write('assessment.json', dict(selected=selected, decision='Visual review required'))
    (tmp_path / 'measurements.jsonl').write_text('\n'.join(json.dumps(r) for r in records()))
    result = review(tmp_path)
    assert result['comparisons']['hard']['cvvdp_jod']['regressed'] == 1
    assert result['promoted'] is False
    assert (tmp_path / 'review.md').is_file()
    selected['validation']['clean']['policy']['pu21_db'] = 100
    write('assessment.json', dict(selected=selected, decision='Visual review required'))
    with pytest.raises(ValueError, match='does not reproduce'):
        review(tmp_path)


def test_incomplete_and_no_candidate(tmp_path):
    (tmp_path / 'status.json').write_text('{"state":"running"}')
    with pytest.raises(RuntimeError, match='requires completion'):
        review(tmp_path)
    (tmp_path / 'status.json').write_text('{"state":"complete"}')
    (tmp_path / 'assessment.json').write_text('{"selected":null,"decision":"Keep existing model"}')
    assert review(tmp_path)['selected'] is None
