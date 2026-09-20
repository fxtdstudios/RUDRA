import json

import pytest

from rudra.batch import digest
from training.recovery_ablation import compare, validate_frozen


def test_comparison_pairs_modes_with_correct_direction():
    records=[dict(asset_id=str(i),condition='clean',method=m,score=i+d)
             for i in range(3) for m,d in [('shipped',0),('highlights',2),('baseline',100)]]
    result=compare(records,'highlights','shipped','clean','score')
    assert result['delta']==2
    assert result['pairs']==3
    assert result['wins']==3


@pytest.fixture
def frozen(tmp_path):
    for name in ('checkpoint','sdr','hdr'):
        (tmp_path/name).write_bytes(name.encode())
    row=dict(asset_id='a',split='val')
    for name in ('sdr','hdr'):
        row[name+'_path']=str(tmp_path/name)
        row[name+'_sha256']=digest(tmp_path/name)
    protocol=dict(split='val',test_set_used=False,rows=[row],
        checkpoint=str(tmp_path/'checkpoint'),checkpoint_sha256=digest(tmp_path/'checkpoint'))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'environment.json').write_text(json.dumps(dict(code_sha256={})))
    records=[dict(asset_id='a',condition=c,method=m) for c in ('clean','hard') for m in ('baseline','shipped')]
    (tmp_path/'scores.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    return tmp_path


def test_frozen_sources_are_verified(frozen):
    assert len(validate_frozen(frozen)[1])==4
    (frozen/'sdr').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'): validate_frozen(frozen)


def test_missing_results_rejected(frozen):
    (frozen/'scores.jsonl').write_text('')
    with pytest.raises(ValueError,match='incomplete'): validate_frozen(frozen)


def test_test_protocol_rejected(frozen):
    path=frozen/'protocol.json'
    protocol=json.loads(path.read_text())
    protocol['test_set_used']=True
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='validation-only'): validate_frozen(frozen)
