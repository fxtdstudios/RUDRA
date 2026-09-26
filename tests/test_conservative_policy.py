import torch
import pytest
from rudra.recovery_policy import RecoveryPolicy, load_policy
from rudra.batch import digest
from training.train_conservative_policy import partition


def test_scene_partition_reserves_both_conditions_and_excludes_validation():
    rows=[dict(scene_id=str(i),split='train' if i<10 else 'val',condition=c)
          for i in range(12) for c in ('clean','hard')]
    fit,cal,val=partition(rows)
    assert len(fit)==16 and len(cal)==4 and len(val)==4
    assert set(fit).isdisjoint(cal)
    assert all(rows[i]['split']=='train' for i in fit+cal)
    assert {rows[i]['scene_id'] for i in fit}.isdisjoint({rows[i]['scene_id'] for i in cal})
    rows[-1]['scene_id']='0'
    with pytest.raises(ValueError,match='leakage'): partition(rows)


def test_confidence_fallback_survives_checkpoint_roundtrip(tmp_path):
    model=RecoveryPolicy(2)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.net[-1].bias[1]=1
    assert model.decisions(torch.zeros(1,2)).item()==1
    model.confidence_threshold=.9
    assert model.decisions(torch.zeros(1,2)).item()==3
    backbone=tmp_path/'backbone.pt'; backbone.write_bytes(b'fixture')
    path=tmp_path/'policy.pt'
    torch.save(dict(model=model.state_dict(),dimensions=2,backbone_sha256=digest(backbone),confidence_threshold=.9),path)
    loaded=load_policy(path,backbone)
    assert loaded.decisions(torch.zeros(1,2)).item()==3
