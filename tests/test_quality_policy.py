import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from rudra.recovery_policy import RecoveryPolicy, regret_loss, load_policy
from training import train_quality_policy as trainer


def test_regret_optimizes_measured_quality_not_degradation_label():
    logits=torch.zeros((2,4),requires_grad=True)
    scores=torch.tensor([[30.,40.,20.,10.],[40.,20.,30.,10.]])
    regret_loss(logits,scores).backward()
    assert logits.grad[0,1]<0 and logits.grad[1,0]<0
    assert logits.grad[0,3]>0
    assert regret_loss(torch.randn(1,4),torch.ones(1,4)).item()==0


def test_selection_is_scene_balanced_and_excludes_test():
    rows=[dict(scene_id=str(i),asset_id=str(i)+str(j),split=split)
          for i,split in enumerate(['train','train','val','test']) for j in range(3)]
    selected=trainer.select_rows(rows,10)
    assert len(selected)==3 and all(r['split']!='test' for r in selected)
    assert selected==trainer.select_rows(list(reversed(rows)),10)
    rows.append(dict(scene_id='0',asset_id='leak',split='val'))
    with pytest.raises(ValueError,match='leakage'): trainer.select_rows(rows,10)


def test_checkpoint_binding(tmp_path):
    policy=RecoveryPolicy(3)
    path=tmp_path/'policy.pt'; backbone=tmp_path/'backbone.pt'; backbone.write_bytes(b'weights')
    torch.save(dict(model=policy.state_dict(),dimensions=3,backbone_sha256='incorrect'),path)
    with pytest.raises(ValueError,match='different'): load_policy(path,backbone)


@pytest.mark.parametrize('interrupt',[False,True])
@pytest.mark.parametrize('joint',[False,True])
def test_workflow_trains_only_train_features_and_never_reads_test(tmp_path,monkeypatch,interrupt,joint):
    from rudra.sdr2hdr import SDR2HDRNet
    import training.infer_sdr2hdr
    torch.set_num_threads(2)
    model=SDR2HDRNet(base_channels=4).eval()
    checkpoint=tmp_path/'base.pt'; checkpoint.write_bytes(b'fixture')
    rows=[]
    for split in ('train','val','test'):
        source=tmp_path/(split+'.png'); source.write_bytes(b'fixture')
        rows.append(dict(asset_id=split,scene_id=split,split=split,sdr_path=str(source),hdr_path=str(source)))
    manifest=tmp_path/'manifest.jsonl'; manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    out=tmp_path/'run'; out.mkdir()
    def load(path,hdr,**kwargs):
        assert 'test.png' not in str(path)
        return np.full((64,64,3),.03 if hdr else .25,dtype=np.float32)
    monkeypatch.setattr(trainer,'load_rgb',load)
    monkeypatch.setattr(trainer,'load_models',lambda *a:(model,None))
    monkeypatch.setattr(trainer,'colorvideovdp_available',lambda:True)
    monkeypatch.setattr(trainer,'hdr_vdp3_jod',lambda *a,**k:(9.,'colorvideovdp'))
    state={}
    args=SimpleNamespace(manifest=manifest,checkpoint=checkpoint,out=out,device='cpu',train_scenes=1,epochs=2,joint_quality=joint)
    if interrupt:
        original=trainer.predict_image
        calls=[]
        def stopped(*a,**kw):
            calls.append(1)
            if len(calls)==5: raise RuntimeError('Simulated interruption')
            return original(*a,**kw)
        monkeypatch.setattr(trainer,'predict_image',stopped)
        with pytest.raises(RuntimeError,match='Simulated interruption'):
            trainer.run(args,state)
        prefix=(out/'measurements.jsonl').read_bytes()
        assert len(prefix.splitlines())==1
        selected=trainer.select_rows(rows,1)
        measured=tmp_path/'train.png'
        measured.write_bytes(b'changed')
        with pytest.raises(ValueError,match='source image changed'):
            trainer.resume_measurements(args,selected,trainer.digest(checkpoint))
        measured.write_bytes(b'fixture')
        args.resume=True
    trainer.run(args,state)
    if interrupt:
        assert len(calls)==17  # 16 successful modes, one interrupted call; no repeated saved modes.
        assert (out/'measurements.jsonl').read_bytes().startswith(prefix)
        with pytest.raises(ValueError,match='policy fitting already started'):
            trainer.resume_measurements(args,selected,trainer.digest(checkpoint))
    assert state['state']=='complete'
    measurements=[json.loads(x) for x in (out/'measurements.jsonl').read_text().splitlines()]
    assert len(measurements)==4
    assert all(len(r['cvvdp_jod'])==(4 if joint or r['split']=='val' else 0) for r in measurements)
    assert all(r['split'] in ('train','val') for r in measurements)
    assert json.loads((out/'assessment.json').read_text())['promoted'] is False
