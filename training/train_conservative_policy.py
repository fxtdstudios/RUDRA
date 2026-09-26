"""Cached CPU experiment; select on reserved TRAIN scenes, evaluate validation once."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import os

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from rudra.batch import digest, save
from rudra.recovery_policy import RecoveryPolicy, MODES
from training.quality_objective import joint_utility
from training.train_quality_policy import summarize, passes


def partition(records):
    scenes=sorted({r['scene_id'] for r in records if r['split']=='train'},
                  key=lambda s:hashlib.sha256(('conservative-v1:'+s).encode()).hexdigest())
    if len(scenes)<5: raise ValueError('Need at least five training scenes')
    calibration=set(scenes[:max(1,len(scenes)//5)])
    fit=[i for i,r in enumerate(records) if r['split']=='train' and r['scene_id'] not in calibration]
    cal=[i for i,r in enumerate(records) if r['split']=='train' and r['scene_id'] in calibration]
    val=[i for i,r in enumerate(records) if r['split']=='val']
    if {records[i]['scene_id'] for i in fit+cal} & {records[i]['scene_id'] for i in val}:
        raise ValueError('Scene leakage')
    return fit,cal,val


def run(source,out):
    torch.set_num_threads(4)
    protocol=json.loads((source/'protocol.json').read_text())
    records=[json.loads(line) for line in (source/'measurements.jsonl').read_text().splitlines()]
    cache=torch.load(source/'feature_cache.pt',weights_only=True,map_location='cpu')
    if not protocol.get('joint_quality') or tuple(protocol['modes'])!=MODES:
        raise ValueError('Requires a completed joint-quality cache')
    if json.loads((source/'status.json').read_text())['state']!='complete':
        raise ValueError('Source run is incomplete')
    if digest(protocol['checkpoint'])!=protocol['backbone_sha256']:
        raise ValueError('Backbone changed')
    if any(r['split'] not in ('train','val') or len(r['cvvdp_jod'])!=4 for r in records):
        raise ValueError('Invalid split or missing perceptual scores')
    pu=torch.tensor([r['pu21_db'] for r in records])
    jod=torch.tensor([r['cvvdp_jod'] for r in records])
    x=cache['features']
    if len(x)!=len(records) or not torch.allclose(pu,cache['quality']):
        raise ValueError('Cache alignment mismatch')
    if not all(torch.isfinite(t).all() for t in (x,pu,jod)):
        raise ValueError('Non-finite cache')
    fit,cal,val=partition(records)
    condition=torch.tensor([r['condition']=='hard' for r in records])
    utility=joint_utility(pu[fit],jod[fit],condition[fit])
    best,labels=utility.max(1)
    labels=torch.where(best>.01,labels,3)  # marginal/tied cases retain shipped behavior
    out.mkdir(parents=True,exist_ok=False)
    state=dict(state='running',stage='fit_cached_policy',pid=os.getpid(),epochs_done=0,epochs_total=900)
    save(out/'status.json',state)
    save(out/'protocol.json',dict(source=str(source.resolve()),checkpoint=protocol['checkpoint'],
        backbone_sha256=protocol['backbone_sha256'],cache_sha256=digest(source/'feature_cache.pt'),
        measurements_sha256=digest(source/'measurements.jsonl'),script_sha256=digest(__file__),
        fit_indices=fit,calibration_indices=cal,validation_indices=val,
        criterion='Select only on reserved training scenes; evaluate frozen selection on reused validation once',
        test_pixels_used=False,promoted=False))
    winner=None; calibration_records=[records[i] for i in cal]
    try:
        with (out/'calibration.jsonl').open('x') as log:
            for seed in (20260925,2,3):
                torch.manual_seed(seed)
                policy=RecoveryPolicy(x.shape[1])
                policy.mean.copy_(x[fit].mean(0)); policy.scale.copy_(x[fit].std(0,unbiased=False).clamp_min(1e-4))
                optimizer=torch.optim.AdamW(policy.parameters(),lr=.001,weight_decay=.01)
                for epoch in range(1,301):
                    policy.train(); optimizer.zero_grad()
                    losses=F.cross_entropy(policy(x[fit]),labels,reduction='none')
                    groups=torch.stack([losses[condition[fit]==c].mean() for c in (False,True)])
                    loss=groups.max()+.1*groups.mean()
                    loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),1); optimizer.step()
                    if epoch%50==0:
                        policy.eval()
                        for threshold in (.5,.7,.9,.95,.99):
                            policy.confidence_threshold=threshold
                            choices=policy.decisions(x[cal]).tolist()
                            report=summarize(calibration_records,choices)
                            score=sum(report[c]['policy']['cvvdp_jod']-report[c]['shipped']['cvvdp_jod'] for c in ('clean','hard'))
                            eligible=passes(report) and score>1e-6
                            log.write(json.dumps(dict(seed=seed,epoch=epoch,threshold=threshold,report=report,eligible=eligible))+'\n'); log.flush()
                            if eligible and (winner is None or score>winner['score']):
                                winner=dict(seed=seed,epoch=epoch,threshold=threshold,score=score,calibration=report)
                                torch.save(dict(model=policy.state_dict(),dimensions=x.shape[1],confidence_threshold=threshold,
                                    backbone_sha256=protocol['backbone_sha256'],modes=MODES,experimental=True,
                                    license='Non-commercial v5 derivative'),out/'candidate.pt')
                        state['epochs_done']+=50; save(out/'status.json',state)
                        print(f"Training {state['epochs_done']}/900",flush=True)
        if winner:
            from rudra.recovery_policy import load_policy
            policy=load_policy(out/'candidate.pt',protocol['checkpoint'])
            choices=policy.decisions(x[val]).tolist()
            winner['validation']=summarize([records[i] for i in val],choices)
            winner['passing']=passes(winner['validation'])
            winner['validation_mode_counts']={m:choices.count(i) for i,m in enumerate(MODES)}
        save(out/'assessment.json',dict(candidate=winner,promoted=False,test_pixels_used=False,
            limitation='Validation has been used in earlier experiments; fresh independent and temporal confirmation still required.'))
        state.update(state='complete',stage='review',passing_candidates=int(bool(winner and winner['passing'])))
    except BaseException as exc:
        state.update(state='failed',error=str(exc)); raise
    finally:
        save(out/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.source,a.out)
