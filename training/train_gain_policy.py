"""Predict measured quality gains with native SDR features; no test pixels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rudra.batch import digest,save
from rudra.quality_gain import degradation_features,gain_decisions
from training.train_conservative_policy import partition
from training.train_quality_policy import summarize,passes
from training.export_bench_pairs import degrade_like_eval
from training.sdr2hdr_dataset import load_rgb


def run(source,out):
    torch.set_num_threads(4)
    protocol=json.loads((source/'protocol.json').read_text())
    records=[json.loads(s) for s in (source/'measurements.jsonl').read_text().splitlines()]
    cache=torch.load(source/'feature_cache.pt',map_location='cpu',weights_only=True)
    if not protocol.get('joint_quality') or len(records)!=len(cache['features']): raise ValueError('Invalid source cache')
    if digest(protocol['checkpoint'])!=protocol['backbone_sha256']: raise ValueError('Backbone changed')
    rows=protocol['rows']
    expected=[(r,c) for r in rows for c in ('clean','hard')]
    if len(expected)!=len(records): raise ValueError('Incomplete source measurements')
    for record,(row,condition) in zip(records,expected):
        if row['split'] not in ('train','val') or any(record[k]!=row[k] for k in ('asset_id','scene_id','split')) or record['condition']!=condition:
            raise ValueError('Source order/split mismatch')
    pu=torch.tensor([r['pu21_db'] for r in records])
    jod=torch.tensor([r['cvvdp_jod'] for r in records])
    if not torch.allclose(pu,cache['quality']) or jod.shape!=pu.shape: raise ValueError('Quality cache mismatch')
    fit,cal,val=partition(records)
    out.mkdir(parents=True,exist_ok=False)
    state=dict(state='running',pid=os.getpid(),stage='extract_native_features',frames_done=0,frames_total=len(records))
    save(out/'status.json',state)
    save(out/'protocol.json',dict(source=str(source.resolve()),source_cache_sha256=digest(source/'feature_cache.pt'),
        measurements_sha256=digest(source/'measurements.jsonl'),backbone_sha256=protocol['backbone_sha256'],
        script_sha256=digest(__file__),features_sha256=digest(ROOT/'rudra/quality_gain.py'),
        fit_indices=fit,calibration_indices=cal,validation_indices=val,
        penalty_grid=[1.,10.,100.],margin_grid=[0.,.1,.25,.5,1.],test_pixels_used=False,promoted=False))
    try:
        extra=[]
        for ri,row in enumerate(rows):
            if digest(row['sdr_path'])!=records[2*ri]['sdr_sha256']: raise ValueError('SDR changed')
            sdr=torch.from_numpy(load_rgb(row['sdr_path'],False)).permute(2,0,1)[None]
            seed=int(hashlib.sha256(row['asset_id'].encode()).hexdigest()[:7],16)
            for condition in ('clean','hard'):
                x=sdr if condition=='clean' else degrade_like_eval(sdr[0],seed)[None]
                extra.append(degradation_features(x)[0])
                state['frames_done']+=1; save(out/'status.json',state)
        x=torch.cat((cache['features'],torch.stack(extra)),1).double()
        if not torch.isfinite(x).all(): raise ValueError('Invalid features')
        torch.save(dict(features=x,extra=torch.stack(extra)),out/'features.pt')
        mean=x[fit].mean(0); scale=x[fit].std(0,unbiased=False).clamp_min(1e-4)
        x=(x-mean)/scale; x=torch.cat((x,torch.ones(len(x),1)),1)
        scores=torch.stack((pu,jod/.05),-1).double()
        target=(scores-scores[:,3:4]).reshape(len(x),8)
        state['stage']='fit_gain_regression'; save(out/'status.json',state)
        winner=None; trials=[]
        for penalty in (1.,10.,100.):
            regularizer=torch.eye(x.shape[1],dtype=x.dtype)*penalty; regularizer[-1,-1]=0
            weights=torch.linalg.solve(x[fit].T@x[fit]+regularizer,x[fit].T@target[fit])
            prediction=(x[cal]@weights).reshape(-1,4,2)
            for margin in (0.,.1,.25,.5,1.):
                choices=gain_decisions(prediction,margin).tolist()
                report=summarize([records[i] for i in cal],choices)
                score=sum(report[c]['policy']['cvvdp_jod']-report[c]['shipped']['cvvdp_jod'] for c in ('clean','hard'))
                eligible=passes(report) and score>1e-6
                trials.append(dict(penalty=penalty,margin=margin,calibration=report,eligible=eligible))
                if eligible and (winner is None or score>winner['score']):
                    winner=dict(penalty=penalty,margin=margin,score=score,calibration=report)
                    torch.save(dict(kind='experimental_gain_regression',weights=weights,mean=mean,scale=scale,margin=margin,
                        backbone_sha256=protocol['backbone_sha256'],license='Non-commercial v5 derivative'),out/'candidate.pt')
        if winner:
            saved=torch.load(out/'candidate.pt',weights_only=True)
            choices=gain_decisions((x[val]@saved['weights']).reshape(-1,4,2),winner['margin']).tolist()
            winner['validation']=summarize([records[i] for i in val],choices)
            winner['passing']=passes(winner['validation'])
        save(out/'assessment.json',dict(candidate=winner,trials=trials,promoted=False,test_pixels_used=False,
            limitation='Exploratory reused validation; no independent or temporal confirmation. Candidate is not a Studio checkpoint.'))
        state.update(state='complete',stage='review',passing_candidates=int(bool(winner and winner['passing'])))
    except BaseException as exc:
        state.update(state='failed',error=str(exc)); raise
    finally: save(out/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.source,a.out)
