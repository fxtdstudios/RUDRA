"""Measure recovery benefit on train/val, fit an SDR-only policy, and gate selection.

Never reads test image pixels and never promotes weights. Native frames are
measured under the same tiled inference used for delivery. Validation measures
real image ColorVideoVDP; no temporal-quality claim is supported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from rudra.batch import digest, save
from rudra.recovery_policy import MODES, RecoveryPolicy, features, regret_loss
from rudra.delivery.bench import pu_psnr
from rudra.hdrvdp import hdr_vdp3_jod, colorvideovdp_available
from training.infer_sdr2hdr import load_models, predict_image
from training.export_bench_pairs import degrade_like_eval
from training.sdr2hdr_dataset import load_rgb
from training.quality_benchmark import capture_environment
from training.quality_objective import joint_utility, balanced_loss


def select_rows(rows, train_limit):
    scenes={}
    for row in rows:
        scene=row['scene_id']; split=row['split']
        if scene in scenes and scenes[scene]!=split:
            raise ValueError('Scene leakage across dataset splits')
        scenes[scene]=split
    chosen={}
    for row in rows:
        if row['split'] not in ('train','val'): continue
        key=(row['split'],row['scene_id'])
        rank=lambda r: hashlib.sha256(('quality-policy-v1:'+r['asset_id']).encode()).hexdigest()
        if key not in chosen or rank(row)<rank(chosen[key]): chosen[key]=row
    key=lambda r: hashlib.sha256(r['scene_id'].encode()).hexdigest()
    train=sorted([r for r in chosen.values() if r['split']=='train'],key=key)[:train_limit]
    val=sorted([r for r in chosen.values() if r['split']=='val'],key=key)
    if not train or not val: raise ValueError('Both training and validation scenes required')
    return train+val


def summarize(records, selections):
    report={}
    for condition in ('clean','hard'):
        indices=[i for i,r in enumerate(records) if r['condition']==condition]
        report[condition]={}
        for name,choices in [('policy',selections),('shipped',[3]*len(records))]:
            report[condition][name]={metric:float(np.mean([records[i][metric][choices[i]] for i in indices]))
                                     for metric in ('pu21_db','cvvdp_jod')}
        report[condition]['frames']=len(indices)
    return report


def passes(report):
    return all(report[c]['policy'][m]>=report[c]['shipped'][m]
               for c in ('clean','hard') for m in ('pu21_db','cvvdp_jod'))


def resume_measurements(a, selected, backbone_hash):
    """Validate an interrupted measurement prefix before reusing any scores."""
    if (a.out/'learning.jsonl').exists() or (a.out/'assessment.json').exists():
        raise ValueError('Resume supports interrupted measurement only; policy fitting already started')
    protocol=json.loads((a.out/'protocol.json').read_text())
    joint=getattr(a,'joint_quality',False)
    if protocol.get('joint_quality',False)!=joint:
        raise ValueError('Resume objective differs from the original experiment')
    if joint and protocol.get('objective_sha256')!=digest(ROOT/'training/quality_objective.py'):
        raise ValueError('Resume objective implementation changed')
    if (protocol['backbone_sha256']!=backbone_hash or
        protocol['manifest_sha256']!=digest(a.manifest) or
        protocol['policy_sha256']!=digest(ROOT/'rudra/recovery_policy.py') or
        tuple(protocol['modes'])!=MODES or protocol['rows']!=selected):
        raise ValueError('Resume inputs or recovery policy differ from the original experiment')
    records=[json.loads(line) for line in (a.out/'measurements.jsonl').read_text().splitlines() if line.strip()]
    expected=[(row,condition) for row in selected for condition in ('clean','hard')]
    if len(records)>len(expected): raise ValueError('Too many saved measurements')
    for record,(row,condition) in zip(records,expected):
        for key in ('asset_id','scene_id','split'):
            if record[key]!=row[key]: raise ValueError('Saved measurement order mismatch')
        if record['condition']!=condition: raise ValueError('Saved measurement condition mismatch')
        for key in ('sdr','hdr'):
            if record[key+'_sha256']!=digest(row[key+'_path']):
                raise ValueError('Previously measured source image changed')
        for metric,length in (('pu21_db',4),('cvvdp_jod',4 if joint or row['split']=='val' else 0)):
            if len(record[metric])!=length or not np.isfinite(record[metric]).all():
                raise ValueError('Invalid saved quality scores')
    return records


def run(a, state):
    torch.set_num_threads(4)
    if not colorvideovdp_available(): raise RuntimeError('Real ColorVideoVDP required')
    torch.manual_seed(20260925)
    joint=getattr(a,'joint_quality',False)
    rows=[json.loads(x) for x in a.manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
    selected=select_rows(rows,a.train_scenes)
    backbone_hash=digest(a.checkpoint)
    resuming=getattr(a,'resume',False)
    completed=resume_measurements(a,selected,backbone_hash) if resuming else []
    if resuming:
        save(a.out/'resume.json',dict(resumed_at=time.time(),reused_measurements=len(completed),
                                    script_sha256=digest(__file__),feature_rebuild=True))
    else:
        capture_environment(a.out)
        save(a.out/'protocol.json',dict(checkpoint=str(a.checkpoint.resolve()),backbone_sha256=backbone_hash,
        manifest_sha256=digest(a.manifest),script_sha256=digest(__file__),policy_sha256=digest(ROOT/'rudra/recovery_policy.py'),
        modes=MODES,rows=selected,train_scenes=sum(r['split']=='train' for r in selected),
        joint_quality=joint,objective_sha256=digest(ROOT/'training/quality_objective.py'),
        val_scenes=sum(r['split']=='val' for r in selected),test_pixels_used=False,
        criterion='No validation mean regression in clean/hard PU21 and real image CVVDP',
        license='Non-commercial v5 derivative',promoted=False,
        note='Shadow-on modes retain the pretrained shadow gate. Native 512/64 tiled inference; one frame per scene.'))
    model=load_models(str(a.checkpoint),None,torch.device(a.device))[0].eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    records=[]; xs=[]
    state.update(stage='measure_recovery_benefit',frames_total=len(selected)*2,frames_done=len(completed))
    save(a.out/'status.json',state)
    with torch.inference_mode(), (a.out/'measurements.jsonl').open('a' if resuming else 'x',encoding='utf-8') as stream:
        for row in selected:
            if row['split']=='test': raise ValueError('Test images must never enter training')
            source=torch.from_numpy(load_rgb(row['sdr_path'],False)).permute(2,0,1)[None].to(a.device)
            reference=load_rgb(row['hdr_path'],True,ceiling=1000)*10000
            if tuple(source.shape[-2:])!=reference.shape[:2]: raise ValueError('Geometry mismatch')
            target=torch.from_numpy(reference/10000).permute(2,0,1)[None].to(a.device)
            seed=int(hashlib.sha256(row['asset_id'].encode()).hexdigest()[:7],16)
            for condition in ('clean','hard'):
                x=source if condition=='clean' else degrade_like_eval(source[0].cpu(),seed)[None].to(a.device)
                feat=features(model,x).cpu()[0]
                if not torch.isfinite(feat).all(): raise ValueError('Non-finite features')
                if len(records)<len(completed):
                    records.append(completed[len(records)])
                    xs.append(feat.clone())
                    continue
                record=dict(asset_id=row['asset_id'],scene_id=row['scene_id'],split=row['split'],condition=condition,
                            sdr_sha256=digest(row['sdr_path']),hdr_sha256=digest(row['hdr_path']),pu21_db=[],cvvdp_jod=[])
                for mode in MODES:
                    pred=predict_image(model,x,True,512,64,mode)
                    nits=pred[0].permute(1,2,0).float().cpu().numpy()*10000
                    if not np.isfinite(nits).all() or (nits<0).any(): raise ValueError('Invalid inference')
                    pu=pu_psnr(nits,reference)
                    if not np.isfinite(pu): raise ValueError('Non-finite PU21 score')
                    record['pu21_db'].append(float(pu))
                    if joint or row['split']=='val':
                        jod,backend=hdr_vdp3_jod(pred,target,color_space='rec2020',diffuse_white_nits=10000)
                        if backend!='colorvideovdp' or not np.isfinite(jod): raise RuntimeError('Real CVVDP failed')
                        record['cvvdp_jod'].append(float(jod))
                xs.append(feat.clone()); records.append(record)
                stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
                state['frames_done']+=1; save(a.out/'status.json',state)
                print(f"Measured {state['frames_done']}/{state['frames_total']} {row['split']} {condition}",flush=True)
    # Clone outside inference mode for autograd; features and target quality are fixed.
    x=torch.tensor(np.stack([f.numpy() for f in xs]),dtype=torch.float32)
    quality=torch.tensor([r['pu21_db'] for r in records],dtype=torch.float32)
    train=torch.tensor([i for i,r in enumerate(records) if r['split']=='train'])
    val=torch.tensor([i for i,r in enumerate(records) if r['split']=='val'])
    torch.save(dict(features=x,quality=quality,train=train,val=val),a.out/'feature_cache.pt')
    if joint:
        jod=torch.tensor([r['cvvdp_jod'] for r in records],dtype=torch.float32)
        conditions=torch.tensor([r['condition']=='hard' for r in records])
        # Only training labels enter the objective and its normalization.
        utility=joint_utility(quality[train],jod[train],conditions[train])
    validation=[records[i] for i in val.tolist()]
    candidates=[]
    state.update(stage='train_policy',epochs_total=a.epochs*3,epochs_done=0)
    with (a.out/'learning.jsonl').open('x',encoding='utf-8') as log:
        for seed in (20260925,2,3):
            torch.manual_seed(seed)
            policy=RecoveryPolicy(x.shape[1])
            policy.mean.copy_(x[train].mean(0)); policy.scale.copy_(x[train].std(0,unbiased=False).clamp_min(1e-4))
            optimizer=torch.optim.AdamW(policy.parameters(),lr=.001,weight_decay=.01)
            for epoch in range(a.epochs):
                policy.train()
                permutation=train if joint else train[torch.randperm(len(train))]
                for batch in permutation.split(len(train) if joint else 32):
                    optimizer.zero_grad()
                    loss=(balanced_loss(policy(x[batch]),utility,conditions[train]) if joint
                          else regret_loss(policy(x[batch]),quality[batch]))
                    loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(),1); optimizer.step()
                policy.eval()
                with torch.no_grad(): choices=policy(x[val]).argmax(1).tolist()
                report=summarize(validation,choices)
                entry=dict(seed=seed,epoch=epoch+1,validation=report,passing=passes(report))
                log.write(json.dumps(entry)+'\n'); log.flush()
                state['epochs_done']+=1; save(a.out/'status.json',state)
                if (epoch+1)%10==0 or epoch+1==a.epochs:
                    path=a.out/f'policy_seed{seed}_epoch{epoch+1}.pt'
                    torch.save(dict(model=policy.state_dict(),dimensions=x.shape[1],backbone_sha256=backbone_hash,
                                    modes=MODES,license='Non-commercial v5 derivative',experimental=True),path)
                    candidates.append(dict(entry,path=str(path)))
    eligible=[c for c in candidates if c['passing']]
    selected_policy=max(eligible,key=lambda c:sum(c['validation'][k]['policy']['cvvdp_jod'] for k in ('clean','hard'))) if eligible else None
    save(a.out/'assessment.json',dict(candidates=candidates,selected=selected_policy,promoted=False,test_pixels_used=False,
        decision='Candidate requires independent visual and temporal review' if eligible else 'No policy passed; keep existing model',
        limitation='Exploratory validation selection across seeds/epochs; no fresh held-out or temporal confirmation.'))
    state.update(state='complete',stage='review_assessment',passing_candidates=len(eligible))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-scenes',type=int,default=256)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--device',default='cuda')
    p.add_argument('--joint-quality',action='store_true',help='Measure training CVVDP and optimize both metrics with condition balancing')
    p.add_argument('--resume',action='store_true',help='Reuse a stopped measurement pass; does not resume policy epochs')
    a=p.parse_args()
    if a.train_scenes<1 or a.epochs<1: raise ValueError('Positive scene and epoch counts required')
    if a.resume:
        if not a.out.is_dir(): raise ValueError('Resume directory does not exist')
        # Preflight before overwriting status or opening measurement output.
        rows=[json.loads(x) for x in a.manifest.read_text().splitlines() if x.strip()]
        resume_measurements(a,select_rows(rows,a.train_scenes),digest(a.checkpoint))
    else:
        a.out.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),state='running',started_at=time.time(),test_pixels_used=False)
    save(a.out/'status.json',state)
    try: run(a,state)
    except BaseException as exc:
        state.update(state='failed',error=str(exc)); raise
    finally: save(a.out/'status.json',state)


if __name__=='__main__': main()
