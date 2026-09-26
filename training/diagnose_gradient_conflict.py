"""Training-only gradient audit. No updates, checkpoint selection or promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from rudra.batch import digest, save
from training.sdr2hdr_dataset import SDRHDRDataset, read_jsonl
from training.train_sdr2hdr import load_image_checkpoint, seed_everything
from training.train_robust_reconstruction import check_splits
from training.robust_reconstruction import reconstruction_risk, region_pu_risk, region_paired_objective


def geometry(a, b):
    dot=sum((x.double()*y.double()).sum() for x,y in zip(a,b))
    aa=sum(x.double().square().sum() for x in a)
    bb=sum(x.double().square().sum() for x in b)
    return dict(dot=float(dot), cosine=float(dot/(aa*bb).sqrt().clamp_min(1e-30)),
                first_norm=float(aa.sqrt()), second_norm=float(bb.sqrt()))


def condition_risk(pred, target, ceiling):
    regions,present=region_pu_risk(pred,target,ceiling)
    return reconstruction_risk(pred,target,ceiling).mean()+(regions*present).sum()/present.sum().clamp_min(1)


def run(a):
    if a.scenes<1: raise ValueError('Positive scene count required')
    rows=read_jsonl(a.manifest); check_splits(rows)
    seed_everything(20260927); torch.set_num_threads(4)
    data=SDRHDRDataset(a.manifest,split='train',crop_size=256,augment=False,deterministic_degradation=True)
    # One deterministic record per scene; ranking is independent of image scores.
    chosen={}
    for i,r in enumerate(data.records):
        chosen.setdefault(r['scene_id'],i)
    indices=[chosen[s] for s in sorted(chosen,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:a.scenes]]
    a.out.mkdir(parents=True,exist_ok=False)
    save(a.out/'protocol.json',dict(manifest_sha256=digest(a.manifest),scenes=len(indices),
         assets=[data.records[i]['asset_id'] for i in indices],split='train',crop_size=256,
         script_sha256=digest(__file__),objective_sha256=digest(Path(__file__).with_name('robust_reconstruction.py')),
         limitation='Local first-order gradients on training crops; not optimizer trajectory or independent quality.',
         checkpoints=[dict(path=str(p),sha256=digest(p)) for p in a.checkpoint],updates=0))
    teacher,_=load_image_checkpoint(a.checkpoint[0],torch.device(a.device))
    teacher.eval().requires_grad_(False)
    results=[]
    for checkpoint in a.checkpoint:
        model,_=load_image_checkpoint(checkpoint,torch.device(a.device)); model.eval()
        for name,p in model.named_parameters():
            p.requires_grad_(not name.startswith(('gate.','shadow_gate.')))
        parameters=[p for p in model.parameters() if p.requires_grad]
        def gradient(loss):
            grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            return [g.detach() if g is not None else torch.zeros_like(p) for p,g in zip(parameters,grads)]
        for index in indices:
            batch=data[index]
            clean,hard,target,ceiling=[batch[k][None].to(a.device) for k in ('clean_sdr','sdr','hdr','ceiling')]
            with torch.no_grad(): teachers=[teacher(x,preserve_outside=True).hdr for x in (clean,hard)]
            pred=[model(x,preserve_outside=True).hdr for x in (clean,hard)]
            risks=[condition_risk(p,target,ceiling) for p in pred]
            gc,gh=[gradient(r) for r in risks]
            total,parts=region_paired_objective(pred,teachers,target,ceiling)
            gt=gradient(total)
            result=dict(checkpoint=str(checkpoint),asset_id=batch['asset_id'],
                        clean_risk=float(risks[0].detach()),hard_risk=float(risks[1].detach()),
                        clean_vs_hard=geometry(gc,gh),clean_vs_update=geometry(gc,gt),
                        hard_vs_update=geometry(gh,gt),hard_region_dominates=bool(parts['region_hard']>parts['region_clean']))
            results.append(result)
            save(a.out/'status.json',dict(state='running',done=len(results),total=len(indices)*len(a.checkpoint)))
            del pred,risks,total,parts,gc,gh,gt
        del model
    summary={str(p):dict(samples=len(group:=[r for r in results if r['checkpoint']==str(p)]),
               conflicting=sum(r['clean_vs_hard']['dot']<0 for r in group),
               update_opposes_clean=sum(r['clean_vs_update']['dot']<0 for r in group),
               hard_region_dominates=sum(r['hard_region_dominates'] for r in group),
               mean_cosine=sum(r['clean_vs_hard']['cosine'] for r in group)/len(group)) for p in a.checkpoint}
    save(a.out/'results.json',dict(summary=summary,measurements=results))
    save(a.out/'status.json',dict(state='complete',done=len(results),total=len(results)))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,action='append',required=True,help='Shipped teacher first')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scenes',type=int,default=24)
    p.add_argument('--device',default='cuda')
    run(p.parse_args())
