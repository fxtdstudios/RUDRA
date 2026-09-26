"""Fixed 24-proposal training-only feasibility test; no validation or promotion."""
import hashlib
import json
from pathlib import Path
import torch
from rudra.batch import digest, save
from training.guarded_update import guarded_step, preserves
from training.robust_reconstruction import reconstruction_risk, region_pu_risk, region_paired_objective
from training.sdr2hdr_dataset import SDRHDRDataset, read_jsonl
from training.train_robust_reconstruction import check_splits
from training.train_sdr2hdr import load_image_checkpoint, seed_everything, _atomic_save


def components(model,batch):
    result=[]
    for x in (batch['clean_sdr'],batch['sdr']):
        pred=model(x,preserve_outside=True).hdr
        risk,present=region_pu_risk(pred,batch['hdr'],batch['ceiling'])
        result.append(torch.cat([reconstruction_risk(pred,batch['hdr'],batch['ceiling']).reshape(-1),risk[present]]))
    return torch.cat(result)


def main():
    seed_everything(20260928); torch.set_num_threads(4)
    out=Path('outputs/guard_feasibility_20260927'); out.mkdir(exist_ok=False)
    manifest=Path('outputs/finetune_views_20260920/data/manifest.jsonl')
    checkpoint=Path('checkpoints/sdr2hdr_shadow_v1.pt')
    check_splits(read_jsonl(manifest))
    data=SDRHDRDataset(manifest,split='train',crop_size=256,augment=False,deterministic_degradation=True)
    scenes={}
    for i,r in enumerate(data.records): scenes.setdefault(r['scene_id'],i)
    indices=[scenes[s] for s in sorted(scenes,key=lambda s:hashlib.sha256(('guard:'+s).encode()).hexdigest())[:32]]
    proposals,diagnostics=indices[:24],indices[24:]
    device=torch.device('cuda')
    model,payload=load_image_checkpoint(checkpoint,device); model.eval()
    teacher,_=load_image_checkpoint(checkpoint,device); teacher.eval().requires_grad_(False)
    for name,p in model.named_parameters(): p.requires_grad_(not name.startswith(('gate.','shadow_gate.')))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-6,weight_decay=.01)
    save(out/'protocol.json',dict(checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest),
         scripts={p:digest(Path('training')/p) for p in ['check_guard_feasibility.py','guarded_update.py','robust_reconstruction.py']},
         proposal_assets=[data.records[i]['asset_id'] for i in proposals],
         diagnostic_assets=[data.records[i]['asset_id'] for i in diagnostics],
         proposals=24,lr=2e-6,tolerance=1e-7,split='train',promoted=False,
         criterion='All clean/hard log and active-region PU components <= pre-update and teacher + tolerance',
         limitation='Training-crop surrogate feasibility only; eight separate training scenes are not independent evaluation.'))
    def batch(index):
        item=data[index]
        return {k:item[k][None].to(device) for k in ('clean_sdr','sdr','hdr','ceiling')}
    records=[]
    for index in proposals:
        b=batch(index)
        with torch.no_grad():
            before=components(model,b); reference=components(teacher,b)
            teachers=[teacher(b[k],preserve_outside=True).hdr for k in ('clean_sdr','sdr')]
        pred=[model(b[k],preserve_outside=True).hdr for k in ('clean_sdr','sdr')]
        loss,_=region_paired_objective(pred,teachers,b['hdr'],b['ceiling'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        measured={}
        def accept():
            with torch.no_grad(): after=components(model,b)
            measured['max_increase']=float((after-before).max())
            return preserves(after,before,reference)
        accepted=guarded_step(model,opt,accept)
        records.append(dict(asset_id=data.records[index]['asset_id'],accepted=accepted,**measured))
        save(out/'status.json',dict(state='running',done=len(records),total=24,accepted=sum(r['accepted'] for r in records)))
    checks=[]
    with torch.no_grad():
        for index in diagnostics:
            b=batch(index); before=components(teacher,b); after=components(model,b)
            checks.append(dict(asset_id=data.records[index]['asset_id'],preserved=preserves(after,before,before),
                               max_increase=float((after-before).max())))
    accepted=sum(r['accepted'] for r in records)
    assessment=dict(accepted=accepted,rejected=24-accepted,diagnostic_preserved=sum(r['preserved'] for r in checks),
         diagnostic_total=8,feasible=accepted>0 and all(r['preserved'] for r in checks),
         promoted=False,proposals=records,diagnostics=checks,limitation='Training surrogates, not release quality.')
    _atomic_save(dict(model=model.state_dict(),config=payload.get('config',{}),step=accepted,
                     experimental=True,license='Non-commercial v5 derivative'),out/'candidate.pt')
    save(out/'assessment.json',assessment); save(out/'status.json',dict(state='complete',done=24,total=24,accepted=accepted))
    print(json.dumps({k:v for k,v in assessment.items() if k not in ('proposals','diagnostics')},indent=2))


if __name__=='__main__': main()
