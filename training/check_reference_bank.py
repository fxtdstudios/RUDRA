"""Fixed training-only reference-bank feasibility; never promotes a model."""
import hashlib
import json
from pathlib import Path
import torch
from rudra.batch import digest, save
from training.check_guard_feasibility import components
from training.guarded_update import guarded_step, preserves
from training.robust_reconstruction import region_paired_objective
from training.sdr2hdr_dataset import SDRHDRDataset, read_jsonl
from training.train_robust_reconstruction import check_splits
from training.train_sdr2hdr import load_image_checkpoint, seed_everything, _atomic_save


def select_groups(records, excluded):
    scenes={}
    for i,r in enumerate(records):
        if r['split']=='train' and r['scene_id'] not in excluded:
            scenes.setdefault(r['scene_id'],i)
    ordered=sorted(scenes,key=lambda s:hashlib.sha256(('bank-v1:'+s).encode()).hexdigest())
    if len(ordered)<44: raise ValueError('Need 44 eligible distinct training scenes')
    indices=[scenes[s] for s in ordered[:44]]
    return indices[:24],indices[24:36],indices[36:]


def main():
    seed_everything(20260929); torch.set_num_threads(4)
    out=Path('outputs/reference_bank_20260927'); out.mkdir(exist_ok=False)
    manifest=Path('outputs/finetune_views_20260920/data/manifest.jsonl')
    checkpoint=Path('checkpoints/sdr2hdr_shadow_v1.pt')
    rows=read_jsonl(manifest); check_splits(rows)
    prior=json.loads(Path('outputs/guard_feasibility_20260927/protocol.json').read_text())
    old_assets=set(prior['proposal_assets']+prior['diagnostic_assets'])
    old_assets.update(json.loads(Path('outputs/gradient_conflict_20260927/protocol.json').read_text())['assets'])
    excluded={r['scene_id'] for r in rows if r['asset_id'] in old_assets}
    data=SDRHDRDataset(manifest,split='train',crop_size=256,augment=False,deterministic_degradation=True)
    proposals,bank,diagnostics=select_groups(data.records,excluded)
    device=torch.device('cuda')
    model,payload=load_image_checkpoint(checkpoint,device); model.eval()
    teacher,_=load_image_checkpoint(checkpoint,device); teacher.eval().requires_grad_(False)
    for name,p in model.named_parameters(): p.requires_grad_(not name.startswith(('gate.','shadow_gate.')))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-6,weight_decay=.01)
    protocol=dict(checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest),
        scripts={p:digest(Path('training')/p) for p in ['check_reference_bank.py','check_guard_feasibility.py','guarded_update.py','robust_reconstruction.py']},
        groups={name:[data.records[i] for i in group] for name,group in [('proposals',proposals),('bank',bank),('diagnostics',diagnostics)]},
        excluded_prior_scenes=sorted(excluded),lr=2e-6,tolerance=1e-7,
        criterion='Current batch and all 12 bank scenes: every clean/hard log and active PU region <= preceding state and shipped teacher + tolerance',
        limitation='Training-only crop surrogates. Bank is deterministic scene sampling, not certified semantic coverage. No independent evaluation.',promoted=False)
    save(out/'protocol.json',protocol)
    def batch(index):
        item=data[index]
        return {k:item[k][None].to(device) for k in ('clean_sdr','sdr','hdr','ceiling')}
    bank_batches=[batch(i) for i in bank]
    with torch.no_grad(): reference_bank=[components(teacher,b) for b in bank_batches]
    records=[]
    for index in proposals:
        b=batch(index)
        with torch.no_grad():
            before=components(model,b); reference=components(teacher,b)
            bank_before=[components(model,bk) for bk in bank_batches]
            teachers=[teacher(b[k],preserve_outside=True).hdr for k in ('clean_sdr','sdr')]
        pred=[model(b[k],preserve_outside=True).hdr for k in ('clean_sdr','sdr')]
        loss,_=region_paired_objective(pred,teachers,b['hdr'],b['ceiling'])
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        observed={}
        def accept():
            with torch.no_grad():
                after=components(model,b)
                observed['batch_preserved']=preserves(after,before,reference)
                bank_after=[components(model,bk) for bk in bank_batches]
                observed['bank_preserved']=sum(preserves(a,p,r) for a,p,r in zip(bank_after,bank_before,reference_bank))
            return observed['batch_preserved'] and observed['bank_preserved']==len(bank)
        accepted=guarded_step(model,opt,accept)
        records.append(dict(asset_id=data.records[index]['asset_id'],accepted=accepted,**observed))
        save(out/'status.json',dict(state='running',done=len(records),total=24,accepted=sum(r['accepted'] for r in records)))
    checks=[]
    with torch.no_grad():
        for index in diagnostics:
            b=batch(index); before=components(teacher,b); after=components(model,b)
            # components contains equally sized clean and degraded halves for one image.
            n=before.numel()//2
            checks.append(dict(asset_id=data.records[index]['asset_id'],preserved=preserves(after,before,before),
                clean_max_increase=float((after[:n]-before[:n]).max()),hard_max_increase=float((after[n:]-before[n:]).max()),
                component_deltas=(after-before).tolist()))
    accepted=sum(r['accepted'] for r in records)
    assessment=dict(accepted=accepted,rejected=24-accepted,diagnostic_preserved=sum(r['preserved'] for r in checks),
        diagnostic_total=8,feasible=accepted>0 and all(r['preserved'] for r in checks),promoted=False,
        proposals=records,diagnostics=checks,limitation='Training surrogates; unchanged model is not improvement.')
    _atomic_save(dict(model=model.state_dict(),config=payload.get('config',{}),step=accepted,experimental=True,
                     license='Non-commercial v5 derivative'),out/'candidate.pt')
    save(out/'assessment.json',assessment); save(out/'status.json',dict(state='complete',done=24,total=24,accepted=accepted))
    print(json.dumps({k:v for k,v in assessment.items() if k not in ('proposals','diagnostics')},indent=2))


if __name__=='__main__': main()
