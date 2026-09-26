"""Fixed-budget paired backbone fine-tune. No validation selection or promotion."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from rudra.batch import digest, save
from training.sdr2hdr_dataset import SDRHDRDataset, read_jsonl
from training.train_sdr2hdr import load_image_checkpoint, seed_everything, _atomic_save
from training.robust_reconstruction import paired_objective, region_paired_objective


def check_splits(rows):
    scenes = {}
    for row in rows:
        scene, split = row['scene_id'], row['split']
        if scene in scenes and scenes[scene] != split:
            raise ValueError('Scene leakage across splits')
        scenes[scene] = split
    if not any(s == 'train' for s in scenes.values()):
        raise ValueError('No training scenes')


def run(a):
    if a.steps < 1 or a.batch_size < 1 or a.lr <= 0:
        raise ValueError('Positive steps, batch size and learning rate required')
    rows = read_jsonl(a.manifest)
    objective_name = getattr(a, 'objective', 'log')
    objective_fn = region_paired_objective if objective_name == 'region-pu' else paired_objective
    check_splits(rows)
    a.out.mkdir(parents=True, exist_ok=False)
    seed_everything(a.seed)
    torch.set_num_threads(4)
    device = torch.device(a.device)
    model, payload = load_image_checkpoint(a.checkpoint, device)
    teacher, _ = load_image_checkpoint(a.checkpoint, device)
    teacher.eval().requires_grad_(False)
    # Preserve the frame-level switch: crops are not representative gate inputs.
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(('gate.', 'shadow_gate.')))
    data = SDRHDRDataset(a.manifest, split='train', crop_size=a.crop_size,
                         augment=True, degradation_probability=1.0)
    sizes = Counter(r['scene_id'] for r in data.records)
    sampler = WeightedRandomSampler([1/sizes[r['scene_id']] for r in data.records],
                                    a.steps*a.batch_size, replacement=True,
                                    generator=torch.Generator().manual_seed(a.seed))
    loader = DataLoader(data, sampler=sampler, batch_size=a.batch_size, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=a.lr, weight_decay=.01)
    config = dict(payload.get('config', {}))
    protocol = dict(checkpoint=str(a.checkpoint.resolve()), backbone_sha256=digest(a.checkpoint),
                    manifest=str(a.manifest.resolve()), manifest_sha256=digest(a.manifest),
                    script_sha256=digest(__file__), objective_sha256=digest(Path(__file__).with_name('robust_reconstruction.py')),
                    steps=a.steps, seed=a.seed, lr=a.lr, crop_size=a.crop_size,
                    batch_size=a.batch_size, train_scenes=len(sizes),
                    objective=('paired censored log + worst-condition region PU21 + 4x region harm'
                               if objective_name == 'region-pu' else
                               'paired censored log risk + 4x per-frame teacher-relative harm'),
                    objective_name=objective_name,
                    frozen='gate and shadow_gate; reconstruction backbone trainable',
                    validation_selection=False, test_pixels_used=False, promoted=False,
                    license='Non-commercial v5 derivative')
    save(a.out/'protocol.json', protocol)
    state = dict(state='running', stage='paired_backbone_training', pid=os.getpid(),
                 steps_done=0, steps_total=a.steps, started_at=time.time())
    save(a.out/'status.json', state)
    try:
        model.train()
        with (a.out/'train.jsonl').open('x') as log:
            for step, batch in enumerate(loader, 1):
                clean, hard, target, ceiling = [batch[k].to(device) for k in ('clean_sdr','sdr','hdr','ceiling')]
                with torch.no_grad():
                    teachers = [teacher(x, preserve_outside=True).hdr for x in (clean,hard)]
                optimizer.zero_grad(set_to_none=True)
                predictions = [model(x, preserve_outside=True).hdr for x in (clean,hard)]
                loss, parts = objective_fn(predictions, teachers, target, ceiling)
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if step == 1 or step % 10 == 0 or step == a.steps:
                    record = dict(step=step, loss=float(loss.detach()), **{k:float(v.detach()) for k,v in parts.items()})
                    log.write(json.dumps(record)+'\n'); log.flush()
                    state['steps_done'] = step; save(a.out/'status.json',state)
                    print(json.dumps(record), flush=True)
                if step == a.steps or step % 100 == 0:
                    _atomic_save(dict(model=model.state_dict(), config=config, step=step,
                                      optimizer=optimizer.state_dict(), experimental=True,
                                      license=protocol['license'], protocol=protocol),a.out/'candidate.pt')
        state.update(state='complete',stage='await_quality_evaluation',promoted=False)
    except BaseException as exc:
        state.update(state='failed',error=repr(exc)); raise
    finally:
        save(a.out/'status.json', state)
    if a.evaluation_source:
        from training.evaluate_robust_reconstruction import run as evaluate
        # Release training tensors before full-resolution metric evaluation.
        del model, teacher, optimizer, predictions, teachers, loss
        if device.type == 'cuda': torch.cuda.empty_cache()
        evaluate(a.out/'candidate.pt',a.evaluation_source,a.out/'evaluation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--crop-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--seed', type=int, default=20260925)
    p.add_argument('--device', default='cuda')
    p.add_argument('--objective', choices=['log','region-pu'], default='log')
    p.add_argument('--evaluation-source', type=Path)
    run(p.parse_args())
