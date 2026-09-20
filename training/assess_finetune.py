"""Compare candidates on fixed scene-balanced validation; no automatic promotion.

Uses native frames, unclamped references, PU21 and real ColorVideoVDP.
Validation selects a seed; a passing seed is then measured once on the full
frozen test split. Reported values are specific to this experiment's corpus.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rudra.delivery.bench import pu_psnr
from rudra.hdrvdp import hdr_vdp3_jod, colorvideovdp_available
from rudra.sdr2hdr import sdr_to_baseline_hdr
from training.infer_sdr2hdr import load_models, predict_image
from training.export_bench_pairs import degrade_like_eval
from training.sdr2hdr_dataset import load_rgb, read_jsonl


@torch.inference_mode()
def score(rows, checkpoints, destination):
    device = torch.device('cuda')
    models = {name: load_models(str(path), None, device)[0] for name,path in checkpoints.items()}
    values = defaultdict(list)
    with destination.open('x', encoding='utf-8') as output:
        for index, row in enumerate(rows):
            source = torch.from_numpy(load_rgb(row['sdr_path'], False)).permute(2,0,1)[None]
            reference = torch.from_numpy(load_rgb(row['hdr_path'], True, ceiling=1000)).permute(2,0,1)[None].to(device)
            ref_nits = reference[0].permute(1,2,0).cpu().numpy()*10000
            for condition in ('clean','hard'):
                x = source if condition == 'clean' else degrade_like_eval(source[0], index)[None]
                x = x.to(device)
                for name in ('baseline', *models):
                    pred = sdr_to_baseline_hdr(x) if name == 'baseline' else predict_image(models[name],x,True,0,64,'all')
                    jod, backend = hdr_vdp3_jod(pred,reference,color_space='rec2020',diffuse_white_nits=10000)
                    if backend != 'colorvideovdp' or not np.isfinite(jod):
                        raise RuntimeError('Real CVVDP unavailable or invalid; refusing proxy scores')
                    pu = pu_psnr(pred[0].permute(1,2,0).cpu().numpy()*10000,ref_nits)
                    if not np.isfinite(pu):
                        raise RuntimeError('Non-finite PU21 score')
                    record = dict(asset_id=row['asset_id'],scene_id=row['scene_id'],condition=condition,
                                  method=name,pu21_db=float(pu),cvvdp_jod=float(jod))
                    output.write(json.dumps(record)+'\n')
                    values[f'{condition}/{name}'].append((pu,jod))
            output.flush()
            print(f'{destination.stem}: {index+1}/{len(rows)}',flush=True)
    return {k:dict(pu21_db=float(np.mean(v,axis=0)[0]),cvvdp_jod=float(np.mean(v,axis=0)[1]),frames=len(v))
            for k,v in values.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    a = p.parse_args()
    if not colorvideovdp_available(): raise RuntimeError('Install real CVVDP before assessment')
    rows = read_jsonl(a.run_dir/'data'/'manifest.jsonl')
    val_by_scene = {}
    for r in rows:
        if r['split']=='val': val_by_scene.setdefault(r['scene_id'],r)
    candidates = {f'seed{s}':a.run_dir/f'gate_seed{s}'/'best.pt' for s in (20260920,2,3)}
    checkpoints = {'shipped':ROOT/'checkpoints/sdr2hdr_shadow_v1.pt',**candidates}
    val = score(list(val_by_scene.values()),checkpoints,a.run_dir/'validation_scores.jsonl')
    passing = [name for name in candidates if all(val[f'{c}/{name}'][m] >= val[f'{c}/shipped'][m]
                    for c in ('clean','hard') for m in ('pu21_db','cvvdp_jod'))]
    report = dict(validation=val,passing_seeds=passing,
                  validation_protocol='First manifest frame per scene, native resolution; fixed order',
                  criterion='No regression against shipped model on either metric in either condition',
                  promoted=False,license='Non-commercial v5 derivative')
    if passing:
        winner = max(passing,key=lambda n:sum(val[f'{c}/{n}']['cvvdp_jod'] for c in ('clean','hard')))
        report['selected_seed'] = winner
        report['test'] = score([r for r in rows if r['split']=='test'],
             {'shipped':checkpoints['shipped'],winner:candidates[winner]},a.run_dir/'test_scores.jsonl')
    else:
        report['decision'] = 'No candidate passed validation; keep the shipped model, test set unscored'
    (a.run_dir/'assessment.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__': main()
