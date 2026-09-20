"""Freeze and score a scene-balanced validation diagnostic; never select on test."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rudra.batch import digest


def capture_environment(out):
    versions = {}
    for package in ('torch', 'numpy', 'opencv-python', 'opencv-python-headless', 'cvvdp'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    files = ('training/quality_benchmark.py', 'training/infer_sdr2hdr.py',
             'training/export_bench_pairs.py', 'training/sdr2hdr_dataset.py',
             'rudra/sdr2hdr.py', 'rudra/hdrvdp.py', 'rudra/delivery/bench.py')
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        platform=platform.platform(), versions=versions,
        code_sha256={name:digest(ROOT/name) for name in files}),indent=2),encoding='utf-8')


def select_scenes(rows, count):
    if count < 1:
        raise ValueError('Scene count must be positive')
    groups = {}
    excluded = {r['scene_id'] for r in rows if r['split'] != 'val'}
    for row in rows:
        if row['split'] == 'val':
            if row['scene_id'] in excluded:
                raise ValueError('Scene appears across validation and another split')
            groups.setdefault(row['scene_id'], []).append(row)
    order = lambda value: hashlib.sha256(('rudra-quality-v1:'+value).encode()).hexdigest()
    return [min(groups[s], key=lambda r: order(r['asset_id']))
            for s in sorted(groups, key=order)[:count]]


def paired_summary(records, metric):
    pairs = {}
    for row in records:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            pairs.setdefault(row['asset_id'], {})[row['method']] = value
    pairs = [p for p in pairs.values() if set(p) == {'shipped', 'baseline'}]
    if not pairs:
        return dict(pairs=0)
    baseline = np.array([p['baseline'] for p in pairs])
    shipped = np.array([p['shipped'] for p in pairs])
    delta = shipped-baseline
    rng = np.random.default_rng(20260920)
    means = delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
    return dict(pairs=len(pairs), baseline=float(baseline.mean()), shipped=float(shipped.mean()),
                delta=float(delta.mean()), wins=int((delta>0).sum()),
                ties=int((delta==0).sum()), ci95=np.quantile(means,[.025,.975]).tolist())


def write_report(out, protocol, records):
    summary = {c:{m:paired_summary([r for r in records if r['condition']==c],m)
                  for m in ('pu21_db','cvvdp_jod','shadows_pu21_db','highlights_pu21_db')}
               for c in ('clean','hard')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines = ['# RUDRA quality diagnostic', '',
        'This is a validation diagnostic, not a release qualification or evidence of superiority over Ruby.', '',
        f"Sample: {len(protocol['rows'])} scenes, one deterministically selected frame per scene, native resolution.",
        'Clean and seeded synthetic-degradation conditions use the same HDR reference. Each scene has equal weight.',
        'Inference: CPU, preservation enabled, both recovery gates, 512-pixel tiles with 64-pixel overlap.',
        'Metrics: absolute linear Rec.2020 nits; PU21 per RGB channel and real ColorVideoVDP image JOD',
        '(standard_hdr_linear display). Higher is better. No exposure fitting. References are not clamped to the model ceiling.',
        'The analytic baseline is RUDRA’s inverse-ACES conversion, not a competing learned model.', '',
        '| Condition / metric | Baseline | Shipped | Difference | Paired 95% bootstrap interval | Wins / pairs |',
        '|---|---:|---:|---:|---|---:|']
    for condition, metrics in summary.items():
        for metric, s in metrics.items():
            if s['pairs']:
                lines.append(f"| {condition} / {metric} | {s['baseline']:.4f} | {s['shipped']:.4f} | {s['delta']:+.4f} | [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | {s['wins']}/{s['pairs']} |")
            else:
                lines.append(f'| {condition} / {metric} | Unavailable | Unavailable | — | — | 0 |')
    lines += ['', '## Limits and next decisions', '',
        '- Small validation sample; intervals describe this sample, not all production footage.',
        '- Shadows: reference luminance below 10 nits; highlights: above 203 nits. Empty regions are omitted, with paired counts shown.',
        '- Non-finite metric values are recorded as unavailable, never replaced with proxy scores.',
        '- This collection is not manually labeled for skin, neon, grain, or sky coverage. Those categories remain unverified.',
        '- Still-frame JOD does not measure temporal stability. No flicker or motion-quality claim is supported.',
        '- Ruby: pending matching outputs, verified color interpretation, and common delivery conditions. No ranking is possible.',
        '- New training candidates: pending completed training and validation selection. No weights promoted.',
        '- Held-out test data remains untouched. Do not tune models on that set.',
        '- Commercial rights and blind HDR-display review remain separate release requirements.', '',
        '## Recommended follow-up', '',
        'Inspect the largest negative paired differences in scores.jsonl. Prioritize fixes only when a pattern repeats across scenes.',
        'After training finishes, run the pre-existing full candidate assessment before evaluating the selected model on test.',
        'Extend this protocol with manually verified content categories and consecutive video references before making public quality claims.', '',
        'Reproduction and source/checkpoint hashes are in protocol.json; the exact metric records are in scores.jsonl.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scenes',type=int,default=12)
    a=p.parse_args(argv)
    rows=[json.loads(line) for line in a.manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected=select_scenes(rows,a.scenes)
    if not selected: raise ValueError('No validation scenes')
    a.out.mkdir(parents=True,exist_ok=False)
    capture_environment(a.out)
    protocol=dict(version=1,manifest=str(a.manifest.resolve()),manifest_sha256=digest(a.manifest),
        checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=digest(a.checkpoint),
        script_sha256=digest(__file__),rows=selected, selection='SHA256 rudra-quality-v1 scene then asset ordering',
        split='val',device='cpu',tile_size=512,overlap=64,reference_nits_scale=10000,
        cvvdp_display='standard_hdr_linear',ruby='pending',candidate='pending',test_set_used=False)
    for row in selected:
        row['sdr_sha256']=digest(row['sdr_path'])
        row['hdr_sha256']=digest(row['hdr_path'])
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    import torch
    from rudra.delivery.bench import pu_psnr
    from rudra.hdrvdp import hdr_vdp3_jod, cvvdp_last_error
    from rudra.sdr2hdr import sdr_to_baseline_hdr
    from training.infer_sdr2hdr import load_models, predict_image
    from training.export_bench_pairs import degrade_like_eval
    from training.sdr2hdr_dataset import load_rgb
    torch.set_num_threads(4)
    model=load_models(str(a.checkpoint),None,torch.device('cpu'))[0]
    records=[]
    with torch.inference_mode(), (a.out/'scores.jsonl').open('x',encoding='utf-8') as stream:
        for index,row in enumerate(selected):
            source=torch.from_numpy(load_rgb(row['sdr_path'],False)).permute(2,0,1)[None]
            ref=load_rgb(row['hdr_path'],True,ceiling=1000)*10000
            if tuple(source.shape[-2:]) != ref.shape[:2]: raise ValueError('Reference geometry mismatch')
            luminance=ref@np.array([.2627,.6780,.0593])
            masks={'shadows':luminance<10,'highlights':luminance>203}
            for condition in ('clean','hard'):
                x=source if condition=='clean' else degrade_like_eval(source[0],index)[None]
                for method in ('baseline','shipped'):
                    started=time.monotonic()
                    pred=sdr_to_baseline_hdr(x) if method=='baseline' else predict_image(model,x,True,512,64,'all')
                    elapsed=time.monotonic()-started
                    nits=pred[0].permute(1,2,0).numpy()*10000
                    if not np.isfinite(nits).all() or np.any(nits<0): raise ValueError('Invalid prediction')
                    pu=pu_psnr(nits,ref)
                    jod,backend=hdr_vdp3_jod(pred,torch.from_numpy(ref/10000).permute(2,0,1)[None],color_space='rec2020',diffuse_white_nits=10000)
                    record=dict(asset_id=row['asset_id'],scene_id=row['scene_id'],condition=condition,method=method,
                        pu21_db=float(pu) if np.isfinite(pu) else None,
                        cvvdp_jod=float(jod) if backend=='colorvideovdp' and np.isfinite(jod) else None,
                        cvvdp_backend=backend,cvvdp_error=cvvdp_last_error() if backend!='colorvideovdp' else None,
                        inference_seconds=elapsed,height=ref.shape[0],width=ref.shape[1])
                    for name,mask in masks.items():
                        value=pu_psnr(nits[mask],ref[mask]) if mask.any() else float('nan')
                        record[name+'_pu21_db']=float(value) if np.isfinite(value) else None
                        record[name+'_pixels']=int(mask.sum())
                    records.append(record)
                    stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            print(f'Scored scene {index+1}/{len(selected)}',flush=True)
    write_report(a.out,protocol,records)
    return 0


if __name__=='__main__': raise SystemExit(main())
