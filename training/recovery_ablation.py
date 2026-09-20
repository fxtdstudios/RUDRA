"""Isolate recovery paths against an existing, frozen validation diagnostic."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from rudra.batch import digest
from training.quality_benchmark import paired_summary

METRICS = ('pu21_db', 'cvvdp_jod', 'shadows_pu21_db', 'highlights_pu21_db')
MODES = ('highlights', 'shadows', 'off')


def validate_frozen(folder):
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    if protocol['split'] != 'val' or protocol['test_set_used']:
        raise ValueError('A validation-only protocol is required')
    if digest(protocol['checkpoint']) != protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    for row in protocol['rows']:
        if row['split'] != 'val': raise ValueError('Non-validation row')
        for kind in ('sdr', 'hdr'):
            if digest(row[kind+'_path']) != row[kind+'_sha256']:
                raise ValueError(f'{kind} source changed: {row["asset_id"]}')
    environment = json.loads((folder/'environment.json').read_text(encoding='utf-8'))
    for name, expected in environment['code_sha256'].items():
        # Selection and old scores are frozen inputs, not rerun by this script.
        # Verify the inference/decoder/metric implementations that we do execute.
        if name == 'training/quality_benchmark.py': continue
        if digest(ROOT/name) != expected:
            raise ValueError(f'Benchmark implementation changed: {name}')
    records = [json.loads(line) for line in (folder/'scores.jsonl').read_text().splitlines()]
    keys = {(r['asset_id'], r['condition'], r['method']) for r in records}
    expected = {(r['asset_id'], c, m) for r in protocol['rows']
                for c in ('clean', 'hard') for m in ('baseline', 'shipped')}
    if keys != expected or len(records) != len(expected):
        raise ValueError('Frozen results are incomplete or duplicated')
    return protocol, records


def compare(records, method, reference, condition, metric):
    selected = []
    for row in records:
        if row['condition'] == condition and row['method'] in (method, reference):
            copy = dict(row)
            copy['method'] = 'shipped' if row['method'] == method else 'baseline'
            selected.append(copy)
    return paired_summary(selected, metric)


def report(out, records):
    summaries = {}
    modes = [m for m in MODES if any(r['method']==m for r in records)]
    lines = ['# Recovery-path ablation', '',
        f'Fixed {len({r["asset_id"] for r in records})}-scene validation diagnostic: native resolution, CPU, 512-pixel tiles, 64-pixel overlap.',
        'All recovery and analytic baseline scores are reused from the hash-verified original run.',
        'Measured modes: '+', '.join(modes)+'. All use the shipped weights.',
        'The learned preservation masks stay enabled in every mode; this isolates the residual recovery paths.',
        'Recovery-off is measured, not assumed identical to the baseline: clipping and numerical differences can remain.', '',
        '| Condition | Mode | PU21 dB | Image JOD | Shadow PU21 dB | Highlight PU21 dB |',
        '|---|---|---:|---:|---:|---:|']
    for condition in ('clean', 'hard'):
        for method in ('baseline', 'shipped', *modes):
            group = [r for r in records if r['condition']==condition and r['method']==method]
            values = []
            for metric in METRICS:
                valid = [r[metric] for r in group if r[metric] is not None]
                values.append(f'{np.mean(valid):.4f}' if valid else 'Unavailable')
            lines.append(f'| {condition} | {method} | '+' | '.join(values)+' |')
        for method in modes:
            summaries[f'{condition}/{method}_vs_all'] = {
                metric:compare(records,method,'shipped',condition,metric) for metric in METRICS}
    lines += ['', 'Paired differences and 95% bootstrap intervals are in summary.json. Higher scores are better.',
        'Shadow pixels have reference luminance below 10 nits; highlights exceed 203 nits. Region sample counts vary.', '',
        '## Interpretation limits', '',
        '- Content-category coverage is not manually verified. Real motion and temporal stability are untested.',
        '- Hard inputs are seeded synthetic degradations, not independent real damaged footage.',
        '- Selecting a mode from this sample is exploratory; confirm on the remaining validation scenes.',
        '- A mode that helps clean inputs may remove useful recovery on degraded inputs.',
        '- No production defaults, model weights, training settings, or held-out test data were changed.',
        '- Ruby comparison remains pending; this is an internal ablation only.']
    (out/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    a=p.parse_args(argv)
    protocol,records=validate_frozen(a.benchmark)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'protocol.json').write_text(json.dumps(dict(parent=str(a.benchmark.resolve()),
        parent_protocol_sha256=digest(a.benchmark/'protocol.json'),
        parent_scores_sha256=digest(a.benchmark/'scores.jsonl'),
        script_sha256=digest(__file__),modes=a.modes,checkpoint_sha256=protocol['checkpoint_sha256']),indent=2))
    import torch
    from rudra.delivery.bench import pu_psnr
    from rudra.hdrvdp import hdr_vdp3_jod
    from training.infer_sdr2hdr import load_models, predict_image
    from training.export_bench_pairs import degrade_like_eval
    from training.sdr2hdr_dataset import load_rgb
    torch.set_num_threads(4)
    model=load_models(protocol['checkpoint'],None,torch.device('cpu'))[0]
    with torch.inference_mode(), (a.out/'scores.jsonl').open('x',encoding='utf-8') as stream:
        for record in records: stream.write(json.dumps(record)+'\n')
        for index,row in enumerate(protocol['rows']):
            source=torch.from_numpy(load_rgb(row['sdr_path'],False)).permute(2,0,1)[None]
            ref=load_rgb(row['hdr_path'],True,ceiling=1000)*10000
            if tuple(source.shape[-2:]) != ref.shape[:2]: raise ValueError('Geometry mismatch')
            reference=torch.from_numpy(ref/10000).permute(2,0,1)[None]
            luminance=ref@np.array([.2627,.6780,.0593])
            masks={'shadows':luminance<10,'highlights':luminance>203}
            for condition in ('clean','hard'):
                x=source if condition=='clean' else degrade_like_eval(source[0],index)[None]
                for mode in dict.fromkeys(a.modes):
                    started=time.monotonic()
                    pred=predict_image(model,x,True,512,64,mode)
                    seconds=time.monotonic()-started
                    nits=pred[0].permute(1,2,0).numpy()*10000
                    if not np.isfinite(nits).all() or np.any(nits<0): raise ValueError('Invalid output')
                    jod,backend=hdr_vdp3_jod(pred,reference,color_space='rec2020',diffuse_white_nits=10000)
                    if backend!='colorvideovdp' or not np.isfinite(jod):
                        raise RuntimeError('Real ColorVideoVDP required; refusing proxy results')
                    pu=pu_psnr(nits,ref)
                    record=dict(asset_id=row['asset_id'],condition=condition,method=mode,
                        scene_id=row['scene_id'],pu21_db=float(pu) if np.isfinite(pu) else None,
                        cvvdp_jod=float(jod),cvvdp_backend=backend,inference_seconds=seconds)
                    for name,mask in masks.items():
                        value=pu_psnr(nits[mask],ref[mask]) if mask.any() else float('nan')
                        record[name+'_pu21_db']=float(value) if np.isfinite(value) else None
                        record[name+'_pixels']=int(mask.sum())
                    records.append(record)
                    stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush()
            print(f'Ablated scene {index+1}/{len(protocol["rows"])}',flush=True)
    report(a.out,records)
    return 0


if __name__=='__main__': raise SystemExit(main())
