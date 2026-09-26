"""Compare a fixed candidate against hash-verified cached shipped measurements.

Same exploratory validation, same degradation seeds and inference settings.
This is not independent evaluation and never promotes a checkpoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from rudra.batch import digest, save
from rudra.delivery.bench import pu_psnr
from rudra.hdrvdp import hdr_vdp3_jod
from training.infer_sdr2hdr import load_models, predict_image
from training.export_bench_pairs import degrade_like_eval
from training.sdr2hdr_dataset import load_rgb


def run(candidate, source, out):
    protocol = json.loads((source/'protocol.json').read_text())
    if digest(protocol['checkpoint']) != protocol['backbone_sha256']:
        raise ValueError('Shipped checkpoint changed')
    if json.loads((source/'status.json').read_text())['state'] != 'complete':
        raise ValueError('Incomplete measurement source')
    if protocol['modes'] != ['off','highlights','shadows','all']:
        raise ValueError('Unexpected recovery mode order')
    rows = [r for r in protocol['rows'] if r['split']=='val']
    records = [json.loads(s) for s in (source/'measurements.jsonl').read_text().splitlines()]
    baseline = {(r['asset_id'],r['condition']):r for r in records if r['split']=='val'}
    if not rows or len(baseline) != 2*len(rows):
        raise ValueError('Incomplete validation pairs')
    out.mkdir(parents=True, exist_ok=False)
    save(out/'protocol.json',dict(candidate=str(candidate.resolve()), candidate_sha256=digest(candidate),
        source=str(source.resolve()), measurements_sha256=digest(source/'measurements.jsonl'),
        script_sha256=digest(__file__), promoted=False, test_pixels_used=False,
        limitation='Reused exploratory validation with cached shipped baseline; not independent evaluation'))
    state = dict(state='running',pid=os.getpid(),frames_done=0,frames_total=len(rows)*2)
    save(out/'status.json',state)
    torch.set_num_threads(4)
    model=load_models(str(candidate),None,torch.device('cuda'))[0].eval()
    results=[]
    try:
        with torch.inference_mode(), (out/'measurements.jsonl').open('x') as log:
            for row in rows:
                for condition in ('clean','hard'):
                    base=baseline[row['asset_id'],condition]
                    if digest(row['sdr_path'])!=base['sdr_sha256'] or digest(row['hdr_path'])!=base['hdr_sha256']:
                        raise ValueError('Source image changed')
                    x=torch.from_numpy(load_rgb(row['sdr_path'],False)).permute(2,0,1)
                    seed=int(hashlib.sha256(row['asset_id'].encode()).hexdigest()[:7],16)
                    if condition=='hard': x=degrade_like_eval(x,seed)
                    ref=load_rgb(row['hdr_path'],True,ceiling=1000)*10000
                    target=torch.from_numpy(ref/10000).permute(2,0,1)[None].cuda()
                    pred=predict_image(model,x[None].cuda(),True,512,64,'all')
                    pu=float(pu_psnr(pred[0].permute(1,2,0).cpu().numpy()*10000,ref))
                    jod,backend=hdr_vdp3_jod(pred,target,color_space='rec2020',diffuse_white_nits=10000)
                    if backend!='colorvideovdp' or not np.isfinite([pu,jod]).all():
                        raise ValueError('Invalid quality measurement')
                    result=dict(asset_id=row['asset_id'],condition=condition,pu21_db=pu,cvvdp_jod=float(jod),
                                pu21_delta=pu-base['pu21_db'][3],jod_delta=float(jod)-base['cvvdp_jod'][3])
                    results.append(result); log.write(json.dumps(result)+'\n'); log.flush()
                    state['frames_done']+=1; save(out/'status.json',state)
        summary={c:{m:float(np.mean([r[m] for r in results if r['condition']==c]))
                    for m in ('pu21_delta','jod_delta')} for c in ('clean','hard')}
        passing=all(v>=0 for values in summary.values() for v in values.values())
        save(out/'assessment.json',dict(summary=summary,passing=passing,promoted=False,
             limitation='Reused validation; independent and temporal verification still required'))
        state.update(state='complete',passing=passing)
    except BaseException as exc:
        state.update(state='failed',error=repr(exc)); raise
    finally: save(out/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); run(a.candidate,a.source,a.out)
