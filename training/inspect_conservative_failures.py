"""List validation switches and regressions without fitting or reading image pixels."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rudra.batch import digest, save
from rudra.recovery_policy import load_policy, MODES
from training.train_quality_policy import summarize


def inspect(run):
    protocol=json.loads((run/'protocol.json').read_text())
    source=Path(protocol['source'])
    for name,key in [('feature_cache.pt','cache_sha256'),('measurements.jsonl','measurements_sha256')]:
        if digest(source/name)!=protocol[key]: raise ValueError('Source data changed')
    cache=torch.load(source/'feature_cache.pt',map_location='cpu',weights_only=True)
    records=[json.loads(line) for line in (source/'measurements.jsonl').read_text().splitlines()]
    indices=protocol['validation_indices']
    if indices!=[i for i,r in enumerate(records) if r['split']=='val']:
        raise ValueError('Validation indices mismatch')
    policy=load_policy(run/'candidate.pt',protocol['checkpoint'])
    with torch.inference_mode():
        probabilities=policy(cache['features'][indices]).softmax(1)
        choices=policy.decisions(cache['features'][indices]).tolist()
    validation=[records[i] for i in indices]
    replay=summarize(validation,choices)
    assessment=json.loads((run/'assessment.json').read_text())['candidate']
    for condition in ('clean','hard'):
        for metric in ('pu21_db','cvvdp_jod'):
            if abs(replay[condition]['policy'][metric]-assessment['validation'][condition]['policy'][metric])>1e-6:
                raise ValueError('Candidate replay mismatch')
    switches=[]
    for i,(record,choice) in enumerate(zip(validation,choices)):
        if choice==3: continue
        delta={metric:record[metric][choice]-record[metric][3] for metric in ('pu21_db','cvvdp_jod')}
        switches.append(dict(scene_id=record['scene_id'],asset_id=record['asset_id'],condition=record['condition'],
                             mode=MODES[choice],confidence=float(probabilities[i,choice]),delta=delta,
                             regressed=any(v < -1e-7 for v in delta.values())))
    result=dict(switches=switches,validation_frames=len(validation),threshold=policy.confidence_threshold,
                regressed_switches=sum(r['regressed'] for r in switches),promoted=False,test_pixels_used=False,
                limitation='Retrospective diagnosis on reused validation; no fitting or threshold adjustment.')
    save(run/'failure_cases.json',result)
    lines=['# Conservative selector failure cases','',result['limitation'],'',
           '| Scene | Condition | Mode | Softmax confidence | PU21 delta | JOD delta |',
           '|---|---|---|---:|---:|---:|']
    for r in sorted(switches,key=lambda r:r['delta']['cvvdp_jod']):
        lines.append(f"| {r['scene_id']} | {r['condition']} | {r['mode']} | {r['confidence']:.4f} | {r['delta']['pu21_db']:+.5f} | {r['delta']['cvvdp_jod']:+.5f} |")
    (run/'failure_cases.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(2); inspect(a.run)
