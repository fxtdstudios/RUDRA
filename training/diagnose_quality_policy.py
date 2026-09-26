"""Explain policy regressions using saved train/validation scores, never test pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rudra.batch import digest, save
from rudra.recovery_policy import MODES, load_policy


def group_stats(records, choices):
    q = np.asarray([r['pu21_db'] for r in records], dtype=float)
    choices = np.asarray(choices, dtype=int)
    if q.shape != (len(choices), 4) or not np.isfinite(q).all():
        raise ValueError('Invalid scores or choices')
    oracle = q.argmax(1)
    index = np.arange(len(q))
    selected = q[index, choices]
    result = dict(frames=len(q), policy_modes=dict(Counter(MODES[c] for c in choices)),
                  oracle_modes=dict(Counter(MODES[c] for c in oracle)),
                  policy_pu21_delta=float((selected-q[:, 3]).mean()),
                  oracle_pu21_delta=float((q[index, oracle]-q[:, 3]).mean()),
                  mean_pu21_regret=float((q[index, oracle]-selected).mean()),
                  pu21_regressed_frames=int((selected < q[:, 3]-1e-7).sum()),
                  fixed_mode_pu21_delta={mode:float((q[:, i]-q[:, 3]).mean()) for i,mode in enumerate(MODES)})
    if all(len(r['cvvdp_jod']) == 4 for r in records):
        j = np.asarray([r['cvvdp_jod'] for r in records])
        result.update(policy_jod_delta=float((j[index, choices]-j[:, 3]).mean()),
                      pu21_oracle_jod_delta=float((j[index, oracle]-j[:, 3]).mean()),
                      jod_regressed_frames=int((j[index, choices] < j[:, 3]-1e-7).sum()),
                      pu21_oracle_jod_regressed_frames=int((j[index, oracle] < j[:, 3]-1e-7).sum()))
    return result


def diagnose(run):
    assessment = json.loads((run/'assessment.json').read_text())
    protocol = json.loads((run/'protocol.json').read_text())
    records = [json.loads(line) for line in (run/'measurements.jsonl').read_text().splitlines()]
    cache = torch.load(run/'feature_cache.pt', map_location='cpu', weights_only=True)
    if tuple(protocol['modes']) != MODES or digest(protocol['checkpoint']) != protocol['backbone_sha256']:
        raise ValueError('Experiment/checkpoint mismatch')
    if any(r['split'] not in ('train','val') for r in records):
        raise ValueError('Unexpected split')
    if cache['features'].shape[0] != len(records):
        raise ValueError('Feature count mismatch')
    for split in ('train','val'):
        if cache[split].tolist() != [i for i,r in enumerate(records) if r['split']==split]:
            raise ValueError('Cache order mismatch')
    if not np.allclose(cache['quality'].numpy(), [r['pu21_db'] for r in records]):
        raise ValueError('Cached quality does not match measurements')
    candidate = max(assessment['candidates'], key=lambda c:sum(c['validation'][k]['policy']['cvvdp_jod'] for k in ('clean','hard')))
    policy = load_policy(candidate['path'], protocol['checkpoint'])
    with torch.inference_mode():
        choices = policy.decisions(cache['features']).tolist()
    result = dict(diagnostic_candidate=candidate['path'], promoted=False, test_pixels_used=False,
                  limitation='Retrospective reused train/validation data; oracle uses HDR references and is not deployable.', groups={})
    for split in ('train','val'):
        for condition in ('clean','hard'):
            indices = [i for i,r in enumerate(records) if r['split']==split and r['condition']==condition]
            stats = group_stats([records[i] for i in indices], [choices[i] for i in indices])
            if split == 'val':
                for metric, delta in (('pu21_db','policy_pu21_delta'), ('cvvdp_jod','policy_jod_delta')):
                    expected = candidate['validation'][condition]['policy'][metric]-candidate['validation'][condition]['shipped'][metric]
                    if not np.isclose(stats[delta], expected, atol=1e-6, rtol=0):
                        raise ValueError('Candidate replay mismatch')
            result['groups'][split+'_'+condition] = stats
    save(run/'diagnosis.json', result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    diagnose(args.run)
