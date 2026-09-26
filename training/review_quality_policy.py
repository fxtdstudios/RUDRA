"""CPU-only review of a completed policy experiment; never reads image pixels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rudra.batch import digest, save
from rudra.recovery_policy import MODES, load_policy


def compare(records, choices):
    """Paired scene bootstrap within each condition, descriptive after selection."""
    if len(records) != len(choices):
        raise ValueError('Choice count does not match validation records')
    report = {}
    for condition in ('clean', 'hard'):
        pairs = [(r, c) for r, c in zip(records, choices) if r['condition'] == condition]
        if not pairs or len({r['scene_id'] for r, _ in pairs}) != len(pairs):
            raise ValueError('Expected one frame per scene per condition')
        result = {'scenes': len(pairs), 'mode_counts': dict(Counter(MODES[c] for _, c in pairs))}
        rng = np.random.default_rng(20260925)
        resamples = rng.integers(0, len(pairs), size=(2000, len(pairs)))
        for metric in ('pu21_db', 'cvvdp_jod'):
            delta = np.array([r[metric][c] - r[metric][3] for r, c in pairs])
            if not np.isfinite(delta).all():
                raise ValueError('Non-finite validation scores')
            result[metric] = {
                'mean_delta': float(delta.mean()),
                'descriptive_95_percent_interval': np.quantile(delta[resamples].mean(1), [.025, .975]).tolist(),
                'improved': int((delta > 1e-7).sum()),
                'regressed': int((delta < -1e-7).sum()),
                'unchanged': int((abs(delta) <= 1e-7).sum()),
                'worst_scenes': [dict(scene_id=pairs[i][0]['scene_id'], asset_id=pairs[i][0]['asset_id'],
                                      mode=MODES[pairs[i][1]], delta=float(delta[i]))
                                 for i in np.argsort(delta)[:10]],
            }
        report[condition] = result
    return report


def review(run):
    state = json.loads((run / 'status.json').read_text())
    if state['state'] != 'complete':
        raise RuntimeError(f"Experiment is {state['state']}; review requires completion")
    assessment = json.loads((run / 'assessment.json').read_text())
    selected = assessment['selected']
    result = dict(selected=selected, promoted=False, test_pixels_used=False,
                  decision=assessment['decision'],
                  limitation='Same validation data used for candidate selection. Intervals are descriptive, '
                             'not independent confirmation. Visual and temporal review remain required.')
    if selected:
        protocol = json.loads((run / 'protocol.json').read_text())
        if tuple(protocol['modes']) != MODES:
            raise ValueError('Recovery action order mismatch')
        if digest(protocol['checkpoint']) != protocol['backbone_sha256']:
            raise ValueError('Reconstruction checkpoint changed since measurement')
        records = [json.loads(line) for line in (run / 'measurements.jsonl').read_text().splitlines()]
        if any(r['split'] not in ('train', 'val') for r in records):
            raise ValueError('Unexpected split in measurements')
        cache = torch.load(run / 'feature_cache.pt', map_location='cpu', weights_only=True)
        indices = cache['val'].tolist()
        if indices != [i for i, r in enumerate(records) if r['split'] == 'val']:
            raise ValueError('Validation cache does not match measurement order')
        if cache['features'].shape[0] != len(records):
            raise ValueError('Feature count does not match measurements')
        policy = load_policy(selected['path'], protocol['checkpoint'])
        with torch.inference_mode():
            choices = policy.decisions(cache['features'][indices]).tolist()
        validation = [records[i] for i in indices]
        # Check that replaying the saved checkpoint reproduces selection results.
        from training.train_quality_policy import summarize
        replay = summarize(validation, choices)
        for condition in ('clean', 'hard'):
            for metric in ('pu21_db', 'cvvdp_jod'):
                if not np.isclose(replay[condition]['policy'][metric],
                                  selected['validation'][condition]['policy'][metric], rtol=0, atol=1e-6):
                    raise ValueError('Saved checkpoint does not reproduce selected scores')
        result['comparisons'] = compare(validation, choices)
        result['policy_sha256'] = digest(selected['path'])
    save(run / 'review.json', result)
    lines = ['# Recovery policy review', '', result['decision'], '', result['limitation'], '',
             'The Studio model has not been replaced. Test images were not read by this review.', '']
    if selected:
        lines += [f"Selected checkpoint: `{selected['path']}`", '',
                  '| Condition | Metric | Mean change vs shipped | Descriptive 95% interval | Regressed scenes |',
                  '|---|---|---:|---|---:|']
        for condition, stats in result['comparisons'].items():
            for metric in ('pu21_db', 'cvvdp_jod'):
                m = stats[metric]
                lo, hi = m['descriptive_95_percent_interval']
                lines.append(f"| {condition} | {metric} | {m['mean_delta']:+.5f} | {lo:+.5f} to {hi:+.5f} | {m['regressed']}/{stats['scenes']} |")
        lines += ['', 'Mode counts and the ten worst scenes for each metric are in `review.json`.',
                  'Inspect those scenes visually, then evaluate sequence flicker before any release decision.']
    (run / 'review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--wait-seconds', type=float, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    deadline = time.monotonic() + args.wait_seconds
    while True:
        state = json.loads((args.run / 'status.json').read_text())
        if state['state'] in ('complete', 'failed') or time.monotonic() >= deadline:
            break
        time.sleep(min(5, max(0, deadline - time.monotonic())))
    review(args.run)
    print(f"Review written to {args.run / 'review.md'}", flush=True)


if __name__ == '__main__':
    main()
