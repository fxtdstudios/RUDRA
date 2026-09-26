"""Read-only retrospective audit; never assigns existing scenes to an independent set."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = Path('outputs/finetune_views_20260920/data/manifest.jsonl')
    data = rows(manifest)
    by_id = {r['asset_id']: r for r in data}
    scenes = defaultdict(set)
    for r in data:
        scenes[r['scene_id']].add(r['split'])
    inventory = {}
    for name in ('source_inventory', 'inv_local', 'inv_nas', 'inv_stuttgart2014'):
        path = Path('E:/RUDRA_v3_20260822') / (name + '.jsonl')
        if not path.exists():
            continue
        records = rows(path)
        ids = {r.get('scene_id') for r in records} - {None}
        unknown = sorted(ids - scenes.keys())
        inventory[name] = dict(records=len(records), scenes=len(ids),
                               exact_scene_overlap=len(ids & scenes.keys()),
                               unmatched_scene_ids=unknown,
                               note='Unmatched identifiers are NOT proof of independence or license clearance.')
    missing = {key: sum(not Path(r[key]).is_file() for r in data)
               for key in ('sdr_path', 'hdr_path', 'metadata_path')}
    runs = {}
    review = []
    for run in ('robust_backbone_20260925', 'robust_backbone_20260926_192336'):
        path = Path('outputs') / run / 'evaluation/measurements.jsonl'
        measurements = rows(path)
        summary = {}
        for condition in ('clean', 'hard'):
            group = [r for r in measurements if r['condition'] == condition]
            summary[condition] = dict(count=len(group), **{
                metric: dict(mean=float(np.mean([r[metric] for r in group])),
                             regressed=sum(r[metric] < -1e-6 for r in group),
                             worst=sorted(group, key=lambda r:r[metric])[:5])
                for metric in ('pu21_delta', 'jod_delta')})
            if run.endswith('192336'):
                for r in sorted(group, key=lambda r:r['jod_delta'])[:10]:
                    source = by_id[r['asset_id']]
                    review.append(dict(**r, scene_id=source['scene_id'],
                                       sdr_path=source['sdr_path'], hdr_path=source['hdr_path'],
                                       usage='diagnostic reused validation; never independent evaluation'))
        runs[run] = summary
    report = dict(manifest=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  records=dict(Counter(r['split'] for r in data)),
                  scenes={s:sum(s in splits for splits in scenes.values()) for s in ('train','val','test')},
                  split_leakage=[s for s,v in scenes.items() if len(v)>1], missing_files=missing,
                  video_records=dict(Counter(r['split'] for r in data if r.get('is_video'))),
                  inventories=inventory, runs=runs,
                  independent_set_status='NOT LOCKED: prior-use and license verification required',
                  limitation='Saved measurements only; no new visual, temporal or independent evaluation; no promotion.')
    (args.out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (args.out/'diagnostic_review.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in review),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('records','scenes','missing_files','split_leakage','video_records')}))
    print(json.dumps({k:dict(scenes=v['scenes'],unmatched=len(v['unmatched_scene_ids'])) for k,v in inventory.items()}))


if __name__ == '__main__':
    main()
