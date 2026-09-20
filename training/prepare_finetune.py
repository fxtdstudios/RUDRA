"""Freeze the existing split and add provenance-checked rendered panorama scenes.

This is a non-commercial v5 derivative experiment, not commercial clearance.
Source pixels are read only. The manifest and provenance are written to --out.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(original, moves, out):
    if out.exists():
        raise ValueError(f"Refusing to replace frozen dataset: {out}")
    rows = [json.loads(line) for line in original.read_text().splitlines() if line.strip()]
    searchable_rows = [(r, json.dumps(r).lower()) for r in rows]
    original_text = json.dumps(rows).lower()
    family_matches = {}
    index = moves / 'panoramas' / '_polyhaven_index.jsonl'
    sources = {r['file']: r for r in map(json.loads, index.read_text().splitlines())}
    excluded, added = set(), []
    for meta_path in sorted((moves / 'pairs_moves' / 'meta').glob('*.json')):
        meta = json.loads(meta_path.read_text())
        if meta['frame'] % 4:
            continue
        source = sources.get(Path(meta['source']).name)
        if source is None or source.get('license') != 'CC0':
            raise ValueError(f"Missing CC0 provenance: {meta_path}")
        slug = source['slug']
        # Conservatively group numbered variants and exclude any previously
        # represented family, so old held-out panoramas cannot enter training.
        family = re.sub(r'(?:_\d+)+$', '', slug)
        scene_id = 'polyhaven:' + family
        if family in original_text:
            if family not in family_matches:
                family_matches[family] = [r for r, text in searchable_rows if family in text]
            matches = family_matches[family]
            if {r['split'] for r in matches} != {'train'}:
                excluded.add(family)
                continue
            # Extra camera views of a training panorama are augmentation, not
            # new independent scenes. Keep its existing scene identity.
            scene_id, split = matches[0]['scene_id'], 'train'
        else:
            bucket = int(hashlib.sha256(('rudra-20260920:' + family).encode()).hexdigest()[:8], 16) % 100
            split = 'train' if bucket < 80 else ('val' if bucket < 90 else 'test')
        stem = meta['stem']
        added.append(dict(asset_id='moves_' + stem, scene_id=scene_id,
                          sdr_path=str(moves/'pairs_moves'/'sdr'/f'{stem}.png'),
                          hdr_path=str(moves/'pairs_moves'/'hdr'/f'{stem}.png'),
                          metadata_path=str(meta_path), split=split, is_video=False,
                          hdr_encoding='log2_extended', sdr_encoding='srgb',
                          source_url=source['url'], source_license='CC0'))
    rows += added
    scenes = defaultdict(set)
    paths = set()
    for row in rows:
        scenes[row['scene_id']].add(row['split'])
        for key in ('sdr_path', 'hdr_path'):
            path = Path(row[key])
            if not path.is_file():
                raise FileNotFoundError(path)
        pair = (row['sdr_path'], row['hdr_path'])
        if pair in paths:
            raise ValueError(f"Duplicate pair: {pair}")
        paths.add(pair)
    if any(len(s) != 1 for s in scenes.values()):
        raise ValueError('Scene leakage')
    if not added or any(not any(r['split'] == s for r in added) for s in ('train','val','test')):
        raise ValueError('New corpus must cover all splits')
    out.mkdir(parents=True)
    manifest = out/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    for split in ('val', 'test'):
        (out/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['split']==split), encoding='utf-8')
    report = dict(original_manifest_sha256=digest(original), source_index_sha256=digest(index),
                  ingest_sha256=digest(moves/'pairs_moves'/'_ingest_config.json'),
                  manifest_sha256=digest(manifest), records=dict(Counter(r['split'] for r in rows)),
                  added_records=dict(Counter(r['split'] for r in added)),
                  scenes=dict(Counter(next(iter(v)) for v in scenes.values())),
                  excluded_existing_families=sorted(excluded),
                  license='NON-COMMERCIAL: original corpus and v5 initialization restrictions remain',
                  scope='Rendered static panoramas plus original corpus; not independent real-world SDR validation')
    (out/'provenance.json').write_text(json.dumps(report,indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'excluded_existing_families'},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--moves', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.original, a.moves, a.out)
