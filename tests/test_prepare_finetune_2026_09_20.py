import hashlib
import json

import pytest
from training.prepare_finetune import prepare


def test_frozen_splits_exclude_old_holdouts_and_keep_train_views(tmp_path):
    moves, out = tmp_path/'moves', tmp_path/'frozen'
    for folder in ('panoramas','pairs_moves/meta','pairs_moves/sdr','pairs_moves/hdr'):
        (moves/folder).mkdir(parents=True)
    original = []
    for family, split in (('existing_train','train'),('existing_val','val'),('existing_test','test')):
        path = tmp_path/f'{family}.png'
        path.write_bytes(b'fixture')
        original.append(dict(asset_id=family,scene_id=family,split=split,sdr_path=str(path),hdr_path=str(path)))
    source_manifest = tmp_path/'original.jsonl'
    source_manifest.write_text(''.join(json.dumps(r)+'\n' for r in original))
    selected = {}
    for i in range(1000):
        name = f'new{i}scene'
        bucket = int(hashlib.sha256(('rudra-20260920:'+name).encode()).hexdigest()[:8],16)%100
        split = 'train' if bucket<80 else ('val' if bucket<90 else 'test')
        selected.setdefault(split,name)
        if len(selected)==3: break
    index = []
    for name in ['existing_train','existing_val','existing_test',*selected.values()]:
        filename = name+'_4k.exr'
        index.append(dict(file=filename,slug=name,license='CC0',url='https://example.invalid/'+filename))
        (moves/'pairs_moves/meta'/f'{name}.json').write_text(json.dumps(dict(frame=0,stem=name,source=filename)))
        for folder in ('sdr','hdr'): (moves/'pairs_moves'/folder/f'{name}.png').write_bytes(b'fixture')
    (moves/'panoramas/_polyhaven_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index))
    (moves/'pairs_moves/_ingest_config.json').write_text('{}')
    prepare(source_manifest,moves,out)
    rows = [json.loads(line) for line in (out/'manifest.jsonl').read_text().splitlines()]
    added = {r['asset_id']:r for r in rows if r['asset_id'].startswith('moves_')}
    assert added['moves_existing_train']['scene_id']=='existing_train'
    assert 'moves_existing_val' not in added and 'moves_existing_test' not in added
    assert rows[:3]==original
    with pytest.raises(ValueError,match='frozen dataset'):
        prepare(source_manifest,moves,out)
