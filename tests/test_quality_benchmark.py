import pytest

from training.quality_benchmark import select_scenes, paired_summary, capture_environment
import json


def test_selection_is_order_independent_and_validation_only():
    rows=[dict(scene_id=s,asset_id=s+str(i),split='val') for s in ('a','b','c') for i in range(3)]
    rows.append(dict(scene_id='training',asset_id='train',split='train'))
    selected=select_scenes(rows,2)
    assert selected==select_scenes(list(reversed(rows)),2)
    assert len({r['scene_id'] for r in selected})==2
    assert all(r['split']=='val' for r in selected)


def test_selection_rejects_scene_leakage():
    with pytest.raises(ValueError,match='across'):
        select_scenes([dict(scene_id='same',asset_id='a',split='val'),
                       dict(scene_id='same',asset_id='b',split='test')],1)


def test_paired_summary_drops_unpaired_values():
    rows=[dict(asset_id=str(i),method=m,score=float(i+gain))
          for i in range(3) for m,gain in [('baseline',0),('shipped',2)]]
    rows.extend([dict(asset_id='missing',method='baseline',score=999),
                 dict(asset_id='missing',method='shipped',score=None)])
    result=paired_summary(rows,'score')
    assert result['pairs']==3
    assert result['delta']==pytest.approx(2)
    assert result['ci95']==pytest.approx([2,2])
    assert result['wins']==3


def test_no_valid_pairs():
    assert paired_summary([], 'score')=={'pairs':0}


def test_environment_is_saved_for_followup_ablation(tmp_path):
    capture_environment(tmp_path)
    data=json.loads((tmp_path/'environment.json').read_text())
    assert 'numpy' in data['versions']
    assert len(data['code_sha256']['rudra/sdr2hdr.py'])==64
