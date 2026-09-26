import pytest
from training.check_reference_bank import select_groups


def test_bank_groups_exclude_nontraining_prior_scenes_and_duplicates():
    rows=[dict(scene_id=str(i),split='train') for i in range(60)]
    rows+=rows[:10]+[dict(scene_id='validation',split='val')]
    groups=select_groups(rows,{'0','1'})
    ids=[rows[i]['scene_id'] for group in groups for i in group]
    assert [len(g) for g in groups]==[24,12,8]
    assert len(set(ids))==44 and not set(ids)&{'0','1','validation'}
    assert groups==select_groups(rows,{'0','1'})


def test_insufficient_scenes_fail_instead_of_reusing_bank():
    with pytest.raises(ValueError,match='44'):
        select_groups([dict(scene_id='a',split='train')]*50,set())
