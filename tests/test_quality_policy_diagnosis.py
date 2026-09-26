import pytest
from training.diagnose_quality_policy import group_stats


def test_diagnosis_separates_policy_error_from_metric_disagreement():
    records = [dict(pu21_db=[1, 5, 2, 4], cvvdp_jod=[1, 3, 2, 4]),
               dict(pu21_db=[1, 2, 3, 4], cvvdp_jod=[1, 2, 3, 4])]
    result = group_stats(records, [1, 1])
    assert result['policy_pu21_delta'] == -.5
    assert result['oracle_pu21_delta'] == .5
    assert result['mean_pu21_regret'] == 1
    assert result['pu21_regressed_frames'] == 1
    assert result['jod_regressed_frames'] == 2
    assert result['pu21_oracle_jod_regressed_frames'] == 1
    assert result['policy_modes'] == {'highlights': 2}
    assert result['oracle_modes'] == {'highlights': 1, 'all': 1}


def test_training_diagnosis_does_not_invent_unmeasured_perceptual_scores():
    result = group_stats([dict(pu21_db=[1, 2, 3, 4], cvvdp_jod=[])], [3])
    assert result['mean_pu21_regret'] == 0
    assert 'policy_jod_delta' not in result
    with pytest.raises(ValueError):
        group_stats([dict(pu21_db=[1, 2, 3, 4])], [])
