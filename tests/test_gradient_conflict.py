import pytest
import torch
from training.diagnose_gradient_conflict import geometry


def test_conflicting_direction_predicts_increasing_clean_risk():
    clean=[torch.tensor([1.,2.]),torch.tensor([3.])]
    hard=[-x for x in clean]
    result=geometry(clean,hard)
    assert result['cosine']==pytest.approx(-1)
    assert result['dot']==pytest.approx(-14)
    assert geometry(clean,clean)['cosine']==pytest.approx(1)


def test_zero_gradient_has_finite_geometry():
    assert geometry([torch.zeros(3)],[torch.ones(3)])['cosine']==0
