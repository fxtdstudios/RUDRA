import pytest
import torch
from training.robust_reconstruction import paired_objective, reconstruction_risk
from training.train_robust_reconstruction import check_splits


def test_per_frame_harm_cannot_cancel_against_improvement():
    target = torch.zeros(2,3,2,2)
    teacher = torch.full_like(target,.1,requires_grad=True)
    pred = torch.stack([torch.zeros(3,2,2),torch.full((3,2,2),.3)]).requires_grad_()
    loss, parts = paired_objective([pred,pred],[teacher,teacher],target,None)
    assert parts['harm'] > 0
    loss.backward()
    assert teacher.grad is None
    assert pred.grad[1].sum() > 0


def test_ceiling_headroom_is_not_punished_as_exact_target():
    target = torch.full((1,3,2,2),.1)
    pred = target*1.1
    assert reconstruction_risk(pred,target,torch.tensor([.1])).item() == 0
    assert reconstruction_risk(pred,target,None).item() > 0


def test_split_leakage_rejected_before_loading_pixels():
    with pytest.raises(ValueError,match='leakage'):
        check_splits([dict(scene_id='a',split='train'),dict(scene_id='a',split='test')])
