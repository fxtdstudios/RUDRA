import numpy as np
import torch
from rudra.delivery.bench import pu21_encode
from training.robust_reconstruction import pu_encode, region_pu_risk, region_paired_objective


def test_torch_pu_matches_existing_benchmark_and_has_finite_gradient():
    x=torch.logspace(-3,5,100,requires_grad=True)
    encoded=pu_encode(x)
    np.testing.assert_allclose(encoded.detach().numpy(),pu21_encode(x.detach().numpy())/100,atol=2e-5)
    encoded.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_highlight_harm_cannot_cancel_against_shadow_improvement():
    target=torch.tensor([.0005,.05]).reshape(1,1,1,2).expand(1,3,1,2)
    teacher=target.clone(); teacher[...,0]*=2
    teacher.requires_grad_()
    pred=target.clone(); pred[...,1]*=2; pred.requires_grad_()
    loss,parts=region_paired_objective([pred,pred],[teacher,teacher],target,None)
    assert parts['region_harm']>0
    loss.backward()
    assert teacher.grad is None and pred.grad[...,1].sum()>0
    assert torch.isfinite(pred.grad).all()


def test_empty_regions_and_censored_targets():
    target=torch.full((2,3,2,2),.1)
    risk,present=region_pu_risk(target*1.2,target,torch.tensor([.1,.1]))
    assert torch.equal(risk,torch.zeros_like(risk))
    assert present.sum()==2
    assert region_pu_risk(target*.8,target,torch.tensor([.1,.1]))[0].sum()>0
    from rudra.sdr2hdr import CENSORED_HEADROOM_STOPS
    low_target=target*.01
    assert region_pu_risk(low_target*(2**(CENSORED_HEADROOM_STOPS+1)),low_target,torch.tensor([.001,.001]))[0].sum()>0
    # Existing censoring tolerance survives the HDR storage round trip.
    assert region_pu_risk(target*1.2,target*.9995,torch.tensor([.1,.1]))[0].sum()==0
