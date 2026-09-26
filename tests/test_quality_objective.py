import torch
from training.quality_objective import joint_utility, balanced_loss


def test_conflicting_metric_gain_is_penalized_and_baseline_is_safe():
    pu=torch.tensor([[40.,42.,41.,40.],[40.,42.,41.,40.]])
    jod=torch.tensor([[9.,8.8,9.1,9.],[9.,8.8,9.1,9.]])
    conditions=torch.tensor([False,True])
    u=joint_utility(pu,jod,conditions)
    assert (u[:,1]<0).all() and (u[:,2]>0).all()
    assert (u[:,3]==0).all()
    logits=torch.zeros_like(u,requires_grad=True)
    balanced_loss(logits,u,conditions).backward()
    assert (logits.grad[:,1]>0).all()
    assert (logits.grad[:,2]<0).all()


def test_group_balance_ignores_duplicated_clean_samples():
    logits=torch.zeros(2,4)
    utility=torch.tensor([[0.,1.,0.,0.],[-2.,0.,-2.,0.]])
    conditions=torch.tensor([False,True])
    idx=torch.tensor([0,0,0,0,1])
    assert torch.allclose(balanced_loss(logits,utility,conditions),
                          balanced_loss(logits[idx],utility[idx],conditions[idx]))
