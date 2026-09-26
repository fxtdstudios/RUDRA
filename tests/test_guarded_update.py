import copy
import pytest
import torch
from training.guarded_update import guarded_step, preserves


def test_rejection_restores_adam_momentum_and_model():
    model=torch.nn.Linear(1,1)
    opt=torch.optim.AdamW(model.parameters(),lr=.1)
    model(torch.ones(1,1)).sum().backward(); opt.step(); opt.zero_grad()
    weights=copy.deepcopy(model.state_dict()); state=copy.deepcopy(opt.state_dict())
    model(torch.ones(1,1)).sum().backward()
    assert not guarded_step(model,opt,lambda:False)
    assert all(torch.equal(v,weights[k]) for k,v in model.state_dict().items())
    for k,v in state['state'].items():
        for name,tensor in v.items():
            assert torch.equal(tensor,opt.state_dict()['state'][k][name])
    assert all(p.grad is None for p in model.parameters())


def test_failed_check_rolls_back_initial_adam_state():
    model=torch.nn.Linear(1,1); opt=torch.optim.AdamW(model.parameters())
    before=copy.deepcopy(model.state_dict())
    model(torch.ones(1,1)).sum().backward()
    def fail(): raise RuntimeError('check failed')
    with pytest.raises(RuntimeError,match='check failed'): guarded_step(model,opt,fail)
    assert not opt.state
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_accepted_update_and_component_guard():
    model=torch.nn.Linear(1,1); opt=torch.optim.AdamW(model.parameters())
    before=model.weight.detach().clone()
    model(torch.ones(1,1)).sum().backward()
    assert guarded_step(model,opt,lambda:True)
    assert not torch.equal(before,model.weight)
    assert not preserves(torch.tensor([.9,1.1]),torch.ones(2),torch.ones(2))
    assert not preserves(torch.tensor([float('nan')]),torch.ones(1),torch.ones(1))
    assert preserves(torch.tensor([.9]),torch.ones(1),torch.ones(1))
