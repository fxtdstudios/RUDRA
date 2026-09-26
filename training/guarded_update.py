"""Transactional optimizer proposal for experiments, not a release quality gate."""
import copy
import torch


def guarded_step(model, optimizer, accept):
    """Restore parameters, buffers and optimizer state on rejection or failure."""
    weights=copy.deepcopy(model.state_dict())
    state=copy.deepcopy(optimizer.state_dict())
    accepted=False
    try:
        optimizer.step()
        accepted=bool(accept())
        return accepted
    finally:
        if not accepted:
            model.load_state_dict(weights)
            optimizer.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)


def preserves(after, before, reference, tolerance=1e-7):
    """All surrogate components must preserve both prior state and frozen teacher."""
    return bool(torch.isfinite(after).all() and
                (after <= before+tolerance).all() and
                (after <= reference+tolerance).all())
