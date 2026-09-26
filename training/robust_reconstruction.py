"""Paired reconstruction objective; surrogate losses do not certify PU21/JOD gains."""
import torch
from rudra.sdr2hdr import censored_log_error


def reconstruction_risk(pred, target, ceiling):
    error, _, _, _ = censored_log_error(pred.float(), target.float(), ceiling)
    return error.flatten(1).mean(1)


def paired_objective(predictions, teachers, target, ceiling, harm_weight=4.0):
    """Equal clean/hard supervision plus per-image teacher-relative harm penalty.

    Unlike batch-average gains, each damaged frame contributes a penalty even
    when another frame improves. Teachers are detached; censoring is retained.
    """
    risks = torch.stack([reconstruction_risk(p, target, ceiling) for p in predictions])
    reference = torch.stack([reconstruction_risk(p.detach(), target, ceiling) for p in teachers])
    harm = (risks - reference).relu().mean()
    groups = risks.mean(1)
    loss = groups.mean() + harm_weight * harm
    return loss, dict(clean_risk=groups[0], hard_risk=groups[1], harm=harm)


def pu_encode(value):
    """Differentiable PU21 banding+glare, nits input; scaled by 100 for training."""
    y = value.float().clamp(.005, 10000).pow(.9062562627)
    return 5.963148142 * (((.353487901 + .3734658629*y) /
                          (1 + 8.277049286e-05*y)).pow(.09150303166) - .9099517204)


def region_pu_risk(pred, target, ceiling):
    """Per-image, per-region errors; saturated reference pixels are lower bounds."""
    target = target.float()
    pred = pred.float()
    encoded_pred, encoded_target = pu_encode(pred*10000), pu_encode(target*10000)
    error = (encoded_pred-encoded_target).square()
    if ceiling is not None:
        from rudra.sdr2hdr import CENSORED_HEADROOM_STOPS
        ceiling = torch.as_tensor(ceiling, device=target.device, dtype=target.dtype).reshape(-1,1,1,1)
        censored = (target >= ceiling*(1-1e-3)) & torch.isfinite(ceiling)
        allowance = pu_encode(ceiling*(2**CENSORED_HEADROOM_STOPS)*10000)
        bounded = (encoded_target-encoded_pred).relu().square() + (encoded_pred-allowance).relu().square()
        error = torch.where(censored,bounded,error)
    error = error.mean(1)
    luma = (target*target.new_tensor([.2627,.678,.0593])[None,:,None,None]).sum(1)*10000
    masks = torch.stack([luma<12, (luma>=12)&(luma<203), luma>=203],1)
    counts = masks.flatten(2).sum(2)
    risks = (error[:,None]*masks).flatten(2).sum(2)/counts.clamp_min(1)
    return risks, counts>0


def region_paired_objective(predictions, teachers, target, ceiling, harm_weight=4.0):
    """Experimental region preservation; PU loss does not guarantee CVVDP gains."""
    base, parts = paired_objective(predictions, teachers, target, ceiling, harm_weight)
    risks = torch.stack([region_pu_risk(p,target,ceiling)[0] for p in predictions])
    reference = torch.stack([region_pu_risk(p.detach(),target,ceiling)[0] for p in teachers])
    present = region_pu_risk(target,target,ceiling)[1][None].expand_as(risks)
    # Penalize each damaged region before averaging across images or conditions.
    harm = ((risks-reference).relu()*present).sum()/present.sum().clamp_min(1)
    groups = (risks*present).sum((1,2))/present.sum((1,2)).clamp_min(1)
    loss = base + groups.max() + harm_weight*harm
    return loss, dict(**parts, region_clean=groups[0], region_hard=groups[1], region_harm=harm)
