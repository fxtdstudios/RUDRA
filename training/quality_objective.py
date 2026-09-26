"""Experimental two-metric objective; fit scales from training samples only."""
import torch


def joint_utility(pu, jod, conditions):
    scores = torch.stack((pu, jod), dim=-1)
    delta = scores-scores[:, 3:4, :]
    utility = torch.empty_like(pu)
    for condition in conditions.unique():
        mask = conditions == condition
        scale = torch.quantile(delta[mask].abs().reshape(-1, 2), .75, dim=0)
        scale = torch.maximum(scale, scale.new_tensor([1., .05]))
        # Any regression is negative, even if the other metric improves.
        utility[mask] = (delta[mask]/scale).amin(-1)
    return utility


def balanced_loss(logits, utility, conditions):
    regret = utility.max(1, keepdim=True).values-utility
    expected = (logits.softmax(1)*regret).sum(1)
    groups = torch.stack([expected[conditions == c].mean() for c in conditions.unique()])
    return groups.max() + .1*groups.mean()
