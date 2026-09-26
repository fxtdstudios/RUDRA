"""Experimental, checkpoint-bound recovery selection from SDR-only features."""
import torch
from torch import nn
from torch.nn import functional as F
from .sdr2hdr import frame_conditioning_stats, luminance, sdr_to_baseline_hdr

MODES = ('off', 'highlights', 'shadows', 'all')


@torch.no_grad()
def features(backbone, sdr):
    """Global pooled features, with degradation statistics from native pixels."""
    view = sdr
    longest = max(sdr.shape[-2:])
    if longest > 512:
        view = F.interpolate(sdr, size=tuple(max(1, round(n*512/longest)) for n in sdr.shape[-2:]), mode='area')
    _, _, mid = backbone.encode(view, sdr_to_baseline_hdr(view))
    return torch.cat((mid.mean((2,3)), mid.amax((2,3)), frame_conditioning_stats(sdr, luminance(sdr))), dim=1)


class RecoveryPolicy(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dimensions))
        self.register_buffer('scale', torch.ones(dimensions))
        self.net = nn.Sequential(nn.Linear(dimensions,64), nn.SiLU(), nn.Linear(64,4))
        self.confidence_threshold = 0.0

    def forward(self, x):
        return self.net((x-self.mean)/self.scale)

    @torch.no_grad()
    def decisions(self, x):
        confidence, choices = self(x).softmax(1).max(1)
        return torch.where(confidence >= self.confidence_threshold, choices, 3)

    @torch.no_grad()
    def choose(self, backbone, sdr):
        if sdr.shape[0] != 1: raise ValueError('One frame per policy decision')
        return MODES[int(self.decisions(features(backbone,sdr)).item())]


def regret_loss(logits, quality_db):
    """Expected measured PU21 regret: equally good actions need no arbitrary label."""
    regret = quality_db.max(1,keepdim=True).values-quality_db
    return (logits.softmax(1)*regret).sum(1).mean()


def load_policy(path, backbone_path, device='cpu'):
    from .batch import digest
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['backbone_sha256'] != digest(backbone_path):
        raise ValueError('Recovery policy belongs to a different reconstruction checkpoint')
    policy=RecoveryPolicy(payload['dimensions'])
    policy.confidence_threshold=float(payload.get('confidence_threshold',0))
    if not 0 <= policy.confidence_threshold <= 1:
        raise ValueError('Invalid recovery confidence threshold')
    policy.load_state_dict(payload['model'],strict=True)
    return policy.to(device).eval()
