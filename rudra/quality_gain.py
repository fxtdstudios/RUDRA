"""Experimental SDR degradation descriptors and conservative gain decisions."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def degradation_features(sdr):
    if sdr.ndim!=4 or sdr.shape[1]!=3 or min(sdr.shape[-2:])<9:
        raise ValueError('Expected RGB frames at least 9x9')
    if not torch.isfinite(sdr).all(): raise ValueError('Non-finite SDR')
    y=.2627*sdr[:,0:1]+.678*sdr[:,1:2]+.0593*sdr[:,2:3]
    dx=(y[:,:,:,1:]-y[:,:,:,:-1]).abs()
    dy=(y[:,:,1:,:]-y[:,:,:-1,:]).abs()
    smooth=F.avg_pool2d(y,3,stride=1,padding=1)
    residual=(y[:,:,1:-1,1:-1]-smooth[:,:,1:-1,1:-1]).abs()
    out=[(sdr<.01).float().mean((1,2,3)),(sdr>.99).float().mean((1,2,3))]
    for v in (y,dx,dy,residual):
        out.extend([v.mean((1,2,3)),v.square().mean((1,2,3)).sqrt()])
    # Pixel boundaries retain native alignment; useful evidence, not a JPEG detector.
    for v in (dx,dy.transpose(2,3)):
        out.extend([v[:,:,:,7::8].mean((1,2,3)),v[:,:,:,3::8].mean((1,2,3))])
    for c in range(3):
        difference=sdr[:,c:c+1]-y
        out.extend([difference.mean((1,2,3)),difference.square().mean((1,2,3)).sqrt()])
    return torch.stack(out,1)


def gain_decisions(predicted,margin):
    """Predicted baseline-relative gains: N x 4 modes x 2 scaled metrics."""
    if predicted.ndim!=3 or predicted.shape[1:]!=(4,2): raise ValueError('Expected N x 4 x 2 gains')
    benefit=predicted.amin(-1)
    benefit=benefit.clone(); benefit[:,3]=0
    best,choice=benefit.max(1)
    return torch.where(torch.isfinite(predicted).all((1,2)) & (best>margin),choice,3)
