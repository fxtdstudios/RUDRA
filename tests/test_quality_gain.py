import torch
import pytest
from rudra.quality_gain import degradation_features,gain_decisions


def test_native_features_detect_high_frequency_corruption():
    clean=torch.full((1,3,32,32),.5)
    noisy=clean.clone(); noisy[:,:,:,::2]+=.1; noisy[:,:,:,1::2]-=.1
    a=degradation_features(clean); b=degradation_features(noisy)
    assert a.shape==b.shape==(1,20)
    assert torch.isfinite(a).all()
    assert b[0,4]>a[0,4]  # horizontal gradient
    assert b[0,8]>a[0,8]  # local residual
    assert a[0,2]==pytest.approx(.5)


def test_gain_selection_rejects_conflicting_metrics_and_nonfinite_values():
    gains=torch.tensor([[[2.,-.1],[.3,.4],[.1,.1],[0.,0.]],
                        [[1.,-1.],[-1.,1.],[-1.,-1.],[0.,0.]]])
    assert gain_decisions(gains,.2).tolist()==[1,3]
    assert gain_decisions(gains,.5).tolist()==[3,3]
    gains[0,0,0]=float('nan')
    assert gain_decisions(gains,0).tolist()==[3,3]


def test_feature_rejects_invalid_input():
    with pytest.raises(ValueError): degradation_features(torch.zeros(1,3,4,4))
    x=torch.zeros(1,3,9,9); x[0,0,0,0]=float('nan')
    with pytest.raises(ValueError): degradation_features(x)
