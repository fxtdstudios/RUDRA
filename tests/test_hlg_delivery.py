import numpy as np
import pytest

from rudra.delivery.profiles import hlg_oetf, hlg_eotf, encode_master


def test_hlg_reference_points_and_display_white():
    np.testing.assert_allclose(hlg_oetf(np.array([0,1/12,1])),[0,.5,1],atol=1e-7)
    # BT.2100 HLG 75% achromatic code is about 203 cd/m2 on a 1000-nit display.
    assert hlg_eotf(np.full((1,1,3),.75))[0,0,0]==pytest.approx(203.15,abs=.1)
    np.testing.assert_allclose(hlg_eotf(np.ones((1,1,3))),1000,atol=.001)


def test_hlg_conversion_recovers_neutral_display_light_below_knee():
    values=np.array([0,.1,1,10,100,203,500],dtype=np.float32)
    rgb=np.repeat(values[:,None,None],3,axis=2)/10000
    code,mastered=encode_master(rgb,'hlg',1000)
    np.testing.assert_allclose(hlg_eotf(code),rgb*10000,atol=.002,rtol=1e-5)
    np.testing.assert_allclose(mastered,rgb*10000,atol=.002)


def test_saturated_hlg_fits_code_range_without_channel_clipping():
    code,mastered=encode_master(np.array([[[.1,.02,.001]]]),'hlg',1000)
    assert np.isfinite(code).all() and code.min()>=0 and code.max()<=1
    assert mastered[0,0,0]/mastered[0,0,1]==pytest.approx(5,rel=1e-5)
