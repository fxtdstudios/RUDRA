import io
import numpy as np
import pytest
from PIL import Image, ImageCms
from ui.ocio_support import decode_input,input_pixels


def encoded(image,profile=None,format='PNG'):
    stream=io.BytesIO()
    image.save(stream,format=format,**({'icc_profile':profile} if profile else {}))
    return stream.getvalue()


def test_untagged_reports_assumption():
    result,info=decode_input(encoded(Image.new('RGB',(2,2),(20,40,60))),dict(input_auto=True))
    assert info['assumed'] and info['profile']=='sRGB'
    assert result.getpixel((0,0))==(20,40,60)


def test_embedded_lab_profile_is_converted_before_model():
    profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB'))
    image=Image.new('LAB',(2,2),(128,145,115))
    raw=encoded(image,profile.tobytes(),'TIFF')
    result,info=decode_input(raw,dict(input_auto=True))
    expected=ImageCms.profileToProfile(image,profile,ImageCms.createProfile('sRGB'),outputMode='RGB')
    np.testing.assert_array_equal(np.asarray(result),np.asarray(expected))
    assert not info['assumed'] and info['mode']=='auto'
    pixels=np.asarray(result,dtype=np.float32)/255
    # ICC output is already sRGB: a manually selected OCIO space must not be applied twice.
    assert input_pixels(pixels,dict(input_auto=True,ocio={'input':'invalid'})) is pixels


def test_invalid_profile_requires_explicit_override():
    raw=encoded(Image.new('RGB',(2,2)),b'broken profile')
    with pytest.raises(ValueError,match='Embedded ICC'):
        decode_input(raw,dict(input_auto=True))
    _,info=decode_input(raw,dict(input_auto=False))
    assert info['mode']=='manual'


@pytest.mark.parametrize('format',['PNG','TIFF'])
@pytest.mark.parametrize('automatic',[True,False])
def test_high_bit_depth_not_silently_truncated(format,automatic):
    image=Image.fromarray(np.array([[1000,1001],[40000,40001]],dtype=np.uint16))
    decoded, info = decode_input(encoded(image,format=format),dict(input_auto=automatic))
    expected=np.asarray(image,dtype=np.float32)/65535
    np.testing.assert_allclose(np.asarray(decoded)[...,0]/255,expected,atol=1e-7)
    assert info['bit_depth']==16
    assert np.asarray(decoded)[0,0,0] != np.asarray(decoded)[0,1,0]


def test_rgb16_png_detected_despite_pillow_rgb_mode():
    import cv2
    rgb=np.tile(np.array([1000,20001,40002],dtype=np.uint16),(4,4,1))
    ok,data=cv2.imencode('.png',rgb)
    assert ok
    decoded,_=decode_input(data.tobytes(),dict(input_auto=True))
    np.testing.assert_allclose(np.asarray(decoded)/255,rgb[...,::-1]/65535,atol=1e-7)
    resized=decoded.resize((2,2),Image.Resampling.LANCZOS)
    np.testing.assert_allclose(np.asarray(resized)[0,0]/255,rgb[0,0,::-1]/65535,atol=1e-7)


def test_16bit_icc_requires_manual_override():
    image=Image.fromarray(np.full((2,2),12345,dtype=np.uint16))
    raw=encoded(image,ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes(),'PNG')
    with pytest.raises(ValueError,match='16-bit embedded ICC'):
        decode_input(raw,dict(input_auto=True))
    decoded,_=decode_input(raw,dict(input_auto=False))
    assert np.asarray(decoded)[0,0,0]/255 == pytest.approx(12345/65535,abs=1e-7)


@pytest.mark.parametrize('entry',['inference','frame','master'])
def test_16bit_reaches_model_without_8bit_rounding(entry,tmp_path,monkeypatch):
    import cv2
    import torch
    from types import SimpleNamespace
    from ui import server
    from training import infer_sdr2hdr
    torch.set_num_threads(4)
    model=infer_sdr2hdr.load_models('checkpoints/sdr2hdr_shadow_v1.pt',None,torch.device('cpu'))[0]
    pixels=np.full((32,32,3),20001,dtype=np.uint16)
    pixels[:,16:]=20002
    ok,raw=cv2.imencode('.png',pixels)
    assert ok
    captured=[]
    name='predict_fields' if entry=='frame' else 'predict_image'
    original=getattr(infer_sdr2hdr,name)
    def observe(model,tensor,*args,**kwargs):
        captured.append(tensor.detach().cpu().numpy().copy())
        return original(model,tensor,*args,**kwargs)
    monkeypatch.setattr(infer_sdr2hdr,name,observe)
    params=dict(input_auto=True,tile_size=0,render_dir=str(tmp_path),container='linear')
    call={'inference':server.run_inference,'frame':server.run_frame,'master':server.run_master}[entry]
    result=call(model,raw.tobytes(),params,SimpleNamespace())
    if entry=='frame':
        header,body=result
        assert header['sdr_dtype']=='float32'
        transmitted=np.frombuffer(body,dtype='<f4',offset=header['offsets']['sdr']).reshape(32,32,3)
        np.testing.assert_allclose(transmitted,pixels/65535,atol=1e-7)
    assert captured
    np.testing.assert_allclose(captured[0][0,0],pixels[...,0]/65535,atol=1e-7)
    assert captured[0][0,0,0,0] != captured[0][0,0,0,16]
