import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ocio=pytest.importorskip('PyOpenColorIO')
from ui.ocio_support import load_config,describe,input_pixels,output_pixels


def settings():
    return dict(config='',cache_id=describe()['cache_id'],input='sRGB Encoded Rec.709 (sRGB)',
        model_srgb='sRGB Encoded Rec.709 (sRGB)',working_linear='Linear Rec.2020',
        output='ACEScg',display='sRGB - Display',view='ACES 1.0 - SDR Video')


def test_identity_input_and_aces_neutral():
    s=settings(); rgb=np.full((2,3,3),.18,dtype=np.float32)
    np.testing.assert_allclose(input_pixels(rgb,dict(ocio=s)),rgb,atol=1e-6)
    result,meta=output_pixels(rgb,dict(ocio=s,container='ocio_exr'))
    np.testing.assert_allclose(result,rgb,atol=2e-5)
    assert meta['output']=='ACEScg'


def test_actual_input_conversion_and_invalid_config():
    s=settings(); s['input']='Linear Rec.709 (sRGB)'
    result=input_pixels(np.full((1,1,3),.18),dict(ocio=s))
    np.testing.assert_allclose(result,.461356,atol=2e-5)
    s['cache_id']='changed'
    with pytest.raises(ValueError,match='changed'):
        input_pixels(np.ones((1,1,3)),dict(ocio=s))


def test_local_config_load(tmp_path):
    path=tmp_path/'config.ocio'; path.write_text(load_config().serialize())
    assert 'ACEScg' in describe(str(path))['spaces']


def test_missing_optional_dependency_is_actionable(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules,'PyOpenColorIO',None)
    with pytest.raises(ValueError,match='optional'):
        load_config()
    pixels=np.zeros((1,1,3))
    assert input_pixels(pixels,{}) is pixels


def test_http_preview_matches_export_pixels(tmp_path,monkeypatch):
    import threading
    import urllib.request
    import torch
    from PIL import Image
    from ui import server
    from training.infer_sdr2hdr import load_models
    torch.set_num_threads(4)
    model=load_models('checkpoints/sdr2hdr_shadow_v1.pt',None,torch.device('cpu'))[0]
    monkeypatch.setattr(server,'ensure_model',lambda args:(model,dict(loaded=True)))
    raw=io.BytesIO(); Image.new('RGB',(32,32),(200,100,50)).save(raw,format='PNG')
    params=dict(ocio=settings(),container='ocio_view_png',render_dir=str(tmp_path))
    result=server.run_master(model,raw.getvalue(),params,SimpleNamespace())
    http=server.ThreadedServer(('127.0.0.1',0),server.make_handler(SimpleNamespace()))
    thread=threading.Thread(target=http.serve_forever,daemon=True); thread.start()
    try:
        request=urllib.request.Request(f'http://127.0.0.1:{http.server_address[1]}/api/master/preview',
            data=raw.getvalue(),headers={'X-Rudra-Params':json.dumps(params)})
        with urllib.request.urlopen(request) as response:
            preview=np.asarray(Image.open(io.BytesIO(response.read())))
        np.testing.assert_array_equal(preview,np.asarray(Image.open(result['path'])))
    finally:
        http.shutdown(); http.server_close(); thread.join()


@pytest.mark.parametrize('preset',['ocio_exr','ocio_view_png'])
def test_real_export_ocio_pixels_and_metadata(tmp_path,preset):
    import torch
    from PIL import Image
    from training.infer_sdr2hdr import load_models,predict_image
    from ui.server import run_master
    from rudra.delivery.exr import read_exr
    torch.set_num_threads(4)
    model=load_models('checkpoints/sdr2hdr_shadow_v1.pt',None,torch.device('cpu'))[0]
    image=Image.new('RGB',(32,32),(100,120,140)); raw=io.BytesIO();image.save(raw,format='PNG')
    params=dict(ocio=settings(),container=preset,render_dir=str(tmp_path))
    result=run_master(model,raw.getvalue(),params,SimpleNamespace())
    sdr=input_pixels(np.asarray(image,dtype=np.float32)/255,params)
    hdr=predict_image(model,torch.from_numpy(sdr).permute(2,0,1)[None],True,0,64)[0].permute(1,2,0).numpy()*10000/203
    expected,_=output_pixels(hdr,params)
    meta=json.loads(Path(result['sidecar']).read_text())
    assert meta['ocio']['cache_id']==settings()['cache_id']
    assert meta['transfer']=='OCIO-defined'
    if preset=='ocio_exr':
        actual,_=read_exr(result['path']); np.testing.assert_allclose(actual,expected,atol=1e-5)
        assert meta['bit_depth']==32
    else:
        actual=np.asarray(Image.open(result['path']))
        np.testing.assert_array_equal(actual,np.rint(np.clip(expected,0,1)*255).astype(np.uint8))
        assert meta['bit_depth']==8
