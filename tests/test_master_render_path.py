import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ui'))
import server


def test_sequence_names_and_preflight(tmp_path):
    params=dict(render_dir=str(tmp_path),render_name='shot',render_mode='sequence',render_count=3,frame_start=1001)
    assert [p.name for p in server.master_targets(params)]==['shot.001001.exr','shot.001002.exr','shot.001003.exr']
    (tmp_path/'shot.001003.json').write_text('existing')
    with pytest.raises(ValueError,match='overwrite'): server.master_targets(params)


@pytest.mark.parametrize('name', ['../escape','a/b','a\\b','', '..'])
def test_render_name_cannot_escape_folder(tmp_path,name):
    with pytest.raises(ValueError): server.master_targets(dict(render_dir=str(tmp_path),render_name=name))


def test_folder_is_required():
    with pytest.raises(ValueError,match='absolute'): server.master_targets({})


def test_render_failure_publishes_nothing(tmp_path,monkeypatch):
    def fail(model,raw,params,args,out):
        out.write_bytes(b'partial')
        raise RuntimeError('failed render')
    monkeypatch.setattr(server,'_render_master',fail)
    with pytest.raises(RuntimeError): server.run_master(None,b'',dict(render_dir=str(tmp_path)),None)
    assert list(tmp_path.iterdir())==[]


def test_render_publish_race_preserves_existing_file(tmp_path,monkeypatch):
    def render(model,raw,params,args,out):
        out.write_bytes(b'new')
        out.with_suffix('.json').write_text('{}')
        (tmp_path/'master.exr').write_bytes(b'other render')
        return {}
    monkeypatch.setattr(server,'_render_master',render)
    with pytest.raises(FileExistsError): server.run_master(None,b'',dict(render_dir=str(tmp_path)),None)
    assert (tmp_path/'master.exr').read_bytes()==b'other render'


def test_real_image_renders_to_chosen_path(tmp_path):
    import torch
    from PIL import Image
    from training.infer_sdr2hdr import load_models
    from rudra.delivery.exr import read_exr
    torch.set_num_threads(4)
    model=load_models('checkpoints/sdr2hdr_shadow_v1.pt',None,torch.device('cpu'))[0]
    buf=io.BytesIO(); Image.new('RGB',(64,48),(100,120,140)).save(buf,format='PNG')
    result=server.run_master(model,buf.getvalue(),dict(render_dir=str(tmp_path),render_name='image',container='linear'),SimpleNamespace())
    assert result['path']==str(tmp_path/'image.exr')
    assert Path(result['sidecar']).is_file()
    data,_=read_exr(tmp_path/'image.exr')
    assert data.shape==(48,64,3)
    with pytest.raises(ValueError,match='overwrite'):
        server.run_master(model,buf.getvalue(),dict(render_dir=str(tmp_path),render_name='image'),SimpleNamespace())


@pytest.mark.parametrize('container',['linear','aces'])
def test_real_sequence_keeps_numbering_pixels_grade_and_metadata(tmp_path,container):
    import numpy as np
    import torch
    from PIL import Image
    from training.infer_sdr2hdr import load_models,predict_image
    from rudra.delivery.exr import read_exr
    from rudra.delivery.colorspace import convert,AP0_CHROMATICITIES,REC2020_CHROMATICITIES
    torch.set_num_threads(4)
    model=load_models('checkpoints/sdr2hdr_shadow_v1.pt',None,torch.device('cpu'))[0]
    params=dict(render_dir=str(tmp_path),render_name='shot',render_mode='sequence',
                render_count=2,frame_start=1001,container=container,
                checkpoint='checkpoints/sdr2hdr_shadow_v1.pt',strength=.5,recovery_mode='highlights')
    assert [p.name for p in server.master_targets(params)]==['shot.001001.exr','shot.001002.exr']
    for offset,color in enumerate(((100,120,140),(180,60,30))):
        image=Image.new('RGB',(64,48),color)
        stream=io.BytesIO(); image.save(stream,format='PNG')
        request=dict(params,render_count=1,frame_start=1001+offset)
        result=server.run_master(model,stream.getvalue(),request,SimpleNamespace())
        pixels,attrs=read_exr(result['path'])
        sdr=torch.from_numpy(np.asarray(image,dtype=np.float32)/255).permute(2,0,1)[None]
        reference=predict_image(model,sdr,True,0,64,'highlights',.5)[0].permute(1,2,0).numpy()*10000/203
        if container=='aces':
            reference=convert(reference,'rec2020','ap0')
            assert attrs['chromaticities']==pytest.approx(AP0_CHROMATICITIES,abs=1e-6)
        else:
            assert attrs['chromaticities']==pytest.approx(REC2020_CHROMATICITIES,abs=1e-6)
        assert pixels.shape==(48,64,3) and np.isfinite(pixels).all()
        np.testing.assert_allclose(pixels,reference,rtol=1e-3,atol=1e-4)
        sidecar=json.loads(Path(result['sidecar']).read_text())
        assert sidecar['resolution']==[64,48]
        assert sidecar['transfer']=='linear' and sidecar['diffuse_white_nits']==203
        assert sidecar['residual_strength']==.5 and sidecar['recovery_mode']=='highlights'
        assert sidecar['checkpoint']==params['checkpoint']
        assert Path(result['path']).name==f'shot.{1001+offset:06d}.exr'
    assert len(list(tmp_path.glob('*.exr')))==2
    assert len(list(tmp_path.glob('*.json')))==2
    with pytest.raises(ValueError,match='overwrite'): server.master_targets(params)
