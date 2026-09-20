import io
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
