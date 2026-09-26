import io
import numpy as np
import pytest
from PIL import Image, ImageCms
from ui.export_formats import srgb_pixels, write_srgb
from ui.server import master_targets


def test_neutral_tone_mapping_and_black():
    values=np.array([[[0.,0.,0.],[203.,203.,203.],[2030.,2030.,2030.]]])
    out=srgb_pixels(values)
    assert out[0,0].tolist()==[0,0,0]
    assert out[0,1].tolist()==[188,188,188]
    assert np.all(out[0,2]>out[0,1])


@pytest.mark.parametrize('kind,ext',[('srgb_png','png'),('srgb_tiff','tif')])
def test_sdr_files_have_profile_and_correct_numbering(tmp_path,kind,ext):
    params=dict(render_dir=str(tmp_path),render_name='shot',container=kind,
                render_mode='sequence',render_count=2,frame_start=1001)
    paths=master_targets(params)
    assert paths[0].name==f'shot.001001.{ext}'
    write_srgb(np.full((4,5,3),203.),paths[0])
    with Image.open(paths[0]) as img:
        assert img.size==(5,4)
        assert img.getpixel((0,0))==(188,188,188)
        profile=ImageCms.ImageCmsProfile(io.BytesIO(img.info['icc_profile']))
        assert 'sRGB' in ImageCms.getProfileDescription(profile)
    with pytest.raises(ValueError,match='overwrite'): master_targets(params)


def test_unknown_preset_rejected(tmp_path):
    with pytest.raises(ValueError,match='Unsupported'):
        master_targets(dict(render_dir=str(tmp_path),container='wrong'))


@pytest.mark.parametrize('kind',['srgb_png','srgb_tiff'])
def test_real_model_sdr_export(tmp_path,kind):
    import json
    import torch
    from types import SimpleNamespace
    from ui.server import run_master
    from training.infer_sdr2hdr import load_models
    torch.set_num_threads(4)
    model=load_models('checkpoints/sdr2hdr_shadow_v1.pt',None,torch.device('cpu'))[0]
    raw=io.BytesIO(); Image.new('RGB',(32,32),(200,150,100)).save(raw,format='PNG')
    result=run_master(model,raw.getvalue(),dict(render_dir=str(tmp_path),container=kind),SimpleNamespace())
    with Image.open(result['path']) as image:
        assert image.size==(32,32)
        assert image.info['icc_profile']
    from pathlib import Path
    sidecar=json.loads(Path(result['sidecar']).read_text())
    assert sidecar['transfer']=='sRGB'
    assert sidecar['bit_depth']==8
    assert 'Reinhard' in sidecar['tone_mapping']
