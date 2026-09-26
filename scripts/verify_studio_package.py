"""Exercise the installed wheel from outside the source checkout on CUDA."""
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import cv2
import numpy as np
import torch
from PIL import Image
import ui.server as server
import rudra
from rudra.delivery.exr import read_exr
from ui.ocio_support import describe


def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    assert Path(server.__file__).is_relative_to(Path(sys.prefix))
    assert Path(rudra.__file__).is_relative_to(Path(sys.prefix))
    assert torch.cuda.is_available(), 'GPU verification requires CUDA'
    torch.set_num_threads(4)
    model,info=server.load_model(a.checkpoint,'cuda')
    config=describe()
    ocio=dict(config='',cache_id=config['cache_id'],input='sRGB Encoded Rec.709 (sRGB)',
              model_srgb='sRGB Encoded Rec.709 (sRGB)',working_linear='Linear Rec.2020',
              output='ACEScg',display='sRGB - Display',view='ACES 1.0 - SDR Video')
    records=[]
    for preset in ('aces','linear','srgb_png','srgb_tiff','ocio_exr','ocio_view_png'):
        directory=a.out/preset
        for index in (1001,1002):
            pixels=np.full((64,96,3),20001+(index-1001),dtype=np.uint16)
            pixels[:,48:]+=1
            ok,encoded=cv2.imencode('.png',pixels); assert ok
            params=dict(input_auto=True,container=preset,render_dir=str(directory),render_name='verify',
                        render_mode='sequence',render_count=1,frame_start=index,ocio=ocio)
            result=server.run_master(model,encoded.tobytes(),params,SimpleNamespace())
            output=Path(result['path']); meta=json.loads(Path(result['sidecar']).read_text())
            assert output.exists() and f'.{index:06d}.' in output.name
            assert meta['input_color']['bit_depth']==16
            if output.suffix=='.exr':
                decoded,attrs=read_exr(output); assert np.isfinite(decoded).all()
                assert decoded.shape==(64,96,3)
                if preset in ('aces','linear'): assert 'chromaticities' in attrs
            else:
                image=Image.open(output); assert image.size==(96,64)
                if preset.startswith('srgb'): assert image.info.get('icc_profile')
            try: server.run_master(model,encoded.tobytes(),params,SimpleNamespace())
            except ValueError: pass
            else: raise AssertionError('Overwrite was not rejected')
            records.append(str(output))
    header,body=server.run_frame(model,encoded.tobytes(),dict(input_auto=True),SimpleNamespace())
    assert header['sdr_dtype']=='float32'
    source=np.frombuffer(body,dtype='<f4',offset=header['offsets']['sdr']).reshape(64,96,3)
    np.testing.assert_allclose(source,pixels/65535,atol=1e-7)
    result=dict(version=rudra.__version__,python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
                module=server.__file__,checkpoint=str(a.checkpoint),renders=records,passed=True,
                scope='Fresh isolated environment on existing Windows host; not clean OS or calibrated HDR verification.')
    (a.out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
