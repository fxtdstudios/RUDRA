"""End-to-end media checks require ffmpeg with libx264/libx265 and zscale."""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from rudra import video


def args_for(source, target):
    parser=argparse.ArgumentParser()
    video.add_arguments(parser)
    return parser.parse_args([str(source),'--output',str(target),'--checkpoint','checkpoints/sdr2hdr_shadow_v1.pt',
                              '--preset','ultrafast'])


def test_fractional_fps_and_nonzero_origin():
    stream={'avg_frame_rate':'30000/1001','time_base':'1/30000'}
    frames=[{'best_effort_timestamp_time':str(2+i*1001/30000)} for i in range(12)]
    result=video.timing(stream,frames)
    assert result['fps']=='30000/1001'
    assert result['duration']==pytest.approx(.4004)
    assert result['start']==2
    frames[5]['best_effort_timestamp_time']=str(2+5*1001/30000+.01)
    with pytest.raises(ValueError,match='Variable frame rate'): video.timing(stream,frames)


def test_colour_metadata_requires_explicit_override():
    args=args_for('in.mp4','out.mp4')
    stream=dict(pix_fmt='yuv420p',width=64,height=64,color_range='tv',color_space='bt709')
    with pytest.raises(ValueError,match='transfer'): video.input_contract(stream,args)
    args.input_transfer='rec709'
    args.input_primaries='rec709'
    assert video.input_contract(stream,args)==dict(transfer='rec709',primaries='rec709',matrix='bt709',range='limited')
    stream['color_transfer']='smpte2084'
    with pytest.raises(ValueError,match='already HDR'): video.input_contract(stream,args)


def test_shadow_smoothing_resets_at_cut():
    smoother=video.ShadowSmoother(.8,.15)
    black=np.zeros((3,36,64))
    assert smoother.update(black,0)==(0,True)
    weight,cut=smoother.update(black,1)
    assert weight==pytest.approx(.2) and not cut
    assert smoother.update(np.ones_like(black),.9)==(.9,True)


def test_truncated_frame_is_not_silently_dropped():
    with pytest.raises(RuntimeError,match='Truncated'): video.read_frame(io.BytesIO(b'123'),6)
    assert video.read_frame(io.BytesIO(),6)==b''


@pytest.fixture
def media_tools():
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'): pytest.skip('ffmpeg/ffprobe unavailable')
    encoders=video.run([video.executable('ffmpeg'),'-hide_banner','-encoders'])
    filters=video.run([video.executable('ffmpeg'),'-hide_banner','-filters'])
    if not all(c in encoders for c in ('libx264','libx265')) or 'zscale' not in filters:
        pytest.skip('Need libx264, libx265, zscale')


def create_source(path, audio=True, offset=0):
    command=[video.executable('ffmpeg'),'-v','error','-nostdin','-n','-f','lavfi','-i',
             'testsrc2=size=96x64:rate=30000/1001:duration=0.4004']
    if audio: command += ['-itsoffset',str(offset),'-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=0.2']
    command += ['-c:v','libx264','-x264-params','colorprim=bt709:transfer=bt709:colormatrix=bt709',
                '-color_range','tv']
    if audio: command += ['-c:a','aac']
    video.run([*command,str(path)])


class BaselinePredictor:
    def __init__(self,*_): pass
    def predict(self,rgb,contract,smoother):
        weight,cut=smoother.update(rgb[::4,::4],.5)
        return rgb*.03,weight,cut


@pytest.mark.parametrize('audio,offset',[(False,0),(True,0),(True,.1)])
def test_video_roundtrip_qc_and_audio_copy(tmp_path,monkeypatch,media_tools,audio,offset):
    source,target=tmp_path/'source.mp4',tmp_path/'master.mp4'
    create_source(source,audio,offset)
    monkeypatch.setattr(video,'Predictor',BaselinePredictor)
    args=args_for(source,target)
    # Some FFmpeg builds omit these stream tags despite x264 VUI settings.
    args.input_transfer=args.input_primaries='rec709'
    assert video.convert_video(args)==0
    report=json.loads(target.with_suffix('.mp4.json').read_text())
    assert report['qc']['passed']
    assert report['qc']['video_frames']==12
    assert report['qc']['audio_streams']==int(audio)
    assert report['timing']['fps']=='30000/1001'
    assert not list(tmp_path.glob('rudra-*'))
    if audio:
        def hashes(path):
            raw=video.run([video.executable('ffprobe'),'-v','error','-select_streams','a',
                           '-show_packets','-show_data_hash','sha256','-show_entries','packet=data_hash',
                           '-of','json',str(path)])
            return [p['data_hash'] for p in json.loads(raw)['packets']]
        assert hashes(source)==hashes(target), 'Compressed audio packets changed'
    with pytest.raises(ValueError,match='overwrite'): video.convert_video(args)


def test_failed_qc_does_not_publish_master(tmp_path,monkeypatch,media_tools):
    source,target=tmp_path/'source.mp4',tmp_path/'master.mp4'
    create_source(source,False)
    monkeypatch.setattr(video,'Predictor',BaselinePredictor)
    def reject(*_,**__): raise RuntimeError('simulated QC failure')
    monkeypatch.setattr(video,'quality_check',reject)
    args=args_for(source,target)
    args.input_transfer=args.input_primaries='rec709'
    with pytest.raises(RuntimeError,match='QC failure'): video.convert_video(args)
    assert not target.exists() and not target.with_suffix('.mp4.json').exists()
    assert not list(tmp_path.glob('rudra-*'))


def test_nonzero_source_origin_keeps_audio_alignment(tmp_path,monkeypatch,media_tools):
    original,source,target=tmp_path/'original.mp4',tmp_path/'offset.mp4',tmp_path/'master.mp4'
    create_source(original,True,.1)
    video.run([video.executable('ffmpeg'),'-v','error','-nostdin','-n','-i',str(original),
               '-map','0','-c','copy','-output_ts_offset','2',str(source)])
    monkeypatch.setattr(video,'Predictor',BaselinePredictor)
    args=args_for(source,target)
    args.input_transfer=args.input_primaries='rec709'
    video.convert_video(args)
    report=json.loads(target.with_suffix('.mp4.json').read_text())
    assert report['timing']['start']==pytest.approx(2)
    assert report['qc']['passed']


@pytest.mark.parametrize('tile_size',[0,32])
def test_video_inference_matches_existing_frame_path(tile_size):
    torch=pytest.importorskip('torch')
    from training.infer_sdr2hdr import predict_image
    path=Path(__file__).resolve().parents[1]/'checkpoints/sdr2hdr_shadow_v1.pt'
    predictor=video.Predictor(path,'cpu',tile_size,8)
    rgb=np.random.default_rng(3).random((40,48,3),dtype=np.float32)
    actual,_,_=predictor.predict(rgb,dict(transfer='srgb',primaries='rec2020'),video.ShadowSmoother())
    x=torch.from_numpy(rgb).permute(2,0,1)[None]
    expected=predict_image(predictor.model,x,True,tile_size,8,'all')[0].permute(1,2,0).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('delivery_format',['hlg','prores422','prores422hq','prores4444'])
def test_delivery_presets(tmp_path,monkeypatch,media_tools,delivery_format):
    source,target=tmp_path/'source.mp4',tmp_path/'master.mov'
    create_source(source,True,.1)
    monkeypatch.setattr(video,'Predictor',BaselinePredictor)
    args=args_for(source,target)
    args.format=delivery_format
    args.input_transfer=args.input_primaries='rec709'
    video.convert_video(args)
    report=json.loads(target.with_suffix('.mov.json').read_text())
    assert report['qc']['passed'] and report['qc']['video_frames']==12
    assert report['qc']['audio_streams']==1
    assert report['delivery_format']==delivery_format
    if delivery_format=='hlg':
        assert report['hlg_reference']['system_gamma']==1.2
    else:
        assert report['qc']['video']['codec_name']=='prores'


def test_prores_alpha_roundtrip(tmp_path,monkeypatch,media_tools):
    import cv2
    image=np.zeros((64,96,4),np.uint16)
    image[...,:3]=24000
    image[...,3]=np.linspace(0,65535,96,dtype=np.uint16)[None,:]
    cv2.imwrite(str(tmp_path/'alpha.png'),image)
    source,target=tmp_path/'alpha.mkv',tmp_path/'alpha_hdr.mov'
    video.run([video.executable('ffmpeg'),'-v','error','-nostdin','-n','-loop','1','-framerate','24',
               '-i',str(tmp_path/'alpha.png'),'-frames:v','4','-c:v','ffv1','-pix_fmt','gbrap16le',str(source)])
    monkeypatch.setattr(video,'Predictor',BaselinePredictor)
    args=args_for(source,target)
    args.input_transfer='srgb'
    args.input_primaries='rec709'
    args.format='prores4444'
    with pytest.raises(ValueError,match='alpha-mode straight'): video.convert_video(args)
    args.alpha_mode='straight'
    video.convert_video(args)
    report=json.loads(target.with_suffix('.mov.json').read_text())
    assert report['qc']['alpha']['passed']
    assert report['qc']['alpha']['max_error_16bit_codes']<=128
    assert report['qc']['video']['pix_fmt']=='yuva444p12le'
    assert report['alpha_preserved']


def test_prores_rejects_wrong_container_before_work(tmp_path):
    args=args_for(tmp_path/'input.mp4',tmp_path/'output.mp4')
    args.format='prores422'
    with pytest.raises(ValueError,match='.mov'): video.convert_video(args)


@pytest.mark.parametrize('delivery_format',['hlg','prores422','prores4444'])
def test_encoded_neutral_patch_levels(tmp_path,monkeypatch,media_tools,delivery_format):
    source,target=tmp_path/'source.mp4',tmp_path/'master.mov'
    create_source(source,False)
    class Patches(BaselinePredictor):
        def predict(self,rgb,contract,smoother):
            hdr=np.zeros_like(rgb)
            hdr[:,rgb.shape[1]//2:]=.0203  # 203 cd/m2 neutral, below the knee
            return hdr,.5,False
    monkeypatch.setattr(video,'Predictor',Patches)
    args=args_for(source,target)
    args.format=delivery_format
    args.input_transfer=args.input_primaries='rec709'
    video.convert_video(args)
    pix='yuv444p12le' if delivery_format=='prores4444' else ('yuv422p10le' if delivery_format=='prores422' else 'yuv420p10le')
    result=subprocess.run([video.executable('ffmpeg'),'-v','error','-i',str(target),'-frames:v','1',
                           '-f','rawvideo','-pix_fmt',pix,'pipe:1'],capture_output=True,check=True)
    y=np.frombuffer(result.stdout,dtype='<u2',count=96*64).reshape(64,96)
    multiplier=4 if delivery_format=='prores4444' else 1
    assert float(np.mean(y[:,8:32]))==pytest.approx(64*multiplier,abs=3*multiplier)
    # HLG reference white 203 nits is about code .75; PQ is about .5807.
    expected_code=.75 if delivery_format=='hlg' else .5807
    assert float(np.mean(y[:,64:88]))==pytest.approx((64+876*expected_code)*multiplier,abs=3*multiplier)
