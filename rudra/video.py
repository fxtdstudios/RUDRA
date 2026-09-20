"""Local SDR video -> HDR10, HLG or ProRes with audio and export QC.

Supports progressive constant-frame-rate clips. Writes a disk-backed PNG spool
instead of retaining a clip in RAM. The final file is published only after QC.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from .hdr10 import master_to_pq
from .delivery.profiles import PROFILES, encode_master


def executable(name):
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'{name} is required on PATH')
    return found


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'{Path(command[0]).name} failed: {result.stderr[-4000:]}')
    return result.stdout


def probe(path, frames=False):
    command = [executable('ffprobe'), '-v', 'error', '-show_streams', '-show_format']
    if frames:
        command += ['-select_streams', 'v:0', '-show_frames', '-show_entries',
                    'frame=best_effort_timestamp_time,duration_time,pkt_duration_time,width,height:frame_side_data']
    return json.loads(run([*command, '-of', 'json', str(path)]))


def timing(stream, frames):
    fps = Fraction(stream.get('avg_frame_rate', '0/1'))
    if fps <= 0 or not frames:
        raise ValueError('No valid video frame rate/timestamps')
    pts = [float(f['best_effort_timestamp_time']) for f in frames]
    if not all(math.isfinite(p) for p in pts):
        raise ValueError('Invalid video timestamps')
    tick = float(Fraction(stream.get('time_base', '1/1000000')))
    tolerance = max(tick * 1.5, 0.000002)
    period = float(1 / fps)
    if any(abs((p - pts[0]) - i * period) > tolerance for i, p in enumerate(pts)):
        raise ValueError('Variable frame rate or discontinuous timestamps: normalize explicitly before conversion')
    return dict(fps=str(fps), frames=len(pts), start=pts[0], duration=len(pts) * period,
                tolerance=tolerance)


def input_contract(stream, args):
    if stream.get('color_transfer') in ('smpte2084', 'arib-std-b67'):
        raise ValueError('Input is already HDR; this command accepts SDR only')
    pix = stream.get('pix_fmt', '')
    if has_alpha(stream):
        if args.format != 'prores4444':
            raise ValueError('Alpha input requires --format prores4444')
        if args.alpha_mode != 'straight':
            raise ValueError('Declare --alpha-mode straight; premultiplied input must be unpremultiplied first')
    if stream.get('field_order', 'progressive') not in ('progressive', 'unknown'):
        raise ValueError('Interlaced input must be deinterlaced explicitly')
    rotation = float(stream.get('tags', {}).get('rotate', 0))
    for data in stream.get('side_data_list', []):
        rotation = float(data.get('rotation', rotation))
    if rotation % 360:
        raise ValueError('Bake the input display rotation before conversion')
    if stream.get('sample_aspect_ratio', '1:1') not in ('1:1', '0:1', 'N/A'):
        raise ValueError('Anamorphic input must be converted to square pixels explicitly')
    if stream['width'] % 2 or stream['height'] % 2:
        raise ValueError('HDR10 4:2:0 requires even dimensions; crop or pad explicitly')
    transfer = args.input_transfer
    if transfer == 'auto':
        transfer = {'bt709': 'rec709', 'iec61966-2-1': 'srgb', 'gamma22': 'gamma22'}.get(stream.get('color_transfer'))
    primaries = args.input_primaries
    if primaries == 'auto':
        primaries = {'bt709': 'rec709', 'bt2020': 'rec2020'}.get(stream.get('color_primaries'))
    rgb = pix.startswith(('rgb', 'bgr', 'gbr'))
    matrix = args.input_matrix
    if matrix == 'auto':
        matrix = 'gbr' if rgb else {'bt709': 'bt709', 'bt2020nc': 'bt2020nc'}.get(stream.get('color_space'))
    value_range = args.input_range
    if value_range == 'auto':
        value_range = 'full' if rgb else {'tv': 'limited', 'pc': 'full'}.get(stream.get('color_range'))
    for name, value in [('transfer',transfer), ('primaries',primaries), ('matrix',matrix), ('range',value_range)]:
        if not value:
            raise ValueError(f'Missing/unsupported colour {name}; specify --input-{name}')
    if rgb and (value_range != 'full' or matrix != 'gbr'):
        raise ValueError('RGB input requires full range and GBR matrix')
    return dict(transfer=transfer,primaries=primaries,matrix=matrix,range=value_range)


def has_alpha(stream):
    return any(s in stream.get('pix_fmt', '') for s in ('rgba', 'bgra', 'argb', 'abgr', 'yuva', 'gbrap', 'ya'))


def decoder_filter(contract, alpha=False):
    matrix = {'bt709':'709','bt2020nc':'2020_ncl','gbr':'gbr'}[contract['matrix']]
    return (f"zscale=matrixin={matrix}:rangein={contract['range']}:matrix=gbr:range=full,"
            + ('format=gbrap16le,format=rgba64le' if alpha else 'format=gbrp16le,format=rgb48le'))


class ShadowSmoother:
    """Optional scalar gate smoothing. Resets at hard cuts; never blends pixels."""
    def __init__(self, retention=0.0, cut_threshold=0.15):
        self.retention, self.cut_threshold = retention, cut_threshold
        self.previous = self.weight = None

    def update(self, thumbnail, weight):
        cut = self.previous is None or float(np.mean(np.abs(thumbnail-self.previous))) >= self.cut_threshold
        self.weight = weight if cut or self.weight is None else self.retention*self.weight+(1-self.retention)*weight
        self.previous = thumbnail.copy()
        return self.weight, cut


class Predictor:
    def __init__(self, checkpoint, device, tile_size, overlap):
        import torch
        from .sdr2hdr import SDR2HDRNet
        self.torch, self.device = torch, torch.device(device)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = SDR2HDRNet.from_config(payload.get('config', {}))
        self.model.load_state_dict(payload.get('model',payload),strict=True)
        self.model.to(self.device).eval()
        self.tile_size, self.overlap = tile_size, overlap

    def predict(self, rgb, contract, smoother):
        import torch
        import torch.nn.functional as F
        from .sdr2hdr import canonicalize_sdr, srgb_to_linear, linear_to_srgb
        from .delivery.colorspace import rgb_to_rgb_matrix
        with torch.inference_mode():
            x = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(self.device)
            x = canonicalize_sdr(x, contract['transfer'], 'full')
            if contract['primaries'] != 'rec2020':
                matrix = torch.as_tensor(rgb_to_rgb_matrix(contract['primaries'],'rec2020'),device=self.device,dtype=x.dtype)
                x = linear_to_srgb(torch.einsum('ij,bjhw->bihw',matrix,srgb_to_linear(x)))
            thumb = F.interpolate(x,size=(36,64),mode='area')[0].cpu().numpy()
            weight = self.model.predict_shadow_weight(x)
            scalar, cut = smoother.update(thumb, float(weight.item()) if weight is not None else 1.0)
            scale = self.model.predict_residual_scale(x)
            def forward(tile):
                return self.model(tile,preserve_outside=True,recovery_mode='all',
                                  shadow_weight=scalar,residual_scale=scale).hdr
            h,w = x.shape[-2:]
            if self.tile_size <= 0 or max(h,w) <= self.tile_size:
                result = forward(x)
            else:
                size, overlap = self.tile_size, self.overlap
                def starts(length):
                    if length <= size: return [0]
                    values = list(range(0,length-size+1,size-overlap))
                    if values[-1] != length-size: values.append(length-size)
                    return values
                result, weights = torch.zeros_like(x), torch.zeros_like(x[:,:1])
                for y in starts(h):
                    for left in starts(w):
                        tile = x[...,y:y+size,left:left+size]
                        th,tw = tile.shape[-2:]
                        wy,wx = torch.ones(th,device=x.device),torch.ones(tw,device=x.device)
                        fy,fx = min(overlap,th//2),min(overlap,tw//2)
                        if y and fy: wy[:fy]=torch.linspace(.001,1,fy,device=x.device)
                        if y+th<h and fy: wy[-fy:]=torch.linspace(1,.001,fy,device=x.device)
                        if left and fx: wx[:fx]=torch.linspace(.001,1,fx,device=x.device)
                        if left+tw<w and fx: wx[-fx:]=torch.linspace(1,.001,fx,device=x.device)
                        blend=(wy[:,None]*wx[None,:])[None,None]
                        result[...,y:y+th,left:left+tw] += forward(tile)*blend
                        weights[...,y:y+th,left:left+tw] += blend
                result /= weights.clamp_min(1e-6)
            return result[0].permute(1,2,0).cpu().numpy(), scalar, cut


def read_frame(stream, count):
    chunks, remaining = [], count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk: break
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b''.join(chunks)
    if data and len(data) != count: raise RuntimeError('Truncated decoded frame')
    return data


def mastering_display(peak, minimum):
    return f'G(8500,39850)B(6550,2300)R(35400,14600)WP(15635,16450)L({round(peak*10000)},{round(minimum*10000)})'


def encode_command(args, source, clock, spool, output, max_cll, max_fall):
    profile = PROFILES[args.format]
    transfer = profile['transfer']
    params = ':'.join(['hdr-opt=1','repeat-headers=1','colorprim=bt2020','transfer=smpte2084',
                       'colormatrix=bt2020nc',f'master-display={mastering_display(args.peak_nits,args.min_nits)}',
                       f'max-cll={max_cll},{max_fall}'])
    cmd = [executable('ffmpeg'),'-hide_banner','-v','error','-nostdin','-n','-copyts',
           '-framerate',clock['fps'],'-i',str(spool/'%08d.png'),
           '-itsoffset',str(-clock['start']),'-i',str(source),'-map','0:v:0']
    if args.audio != 'none': cmd += ['-map','1:a?','-c:a',args.audio]
    if args.audio == 'aac': cmd += ['-b:a','320k']
    pix = 'yuva444p10le' if args.preserve_alpha else profile['pixel_format']
    cmd += ['-map_metadata','1','-map_chapters','-1','-vf',
            f'zscale=matrixin=gbr:transferin={transfer}:primariesin=2020:rangein=full:'
            f'matrix=2020_ncl:transfer={transfer}:primaries=2020:range=limited,format={pix}',
            '-c:v',profile['encoder']]
    if profile['codec'] == 'hevc':
        if args.format == 'hlg':
            params = 'repeat-headers=1:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc'
        cmd += ['-preset',args.preset,'-crf',str(args.crf),'-x265-params',params,'-tag:v','hvc1']
    else:
        cmd += ['-profile:v',str(profile['profile']),'-tag:v',profile['tag'],
                '-alpha_bits','16' if args.preserve_alpha else '0']
    cmd += ['-color_primaries','bt2020','-color_trc',transfer,'-colorspace','bt2020nc',
            '-color_range','tv','-fps_mode','passthrough',
            '-frames:v',str(clock['frames']),'-t',str(clock['duration']),'-avoid_negative_ts','disabled']
    if output.suffix.lower() in ('.mp4','.mov'): cmd += ['-movflags','+faststart+write_colr']
    return [*cmd,str(output)]


def quality_check(source_info, clock, output, audio_mode, expected_hdr=None, delivery_format='hdr10', alpha=False):
    info, video_frames = probe(output), probe(output,frames=True)['frames']
    src_video = next(s for s in source_info['streams'] if s['codec_type']=='video')
    video = next(s for s in info['streams'] if s['codec_type']=='video')
    errors = []
    profile = PROFILES[delivery_format]
    pix = ('yuva444p12le' if alpha else 'yuv444p12le') if delivery_format=='prores4444' else profile['pixel_format']
    for key,value in dict(codec_name=profile['codec'],pix_fmt=pix,color_primaries='bt2020',
                           color_transfer=profile['transfer'],color_space='bt2020nc',color_range='tv',
                           width=src_video['width'],height=src_video['height']).items():
        # MOV's nclc colour atom carries primaries/transfer/matrix, not a range
        # bit. ProRes video-range conversion is explicit in the encoder filter;
        # ffprobe can legitimately report no separate color_range field.
        if key=='color_range' and profile['codec']=='prores' and video.get(key) is None:
            continue
        if video.get(key)!=value: errors.append(f'{key}: expected {value}, got {video.get(key)}')
    if len(video_frames)!=clock['frames']: errors.append('Video frame count changed')
    period = float(1/Fraction(clock['fps']))
    tick = float(Fraction(video['time_base']))
    for i,frame in enumerate(video_frames):
        if abs(float(frame['best_effort_timestamp_time'])-i*period)>max(2*tick,0.00001):
            errors.append('Output presentation timestamps changed'); break
    side = [d for f in video_frames[:1] for d in f.get('side_data_list',[])]
    if delivery_format == 'hdr10':
        if not any(d['side_data_type']=='Mastering display metadata' for d in side): errors.append('Missing mastering display SEI')
        if not any(d['side_data_type']=='Content light level metadata' for d in side): errors.append('Missing MaxCLL/MaxFALL SEI')
    if delivery_format=='hlg' and any(d['side_data_type'] in ('Mastering display metadata','Content light level metadata') for d in side):
        errors.append('HLG incorrectly contains HDR10 static metadata')
    if profile['codec']=='prores' and video.get('codec_tag_string')!=profile['tag']:
        errors.append('Incorrect ProRes profile tag')
    if expected_hdr and delivery_format=='hdr10':
        for data in side:
            if data['side_data_type']=='Content light level metadata':
                if data.get('max_content')!=expected_hdr['max_cll'] or data.get('max_average')!=expected_hdr['max_fall']:
                    errors.append('Encoded content-light metadata differs from measured values')
            if data['side_data_type']=='Mastering display metadata':
                for key, expected in [('max_luminance',expected_hdr['peak']),('min_luminance',expected_hdr['minimum'])]:
                    if abs(float(Fraction(data[key]))-expected)>.0001: errors.append(f'Wrong mastering {key}')
    if 'duration' in video and abs(float(video['duration'])-clock['duration'])>max(2*tick,.00001):
        errors.append('Video duration changed')
    source_audio = [s for s in source_info['streams'] if s['codec_type']=='audio'] if audio_mode!='none' else []
    target_audio = [s for s in info['streams'] if s['codec_type']=='audio']
    if len(source_audio)!=len(target_audio): errors.append('Audio stream count changed')
    for src,dst in zip(source_audio,target_audio):
        for key in ('channels','sample_rate'):
            if src.get(key)!=dst.get(key): errors.append(f'Audio {key} changed')
        if audio_mode=='copy' and src.get('codec_name')!=dst.get('codec_name'): errors.append('Audio codec changed in copy mode')
        expected_start = max(0,float(src.get('start_time',0))-clock['start'])
        if abs(float(dst.get('start_time',0))-expected_start)>.06: errors.append('Audio start offset changed')
        if 'duration' in src and 'duration' in dst:
            expected_end=min(clock['duration'],float(src.get('start_time',0))+float(src['duration'])-clock['start'])
            actual_end=float(dst.get('start_time',0))+float(dst['duration'])
            if abs(actual_end-expected_end)>.06: errors.append('Audio end offset changed')
    # Decode the complete result: valid tags alone do not establish decodability.
    run([executable('ffmpeg'),'-v','error','-xerror','-nostdin','-i',str(output),'-map','0:v:0','-map','0:a?','-f','null','-'])
    if errors: raise RuntimeError('Export QC failed: '+'; '.join(errors))
    return dict(passed=True,video_frames=len(video_frames),audio_streams=len(target_audio),
                video=video,hdr_side_data=side,decoded_without_errors=True,
                range_contract='limited; ProRes MOV nclc may omit a separate range flag' if profile['codec']=='prores' else 'limited, signalled')


def check_alpha(output, spool, frame_count, width, height):
    """Check every decoded alpha pixel against the untouched input spool.

prores_ks accepts a 10-bit input plane even when alpha_bits=16. Permit at
most two 10-bit code steps; do not claim arbitrary 16-bit alpha is lossless.
"""
    import cv2
    maximum = 0
    with (spool/'alpha_qc.log').open('wb') as errors:
        command=[executable('ffmpeg'),'-v','error','-nostdin','-i',str(output),'-map','0:v:0',
                 '-vf','alphaextract,format=gray16le','-fps_mode','passthrough',
                 '-f','rawvideo','-pix_fmt','gray16le','pipe:1']
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
        try:
            for index in range(frame_count):
                raw=read_frame(process.stdout,width*height*2)
                if not raw: raise RuntimeError('Missing decoded alpha frame')
                actual=np.frombuffer(raw,dtype='<u2').reshape(height,width).astype(np.int32)
                expected=cv2.imread(str(spool/f'{index:08d}.png'),cv2.IMREAD_UNCHANGED)[...,3].astype(np.int32)
                maximum=max(maximum,int(np.max(np.abs(actual-expected))))
            if process.stdout.read(1): raise RuntimeError('Unexpected extra alpha frames')
            if process.wait(): raise RuntimeError('Alpha decode failed')
        finally:
            process.stdout.close()
            if process.poll() is None: process.kill()
            process.wait()
    if maximum>128:
        raise RuntimeError(f'Alpha QC failed: maximum 16-bit code error {maximum} exceeds 128')
    return dict(passed=True,max_error_16bit_codes=maximum,tolerance_16bit_codes=128,
                note='Source alpha bypasses grading; encoder input quantizes to 10 bits')


def convert_video(args, progress=None):
    import cv2
    source, output = args.input.resolve(), args.output.resolve()
    sidecar = output.with_suffix(output.suffix+'.json')
    if output.suffix.lower() not in ('.mp4','.mov','.mkv'): raise ValueError('Output must be MP4, MOV or MKV')
    profile = PROFILES[args.format]
    if profile['codec']=='prores' and output.suffix.lower()!='.mov':
        raise ValueError('ProRes delivery requires a .mov output')
    if not source.is_file(): raise FileNotFoundError(source)
    if source==output or output.exists() or sidecar.exists(): raise ValueError('Refusing to overwrite source/output/sidecar')
    encoders = run([executable('ffmpeg'), '-hide_banner', '-encoders'])
    filters = run([executable('ffmpeg'), '-hide_banner', '-filters'])
    if profile['encoder'] not in encoders or 'zscale' not in filters:
        raise RuntimeError(f"This FFmpeg build needs {profile['encoder']} and zscale support")
    if not 0 <= args.shadow_smoothing < 1 or not 0 < args.cut_threshold <= 1: raise ValueError('Invalid smoothing/cut threshold')
    if args.tile_size < 0 or args.tile_overlap < 0 or (args.tile_size and args.tile_overlap >= args.tile_size): raise ValueError('Invalid tile size/overlap')
    if not 0 <= args.crf <= 51 or not 0 <= args.min_nits < args.peak_nits: raise ValueError('Invalid mastering/encoding settings')
    master_to_pq(np.zeros((1,1,3),np.float32),args.peak_nits,args.knee_nits)
    source_info, frame_info = probe(source), probe(source,frames=True)
    videos = [s for s in source_info['streams'] if s['codec_type']=='video']
    if len(videos)!=1: raise ValueError('Select a source containing exactly one video stream')
    stream, frames = videos[0], frame_info['frames']
    contract, clock = input_contract(stream,args), timing(stream,frames)
    args.preserve_alpha = has_alpha(stream)
    if any(f['width']!=stream['width'] or f['height']!=stream['height'] for f in frames): raise ValueError('Changing frame dimensions are unsupported')
    output.parent.mkdir(parents=True,exist_ok=True)
    work_parent = (args.work_dir or output.parent).resolve()
    work_parent.mkdir(parents=True,exist_ok=True)
    # PNG spool size depends on content; reserve one uncompressed frame plus margin.
    channels = 4 if args.preserve_alpha else 3
    frame_bytes=stream['width']*stream['height']*channels*2
    if shutil.disk_usage(work_parent).free < frame_bytes*2+64*1024**2: raise RuntimeError('Insufficient working disk space')
    predictor = Predictor(args.checkpoint,args.device,args.tile_size,args.tile_overlap)
    smoother = ShadowSmoother(args.shadow_smoothing,args.cut_threshold)
    started=time.monotonic()
    report=dict(source=str(source),output=str(output),input_contract=contract,timing=clock,
                delivery_format=args.format,alpha_mode=args.alpha_mode if args.preserve_alpha else None,
                encoder_input_precision_bits=10,alpha_preserved=args.preserve_alpha,
                checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                mastering_peak_nits=args.peak_nits,mastering_min_nits=args.min_nits if args.format=='hdr10' else None,
                audio_mode=args.audio,shadow_smoothing=args.shadow_smoothing,cut_threshold=args.cut_threshold,
                preserve_outside=True,recovery_mode='all',tile_size=args.tile_size,tile_overlap=args.tile_overlap,
                timestamp_policy='Video begins at zero; audio retains relative offset, clipped to video interval',
                scene_cuts=[],frames=[],warnings=['Scalar shadow smoothing is not validated temporal reconstruction'] if args.shadow_smoothing else [])
    with tempfile.TemporaryDirectory(prefix='rudra-video-',dir=work_parent) as work:
        spool=Path(work)
        with (spool/'decode.log').open('wb') as decoder_log:
            cmd=[executable('ffmpeg'),'-hide_banner','-v','error','-xerror','-nostdin','-noautorotate','-i',str(source),
                 '-map','0:v:0','-vf',decoder_filter(contract,args.preserve_alpha),'-fps_mode','passthrough',
                 '-pix_fmt','rgba64le' if args.preserve_alpha else 'rgb48le','-f','rawvideo','pipe:1']
            process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=decoder_log)
            try:
                for index in range(clock['frames']):
                    data=read_frame(process.stdout,frame_bytes)
                    if not data: raise RuntimeError('Decoder ended before expected frame count')
                    decoded=np.frombuffer(data,dtype='<u2').reshape(stream['height'],stream['width'],channels)
                    rgb=decoded[...,:3].astype(np.float32)/65535
                    hdr,weight,cut=predictor.predict(rgb,contract,smoother)
                    if not np.isfinite(hdr).all() or np.any(hdr<0): raise RuntimeError(f'Invalid HDR pixels at frame {index}')
                    pq,mastered=encode_master(hdr,args.format,args.peak_nits,args.knee_nits)
                    if not np.isfinite(pq).all(): raise RuntimeError(f'Invalid PQ pixels at frame {index}')
                    maxrgb=mastered.max(axis=2)
                    report['frames'].append(dict(index=index,shadow_weight=weight,max_cll=float(maxrgb.max()),
                                                 frame_average=float(maxrgb.mean())))
                    if cut: report['scene_cuts'].append(index)
                    if shutil.disk_usage(spool).free < frame_bytes*2+64*1024**2: raise RuntimeError('Working disk is full')
                    packed=(np.clip(pq,0,1)*65535+.5).astype(np.uint16)
                    packed = packed[...,::-1]
                    if args.preserve_alpha:
                        packed=np.concatenate((packed,decoded[...,3:4]),axis=2)
                    if not cv2.imwrite(str(spool/f'{index:08d}.png'),packed): raise RuntimeError('Failed writing signal spool')
                    if index%10==0 or index+1==clock['frames']:
                        if progress: progress(dict(phase='inference', frames_done=index+1, frames_total=clock['frames']))
                        print(f'Frame {index+1}/{clock["frames"]}, {(index+1)/max(time.monotonic()-started,.001):.2f} fps',flush=True)
                if process.stdout.read(1): raise RuntimeError('Decoder produced unexpected extra frames')
                if process.wait(): raise RuntimeError('Decoder failed; '+(spool/'decode.log').read_text(errors='replace')[-2000:])
            finally:
                process.stdout.close()
                if process.poll() is None: process.kill()
                process.wait()
        max_cll=math.ceil(max(f['max_cll'] for f in report['frames']))
        max_fall=math.ceil(max(f['frame_average'] for f in report['frames']))
        # Stage on the output volume so the final rename is atomic.
        with tempfile.TemporaryDirectory(prefix='rudra-master-',dir=output.parent) as staging:
            staged=Path(staging)/output.name
            command=encode_command(args,source,clock,spool,staged,max_cll,max_fall)
            report['encode_command']=command
            if progress: progress(dict(phase='encoding', frames_done=clock['frames'], frames_total=clock['frames']))
            run(command)
            if progress: progress(dict(phase='quality_check', frames_done=clock['frames'], frames_total=clock['frames']))
            report['qc']=quality_check(source_info,clock,staged,args.audio,
                                      dict(max_cll=max_cll,max_fall=max_fall,peak=args.peak_nits,minimum=args.min_nits),
                                      args.format,args.preserve_alpha)
            if args.preserve_alpha:
                report['qc']['alpha']=check_alpha(staged,spool,clock['frames'],stream['width'],stream['height'])
            if args.format=='hlg':
                report['hlg_reference']=dict(display_peak_nits=args.peak_nits,black_nits=0,
                                             system_gamma=1.2+.42*math.log10(args.peak_nits/1000))
            report.update(max_cll=max_cll,max_fall=max_fall,elapsed_seconds=time.monotonic()-started)
            staged_json=Path(staging)/sidecar.name
            staged_json.write_text(json.dumps(report,indent=2),encoding='utf-8')
            if output.exists() or sidecar.exists(): raise ValueError('Output appeared during processing; refusing overwrite')
            staged.rename(output)
            staged_json.rename(sidecar)
    print(f'QC passed: {output}\nReport: {sidecar}',flush=True)
    return 0


def add_arguments(parser):
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--format',choices=list(PROFILES),default='hdr10',help='Delivery preset; ProRes uses PQ and requires MOV')
    parser.add_argument('--alpha-mode',choices=['straight'],help='Required declaration for alpha input; unpremultiply before conversion')
    parser.add_argument('--device',default='cpu',help='cpu (default) or cuda; choose explicitly when training shares the GPU')
    parser.add_argument('--input-transfer',choices=['auto','srgb','rec709','gamma22','gamma24'],default='auto')
    parser.add_argument('--input-primaries',choices=['auto','rec709','rec2020'],default='auto')
    parser.add_argument('--input-matrix',choices=['auto','bt709','bt2020nc','gbr'],default='auto')
    parser.add_argument('--input-range',choices=['auto','full','limited'],default='auto')
    parser.add_argument('--audio',choices=['copy','aac','none'],default='copy',help='copy preserves encoded audio; use aac for incompatible MP4 audio')
    parser.add_argument('--peak-nits',type=float,default=1000)
    parser.add_argument('--min-nits',type=float,default=.005,help='HDR10 mastering-display black metadata; HLG reference black is zero')
    parser.add_argument('--knee-nits',type=float)
    parser.add_argument('--crf',type=int,default=12)
    parser.add_argument('--preset',choices=['ultrafast','superfast','veryfast','faster','fast','medium','slow','slower','veryslow'],default='medium')
    parser.add_argument('--tile-size',type=int,default=512)
    parser.add_argument('--tile-overlap',type=int,default=64)
    parser.add_argument('--shadow-smoothing',type=float,default=0,help='Previous shadow-weight retention, [0,1); opt-in and reset at cuts')
    parser.add_argument('--cut-threshold',type=float,default=.15)
    parser.add_argument('--work-dir',type=Path,help='Directory for temporary PQ frames; removed after completion or failure')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args=parser.parse_args(argv)
    return convert_video(args)


if __name__=='__main__': raise SystemExit(main())
