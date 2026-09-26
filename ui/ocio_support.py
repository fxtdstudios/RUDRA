"""Optional, explicit OCIO bridge around the fixed model color contract."""
from pathlib import Path
import hashlib
import numpy as np
import io


class PrecisionImage:
    """Float RGB in the legacy 0..255 domain, without integer quantization."""
    def __init__(self, pixels):
        self.pixels = np.asarray(pixels, dtype=np.float32)
        self.height, self.width = self.pixels.shape[:2]
        self.size = (self.width, self.height)

    def __array__(self, dtype=None, copy=None):
        return np.array(self.pixels, dtype=dtype, copy=True) if copy else np.asarray(self.pixels, dtype=dtype)

    def resize(self, size, resample=None):
        from PIL import Image
        channels = [np.asarray(Image.fromarray(self.pixels[..., c]).resize(size, resample))
                    for c in range(3)]
        return PrecisionImage(np.clip(np.stack(channels, axis=-1), 0, 255))


def decode_input(image_bytes, params):
    """Honor embedded ICC before inference; never guess a profile from appearance."""
    from PIL import Image, ImageCms
    image=Image.open(io.BytesIO(image_bytes))
    # Pillow exposes 16-bit RGB PNG/TIFF as RGB but convert('RGB') loses precision.
    # Inspect container metadata before any color conversion, including manual mode.
    bits = (image_bytes[24] if image.format == 'PNG' and len(image_bytes)>24 else 8)
    if image.format == 'TIFF':
        sample_bits = image.tag_v2.get(258, (8,))
        bits = max(sample_bits) if isinstance(sample_bits,tuple) else int(sample_bits)
    automatic=bool(params.get('input_auto',False))
    profile=image.info.get('icc_profile')
    if bits > 8 or image.mode in ('I','F','I;16','I;16B','I;16L'):
        if automatic and profile:
            raise ValueError('16-bit embedded ICC conversion is not supported at full precision. Select manual input interpretation with the correct color space.')
        if image.format == 'TIFF' and image.tag_v2.get(262, 1) not in (1, 2):
            raise ValueError('High-bit-depth TIFF requires grayscale or RGB photometric interpretation.')
        import cv2
        raw = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16:
            raise ValueError('High-bit-depth input must be an unsigned 16-bit PNG or TIFF.')
        if raw.ndim == 2:
            raw = np.repeat(raw[..., None], 3, axis=2)
        elif raw.ndim == 3 and raw.shape[2] in (3, 4):
            raw = raw[..., :3][..., ::-1]
        else:
            raise ValueError('Unsupported 16-bit channel layout.')
        name = 'sRGB' if automatic else (params.get('ocio') or {}).get('input', 'sRGB')
        return PrecisionImage(raw.astype(np.float32) * (255.0 / 65535.0)), dict(
            mode='auto' if automatic else 'manual', profile=name, bit_depth=16,
            status='16-bit — assumed sRGB' if automatic else '16-bit — manual input interpretation',
            assumed=automatic)
    if automatic and profile:
        try:
            source=ImageCms.ImageCmsProfile(io.BytesIO(profile))
            name=ImageCms.getProfileDescription(source).strip()
            image=ImageCms.profileToProfile(image,source,ImageCms.createProfile('sRGB'),outputMode='RGB')
        except Exception as exc:
            raise ValueError('Embedded ICC profile could not be converted. Choose manual input interpretation to override.') from exc
        return image,dict(mode='auto',profile=name,status='Embedded ICC converted to sRGB',assumed=False)
    if automatic:
        return image.convert('RGB'),dict(mode='auto',profile='sRGB',status='No embedded ICC — assumed sRGB',assumed=True)
    return image.convert('RGB'),dict(mode='manual',profile=(params.get('ocio') or {}).get('input','sRGB'),
                                   status='Manual input interpretation',assumed=False)

BUILTIN = 'studio-config-v2.2.0_aces-v1.3_ocio-v2.4'


def load_config(path=''):
    try:
        import PyOpenColorIO as ocio
    except ImportError as exc:
        raise ValueError('OCIO is optional. Install rudra-hdr[ocio] in the Studio Python environment.') from exc
    if path:
        file = Path(path).expanduser().resolve(strict=True)
        config = ocio.Config.CreateFromFile(str(file))
    else:
        config = ocio.Config.CreateFromBuiltinConfig(BUILTIN)
    config.validate()
    return config


def describe(path=''):
    config=load_config(path)
    return dict(ok=True,config=path or BUILTIN,cache_id=config.getCacheID(),
                spaces=list(config.getColorSpaceNames()),
                displays={d:list(config.getViews(d)) for d in config.getDisplays()})


def apply(rgb, processor):
    import PyOpenColorIO as ocio
    pixels=np.array(rgb,dtype=np.float32,order='C',copy=True)
    if pixels.ndim!=3 or pixels.shape[2]!=3 or not np.isfinite(pixels).all():
        raise ValueError('OCIO requires finite HxWx3 RGB pixels')
    processor.getDefaultCPUProcessor().apply(ocio.PackedImageDesc(pixels,pixels.shape[1],pixels.shape[0],3))
    if not np.isfinite(pixels).all(): raise ValueError('OCIO produced non-finite pixels')
    return pixels


def transform(rgb, settings, source, target):
    config=load_config(settings.get('config',''))
    if settings.get('cache_id') and settings['cache_id']!=config.getCacheID():
        raise ValueError('OCIO configuration changed; reload it')
    if not config.getColorSpace(source) or not config.getColorSpace(target):
        raise ValueError('Select valid OCIO source and destination color spaces')
    return apply(rgb,config.getProcessor(source,target))


def input_pixels(rgb, params):
    # Auto decoder already produced model-domain sRGB through ICC conversion.
    if params.get('input_auto'): return rgb
    settings=params.get('ocio')
    if not settings: return rgb
    result=transform(rgb,settings,settings['input'],settings['model_srgb'])
    # This model accepts normalized SDR sRGB, not unbounded camera-log/scene HDR.
    return np.clip(result,0,1)


def output_pixels(scene_linear, params):
    settings=params.get('ocio')
    if not settings: raise ValueError('Load and enable an OCIO config first')
    config=load_config(settings.get('config',''))
    expected=settings.get('cache_id')
    if expected and expected!=config.getCacheID():
        raise ValueError('OCIO configuration changed; reload it before rendering')
    source=settings['working_linear']
    if not config.getColorSpace(source): raise ValueError('Select the linear Rec.2020 bridge color space')
    if params['container']=='ocio_exr':
        output=transform(scene_linear,settings,source,settings['output'])
        target=settings['output']
    else:
        import PyOpenColorIO as ocio
        display,view=settings['display'],settings['view']
        if display not in config.getDisplays() or view not in config.getViews(display):
            raise ValueError('Select a valid OCIO display/view')
        tx=ocio.DisplayViewTransform(src=source,display=display,view=view)
        output=apply(scene_linear,config.getProcessor(tx))
        target=f'{display} / {view}'
    metadata=dict(config=settings.get('config') or BUILTIN,cache_id=config.getCacheID(),
                  input=settings['input'],model_srgb=settings['model_srgb'],
                  working_linear=source,output=target,linear_scale='1 = 203 nits',
                  config_sha256=hashlib.sha256(config.serialize().encode()).hexdigest())
    return output,metadata
