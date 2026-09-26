"""Explicit output transforms for Studio still/sequence delivery."""
import numpy as np
from PIL import Image, ImageCms

PRESETS = {'aces': ('exr', 'ACES 2065-1 (AP0)'),
           'ocio_exr': ('exr', 'OCIO color-space EXR'),
           'ocio_view_png': ('png', 'OCIO display/view PNG 8-bit'),
           'linear': ('exr', 'scene-linear Rec.2020'),
           'srgb_png': ('png', 'sRGB SDR — PNG 8-bit'),
           'srgb_tiff': ('tif', 'sRGB SDR — TIFF 8-bit')}


def preset(params):
    key = params.get('container', 'aces')
    if key not in PRESETS:
        raise ValueError('Unsupported export preset')
    return PRESETS[key]


def srgb_pixels(nits):
    """Rec.2020 nits -> luminance Reinhard (203 nit scale) -> sRGB.

    SDR mapping is explicit, fixed across sequence frames, with final gamut clipping.
    This is an export transform, not a reconstruction of missing source detail.
    """
    rgb = np.maximum(np.asarray(nits, dtype=np.float64), 0) / 203.0
    if not np.isfinite(rgb).all():
        raise ValueError('Non-finite export pixels')
    y = rgb @ np.array([.2627, .6780, .0593])
    rgb = rgb / (1 + y[..., None])
    matrix = np.array([[1.660491,-.587641,-.072850],
                       [-.124550,1.132900,-.008349],
                       [-.018151,-.100579,1.118730]])
    linear = np.clip(rgb @ matrix.T, 0, 1)
    encoded = np.where(linear <= .0031308, linear * 12.92,
                       1.055 * linear**(1/2.4) - .055)
    return np.rint(encoded * 255).astype(np.uint8)


def write_srgb(nits, path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.fromarray(srgb_pixels(nits)).save(path, icc_profile=profile)
