"""Video delivery contracts and BT.2100 HLG display-light conversion.

HLG uses the inverse OOTF and OETF in ITU-R BT.2100-2 Table 5, with
zero display black and gamma = 1.2 + 0.42 log10(Lw / 1000).
"""
import numpy as np

from ..hdr10 import master_to_peak, pq_oetf


PROFILES = {
    'hdr10': dict(codec='hevc', encoder='libx265', transfer='smpte2084', pixel_format='yuv420p10le'),
    'hlg': dict(codec='hevc', encoder='libx265', transfer='arib-std-b67', pixel_format='yuv420p10le'),
    'prores422': dict(codec='prores', encoder='prores_ks', transfer='smpte2084', pixel_format='yuv422p10le', profile=2, tag='apcn'),
    'prores422hq': dict(codec='prores', encoder='prores_ks', transfer='smpte2084', pixel_format='yuv422p10le', profile=3, tag='apch'),
    'prores4444': dict(codec='prores', encoder='prores_ks', transfer='smpte2084', pixel_format='yuv444p10le', profile=4, tag='ap4h'),
}

HLG_A = 0.17883277
HLG_B = 1 - 4 * HLG_A
HLG_C = 0.5 - HLG_A * np.log(4 * HLG_A)
LUMA = np.array([0.2627, 0.6780, 0.0593])


def hlg_oetf(scene):
    scene = np.maximum(np.asarray(scene, dtype=np.float64), 0)
    return np.where(scene <= 1/12, np.sqrt(3*scene),
                    HLG_A*np.log(np.maximum(12*scene-HLG_B, 1e-12))+HLG_C)


def hlg_eotf(code, peak_nits=1000):
    code = np.asarray(code, dtype=np.float64)
    scene = np.where(code <= .5, code**2/3,
                     (np.exp((code-HLG_C)/HLG_A)+HLG_B)/12)
    gamma = 1.2 + .42*np.log10(peak_nits/1000)
    luminance = np.sum(scene*LUMA, axis=-1, keepdims=True)
    return scene*np.maximum(luminance, 1e-12)**(gamma-1)*peak_nits


def encode_master(rgb_normalized, profile, peak_nits=1000, knee_nits=None):
    mastered = master_to_peak(rgb_normalized, peak_nits, knee_nits)
    if profile != 'hlg':
        return pq_oetf(mastered), mastered
    # Display light -> scene light. Apply gamma to luminance, not each channel.
    gamma = 1.2 + .42*np.log10(peak_nits/1000)
    display = mastered.astype(np.float64)/peak_nits
    luminance = np.sum(display*LUMA, axis=-1, keepdims=True)
    scene = display*np.maximum(luminance, 1e-12)**((1-gamma)/gamma)
    # Saturated display colours can lie outside legal HLG scene RGB. Reduce
    # them together to preserve RGB ratios instead of clipping channels.
    scene /= np.maximum(1, np.max(scene, axis=-1, keepdims=True))
    code = np.clip(hlg_oetf(scene), 0, 1)
    return code.astype(np.float32), hlg_eotf(code, peak_nits).astype(np.float32)
