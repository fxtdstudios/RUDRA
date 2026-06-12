"""Lightweight validation metrics for RUDRA. HDR-VDP-3 and LPIPS should be added externally."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Existing metrics (unchanged)
# ---------------------------------------------------------------------------


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB."""
    mse = F.mse_loss(pred, target).clamp(min=1e-12)
    return 20.0 * torch.log10(torch.tensor(data_range, device=pred.device, dtype=pred.dtype)) - 10.0 * torch.log10(mse)


def highlight_reconstruction_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
    ev_threshold: float = 2.0,
    rel_tol: float = 0.05,
) -> torch.Tensor:
    """Fraction of highlight pixels reconstructed within *rel_tol* relative error.

    Highlights are scene-linear-relative: pixels whose luminance is more than
    *ev_threshold* stops above 0.18 middle grey. (The old absolute-nit threshold
    was unreachable for relative scene-linear data — see spatial_descriptor.)
    Returns 1.0 when no highlight pixels exist.
    """
    y_t = 0.2627 * target[:, 0:1] + 0.6780 * target[:, 1:2] + 0.0593 * target[:, 2:3]
    y_p = 0.2627 * pred[:, 0:1] + 0.6780 * pred[:, 1:2] + 0.0593 * pred[:, 2:3]
    ev = torch.log2(y_t.clamp(min=1e-8) / 0.18)
    mask = ev > ev_threshold
    if not mask.any():
        return torch.ones((), device=target.device, dtype=target.dtype)
    rel_err = (y_p - y_t).abs() / y_t.abs().clamp(min=1e-6)
    return (rel_err[mask] <= rel_tol).float().mean()


def exposure_ev_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute difference in median-based exposure value (EV) between *pred* and *target*."""

    def ev(x: torch.Tensor) -> torch.Tensor:
        y = 0.2627 * x[:, 0:1] + 0.6780 * x[:, 1:2] + 0.0593 * x[:, 2:3]
        med = y.flatten(1).median(dim=-1).values.clamp(min=1e-8)
        return torch.log2(med / 0.18 + 1e-8)

    return (ev(pred) - ev(target)).abs().mean()


# ---------------------------------------------------------------------------
# Delta E 2000
# ---------------------------------------------------------------------------

# Linear RGB → CIE XYZ (D65) for the supported working color spaces. The metric
# must use the same working space the rest of the pipeline operates in, set once
# via config.color_space (review §5), not a hardcoded Rec.2020 assumption.
_RGB_TO_XYZ_BY_SPACE = {
    "rec709": torch.tensor(
        [[0.4123908, 0.3575843, 0.1804808],
         [0.2126390, 0.7151687, 0.0721923],
         [0.0193308, 0.1191948, 0.9505322]], dtype=torch.float64),
    "rec2020": torch.tensor(
        [[0.6369580, 0.1446169, 0.1688810],
         [0.2627002, 0.6779981, 0.0593017],
         [0.0000000, 0.0280727, 1.0609851]], dtype=torch.float64),
    "acescg": torch.tensor(
        [[0.6624542, 0.1340042, 0.1561877],
         [0.2722287, 0.6740818, 0.0536895],
         [-0.0055746, 0.0040607, 1.0103391]], dtype=torch.float64),
}
# Back-compat alias.
_REC2020_TO_XYZ = _RGB_TO_XYZ_BY_SPACE["rec2020"]

# D65 reference white
_D65_X: float = 0.95047
_D65_Y: float = 1.00000
_D65_Z: float = 1.08883


def _linear_rgb_to_lab(rgb: torch.Tensor, color_space: str = "rec2020") -> torch.Tensor:
    """Convert scene-linear RGB (B,3,H,W) in *color_space* to CIE L*a*b*.

    The conversion goes through XYZ (D65 illuminant).  Input values are
    clamped to ≥0 before the conversion to avoid NaNs from negative
    scene-referred values.
    """
    rgb = rgb.clamp(min=0.0).to(torch.float64)
    B, C, H, W = rgb.shape
    # (B,3,H*W)
    flat = rgb.reshape(B, 3, -1)
    mat = _RGB_TO_XYZ_BY_SPACE.get(color_space, _REC2020_TO_XYZ).to(device=rgb.device)  # (3,3)
    xyz = torch.einsum("ij,bjn->bin", mat, flat).reshape(B, 3, H, W)

    # Normalise by D65 reference white
    ref = torch.tensor([_D65_X, _D65_Y, _D65_Z], dtype=torch.float64, device=rgb.device).reshape(1, 3, 1, 1)
    xyz = xyz / ref

    # f(t) piece-wise
    delta: float = 6.0 / 29.0
    delta_sq: float = delta * delta
    delta_cb: float = delta * delta * delta
    f = torch.where(
        xyz > delta_cb,
        xyz.pow(1.0 / 3.0),
        xyz / (3.0 * delta_sq) + 4.0 / 29.0,
    )

    fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1).to(rgb.dtype)


def delta_e_2000(pred: torch.Tensor, target: torch.Tensor, color_space: str = "rec2020") -> torch.Tensor:
    """CIE DE2000 colour difference averaged over all pixels and the batch.

    Args:
        pred: Predicted image in scene-linear ``color_space`` RGB, shape ``(B, 3, H, W)``.
        target: Reference image in scene-linear ``color_space`` RGB, shape ``(B, 3, H, W)``.
        color_space: Working RGB space (rec2020 / rec709 / acescg).

    Returns:
        Scalar tensor with the mean ΔE₀₀ value.
    """
    lab1 = _linear_rgb_to_lab(target, color_space).to(torch.float64)
    lab2 = _linear_rgb_to_lab(pred, color_space).to(torch.float64)

    L1, a1, b1 = lab1[:, 0:1], lab1[:, 1:2], lab1[:, 2:3]
    L2, a2, b2 = lab2[:, 0:1], lab2[:, 1:2], lab2[:, 2:3]

    # Step 1 ------------------------------------------------------------------
    C1_ab = torch.sqrt(a1 * a1 + b1 * b1)
    C2_ab = torch.sqrt(a2 * a2 + b2 * b2)
    C_ab_mean = (C1_ab + C2_ab) / 2.0

    C_ab_mean_pow7 = C_ab_mean.pow(7)
    _25_pow7 = 25.0**7
    G = 0.5 * (1.0 - torch.sqrt(C_ab_mean_pow7 / (C_ab_mean_pow7 + _25_pow7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)

    h1p = torch.atan2(b1, a1p) % (2.0 * math.pi)
    h2p = torch.atan2(b2, a2p) % (2.0 * math.pi)

    # Step 2 ------------------------------------------------------------------
    dLp = L2 - L1
    dCp = C2p - C1p

    h_diff = h2p - h1p
    C_prod = C1p * C2p

    dhp = torch.where(
        C_prod == 0.0,
        torch.zeros_like(h_diff),
        torch.where(
            h_diff.abs() <= math.pi,
            h_diff,
            torch.where(h_diff > math.pi, h_diff - 2.0 * math.pi, h_diff + 2.0 * math.pi),
        ),
    )
    dHp = 2.0 * torch.sqrt(C_prod) * torch.sin(dhp / 2.0)

    # Step 3 ------------------------------------------------------------------
    Lp_mean = (L1 + L2) / 2.0
    Cp_mean = (C1p + C2p) / 2.0

    hp_sum = h1p + h2p
    hp_diff_abs = (h1p - h2p).abs()

    hp_mean = torch.where(
        C_prod == 0.0,
        hp_sum,
        torch.where(
            hp_diff_abs <= math.pi,
            hp_sum / 2.0,
            torch.where(hp_sum < 2.0 * math.pi, (hp_sum + 2.0 * math.pi) / 2.0, (hp_sum - 2.0 * math.pi) / 2.0),
        ),
    )

    T = (
        1.0
        - 0.17 * torch.cos(hp_mean - math.radians(30))
        + 0.24 * torch.cos(2.0 * hp_mean)
        + 0.32 * torch.cos(3.0 * hp_mean + math.radians(6))
        - 0.20 * torch.cos(4.0 * hp_mean - math.radians(63))
    )

    Lp_mean_50_sq = (Lp_mean - 50.0).pow(2)
    S_L = 1.0 + 0.015 * Lp_mean_50_sq / torch.sqrt(20.0 + Lp_mean_50_sq)
    S_C = 1.0 + 0.045 * Cp_mean
    S_H = 1.0 + 0.015 * Cp_mean * T

    Cp_mean_pow7 = Cp_mean.pow(7)
    R_C = 2.0 * torch.sqrt(Cp_mean_pow7 / (Cp_mean_pow7 + _25_pow7))
    d_theta = 30.0 * torch.exp(-(((hp_mean * 180.0 / math.pi - 275.0) / 25.0).pow(2)))
    R_T = -torch.sin(2.0 * d_theta * math.pi / 180.0) * R_C

    # Parametric weighting factors (all 1 for the standard formula)
    k_L = k_C = k_H = 1.0

    term_L = dLp / (k_L * S_L)
    term_C = dCp / (k_C * S_C)
    term_H = dHp / (k_H * S_H)

    dE00 = torch.sqrt(term_L.pow(2) + term_C.pow(2) + term_H.pow(2) + R_T * term_C * term_H)
    return dE00.mean().to(pred.dtype)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def _fspecial_gaussian(size: int, sigma: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a 2-D Gaussian kernel (normalised) matching MATLAB's *fspecial('gaussian', ...)*."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel = g.unsqueeze(1) * g.unsqueeze(0)
    return kernel / kernel.sum()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Structural Similarity Index (SSIM) averaged over channels and batch.

    Uses a Gaussian sliding window of *window_size* × *window_size* with
    standard deviation *sigma*.

    Args:
        pred: Predicted image, shape ``(B, C, H, W)``.
        target: Reference image, shape ``(B, C, H, W)``.
        window_size: Side length of the square Gaussian window.
        sigma: Standard deviation of the Gaussian window.
        data_range: Dynamic range of the input (e.g. 1.0 for [0, 1] images).

    Returns:
        Scalar tensor with the mean SSIM value.
    """
    C = pred.shape[1]
    kernel = _fspecial_gaussian(window_size, sigma, device=pred.device, dtype=pred.dtype)
    # (C, 1, window_size, window_size) – groups convolution, one kernel per channel
    window = kernel.unsqueeze(0).unsqueeze(0).expand(C, -1, -1, -1)

    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Advanced Visual Metrics (LPIPS and HDR-VDP-3 Proxy)
# ---------------------------------------------------------------------------

_LPIPS_NET = None

def lpips_metric(pred: torch.Tensor, target: torch.Tensor, device: str = "cuda") -> float:
    """
    Compute LPIPS perceptual similarity metric between pred and target.
    Expects pred and target in scene-linear Rec.2020 RGB, (B, 3, H, W).
    Tone-maps to SDR first (using Reinhard) since LPIPS is display-referred.
    """
    global _LPIPS_NET
    if _LPIPS_NET is None:
        try:
            import lpips
            # Initialize LPIPS network on the specified device. Use vgg for high accuracy.
            _LPIPS_NET = lpips.LPIPS(net="vgg").to(device)
            _LPIPS_NET.eval()
        except Exception as e:
            # Fallback to mean L1 error if LPIPS initialization fails
            return float(F.l1_loss(pred, target).mean().item())
            
    # Tonemap to [0, 1] preview
    def tonemap(x: torch.Tensor) -> torch.Tensor:
        return (x.clamp(min=0.0) / (1.0 + x.clamp(min=0.0))).clamp(0.0, 1.0)
        
    pred_tm = tonemap(pred)
    target_tm = tonemap(target)
    
    # Scale to [-1, 1] as expected by LPIPS
    pred_lpips = pred_tm * 2.0 - 1.0
    target_lpips = target_tm * 2.0 - 1.0
    
    with torch.no_grad():
        loss = _LPIPS_NET(pred_lpips.to(device), target_lpips.to(device))
        return float(loss.mean().item())


def hdr_vdp_proxy(
    pred: torch.Tensor,
    target: torch.Tensor,
    peak_nits: float = 4000.0,
) -> float:
    """Internal HDR difference proxy — NOT HDR-VDP-3 (review §4.1).

    This is a lightweight Naka-Rushton/CSF approximation used only as a fallback
    when ColorVideoVDP (the real metric) is unavailable. Its absolute scale is
    not comparable to published HDR-VDP-3 numbers; for reporting, install
    ``cvvdp`` so ``validation_metrics`` uses the real ``colorvideovdp`` backend.

    Simulates an HDR difference response by performing:
      1. Absolute luminance mapping (Rec.2020 RGB -> Nits).
      2. Retina adaptation (Naka-Rushton compressive response).
      3. Multi-scale visual channel decomposition (Gaussian pyramid).
      4. Contrast Sensitivity Function (CSF) weighting per channel.
      5. Minkowski spatial error pooling (p=4).
      
    Returns a Just Objectionable Difference (JOD) metric between [0.0, 10.0].
    Higher JOD is better; 10.0 represents visually identical images.
    """
    with torch.no_grad():
        # 1. Absolute luminance conversion
        def get_luminance(rgb: torch.Tensor) -> torch.Tensor:
            y = 0.2627 * rgb[:, 0:1] + 0.6780 * rgb[:, 1:2] + 0.0593 * rgb[:, 2:3]
            return y.clamp(min=0.0) * peak_nits
            
        y_pred = get_luminance(pred)
        y_tgt = get_luminance(target)
        
        # 2. Retinal dynamic response (Naka-Rushton adaptation)
        sigma = 10.0  # Adaptation threshold in nits
        exponent = 0.74
        
        def retina_response(y: torch.Tensor) -> torch.Tensor:
            y_pow = y.pow(exponent)
            return y_pow / (y_pow + sigma**exponent)
            
        r_pred = retina_response(y_pred)
        r_tgt = retina_response(y_tgt)
        
        # 3. Multi-scale visual channel decomposition
        diff_r = (r_pred - r_tgt).abs()
        
        scales = [diff_r]
        current = diff_r
        for _ in range(3):
            # Downsample and blur
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            scales.append(current)
            
        # 4. Contrast Sensitivity Function weighting per scale
        # Peak sensitivity around scale 2, dropoff at fine and coarse scales
        weights = [0.15, 0.70, 1.00, 0.40]
        
        # 5. Minkowski spatial difference pooling (p=4)
        mink_sum = torch.zeros((), device=pred.device, dtype=pred.dtype)
        for s_idx, scale in enumerate(scales):
            w = weights[s_idx]
            # Minkowski power p=4
            mink_sum = mink_sum + w * scale.pow(4).mean()
            
        difference = mink_sum.pow(0.25)
        
        # Map to JOD (0.0 to 10.0)
        jod = (10.0 - 5.0 * difference).clamp(0.0, 10.0)
        return float(jod.item())


# ---------------------------------------------------------------------------
# Aggregate helper
# ---------------------------------------------------------------------------


def validation_metrics(pred: torch.Tensor, target: torch.Tensor, color_space: str = "rec2020") -> dict[str, float]:
    """Compute all lightweight and advanced visual validation metrics.

    ``color_space`` selects the working RGB space for the colorimetric metric
    (ΔE2000), and should match config.color_space (review §5).
    """
    pred_tm = pred.clamp(0, 1)
    target_tm = target.clamp(0, 1)
    device = str(pred.device)

    # Real HDR perceptual metric (ColorVideoVDP) when available, else the
    # labeled proxy. Lazy import avoids a circular dependency (review §4.1).
    from .hdrvdp import hdr_vdp3_jod
    hdr_jod, hdr_backend = hdr_vdp3_jod(pred, target, color_space=color_space)

    return {
        "psnr_tm": float(psnr(pred_tm, target_tm).detach().cpu()),
        "ssim_tm": float(ssim(pred_tm, target_tm).detach().cpu()),
        "hra": float(highlight_reconstruction_accuracy(pred, target).detach().cpu()),
        "ev_error": float(exposure_ev_error(pred, target).detach().cpu()),
        "delta_e_2000": float(delta_e_2000(pred, target, color_space).detach().cpu()),
        "lpips": lpips_metric(pred, target, device=device),
        "hdr_vdp3": hdr_jod,
        "hdr_vdp_backend": hdr_backend,
        # Back-compat: keep the proxy key populated with the same value.
        "hdr_vdp_proxy": hdr_jod if hdr_backend == "proxy" else hdr_vdp_proxy(pred, target),
    }
