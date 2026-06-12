"""Self-contained, round-trip-exact camera log transfer curves for RUDRA.

This module exists to fix a correctness bug (review §3.1): training targets were
encoded to log with one set of curves (an external ``color_utils`` module that is
not shipped in this package) while ``normalization.py`` decoded them with a crude
generic approximation. The forward/inverse pair was therefore not an identity, so
the "scene-linear ground truth" fed to the losses — and the dynamic-range
descriptor for any log-format input — was physically wrong.

Every curve here defines ``*_to_linear`` and ``linear_to_*`` as an analytic
inverse pair. ``tests/test_color_curves.py`` asserts ``decode(encode(x)) ≈ x`` so
self-consistency is guaranteed by construction, independent of any external
module. Formulas follow the published vendor specifications.

All functions operate on ``torch.Tensor`` of any shape and are differentiable.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-8


def _log10(x: torch.Tensor) -> torch.Tensor:
    return torch.log10(x.clamp(min=_EPS))


def _log2(x: torch.Tensor) -> torch.Tensor:
    return torch.log2(x.clamp(min=_EPS))


# ─────────────────────────────────────────────────────────────────────────────
#  ARRI LogC3 (EI 800)
# ─────────────────────────────────────────────────────────────────────────────
_LOGC3 = dict(cut=0.010591, a=5.555556, b=0.052272, c=0.247190, d=0.385537, e=5.367655, f=0.092809)


def linear_to_logc3(x: torch.Tensor) -> torch.Tensor:
    p = _LOGC3
    log_part = p["c"] * _log10(p["a"] * x + p["b"]) + p["d"]
    lin_part = p["e"] * x + p["f"]
    return torch.where(x > p["cut"], log_part, lin_part)


def logc3_to_linear(t: torch.Tensor) -> torch.Tensor:
    p = _LOGC3
    t_cut = p["e"] * p["cut"] + p["f"]
    log_part = (torch.pow(10.0, (t - p["d"]) / p["c"]) - p["b"]) / p["a"]
    lin_part = (t - p["f"]) / p["e"]
    return torch.where(t > t_cut, log_part, lin_part)


# ─────────────────────────────────────────────────────────────────────────────
#  ARRI LogC4 (2022 specification)
# ─────────────────────────────────────────────────────────────────────────────
_LC4_a = (2.0 ** 18 - 16.0) / 117.45
_LC4_b = (1023.0 - 95.0) / 1023.0
_LC4_c = 95.0 / 1023.0
_LC4_s = (7.0 * math.log(2.0) * 2.0 ** (7.0 - 14.0 * _LC4_c / _LC4_b)) / (_LC4_a * _LC4_b)
_LC4_t = (2.0 ** (14.0 * (-_LC4_c / _LC4_b) + 6.0) - 64.0) / _LC4_a


def linear_to_logc4(x: torch.Tensor) -> torch.Tensor:
    log_part = (_log2(_LC4_a * x + 64.0) - 6.0) / 14.0 * _LC4_b + _LC4_c
    lin_part = (x - _LC4_t) / _LC4_s
    return torch.where(x >= _LC4_t, log_part, lin_part)


def logc4_to_linear(y: torch.Tensor) -> torch.Tensor:
    p = 14.0 * (y - _LC4_c) / _LC4_b + 6.0
    log_part = (torch.pow(2.0, p) - 64.0) / _LC4_a
    lin_part = y * _LC4_s + _LC4_t
    return torch.where(y >= 0.0, log_part, lin_part)


# ─────────────────────────────────────────────────────────────────────────────
#  Sony S-Log3
# ─────────────────────────────────────────────────────────────────────────────
_SLOG3_break = 0.01125000
_SLOG3_y_break = 171.2102946929


def linear_to_slog3(x: torch.Tensor) -> torch.Tensor:
    log_part = (420.0 + _log10((x + 0.01) / (0.18 + 0.01)) * 261.5) / 1023.0
    lin_part = (x * (_SLOG3_y_break - 95.0) / _SLOG3_break + 95.0) / 1023.0
    return torch.where(x >= _SLOG3_break, log_part, lin_part)


def slog3_to_linear(y: torch.Tensor) -> torch.Tensor:
    code = y * 1023.0
    log_part = torch.pow(10.0, (code - 420.0) / 261.5) * (0.18 + 0.01) - 0.01
    lin_part = (code - 95.0) * _SLOG3_break / (_SLOG3_y_break - 95.0)
    return torch.where(code >= _SLOG3_y_break, log_part, lin_part)


# ─────────────────────────────────────────────────────────────────────────────
#  Panasonic V-Log
# ─────────────────────────────────────────────────────────────────────────────
_VLOG = dict(cut=0.01, b=0.00873, c=0.241514, d=0.598206)
_VLOG_ycut = 5.6 * _VLOG["cut"] + 0.125


def linear_to_vlog(x: torch.Tensor) -> torch.Tensor:
    p = _VLOG
    log_part = p["c"] * _log10(x + p["b"]) + p["d"]
    lin_part = 5.6 * x + 0.125
    return torch.where(x >= p["cut"], log_part, lin_part)


def vlog_to_linear(y: torch.Tensor) -> torch.Tensor:
    p = _VLOG
    log_part = torch.pow(10.0, (y - p["d"]) / p["c"]) - p["b"]
    lin_part = (y - 0.125) / 5.6
    return torch.where(y >= _VLOG_ycut, log_part, lin_part)


# ─────────────────────────────────────────────────────────────────────────────
#  RED Log3G10 (with 0.01 black offset)
# ─────────────────────────────────────────────────────────────────────────────
_L3G10 = dict(a=0.224282, b=155.975327, c=0.01, g=15.1927)


def linear_to_log3g10(x: torch.Tensor) -> torch.Tensor:
    p = _L3G10
    xs = x + p["c"]
    log_part = p["a"] * _log10(xs * p["b"] + 1.0)
    lin_part = xs * p["g"]
    return torch.where(xs >= 0.0, log_part, lin_part)


def log3g10_to_linear(y: torch.Tensor) -> torch.Tensor:
    p = _L3G10
    log_part = (torch.pow(10.0, y / p["a"]) - 1.0) / p["b"]
    lin_part = y / p["g"]
    xs = torch.where(y >= 0.0, log_part, lin_part)
    return xs - p["c"]


# ─────────────────────────────────────────────────────────────────────────────
#  DaVinci Intermediate
# ─────────────────────────────────────────────────────────────────────────────
_DVI = dict(a=0.0075, b=7.0, c=0.07329248, m=10.44426855, lin_cut=0.00262409)
_DVI_logcut = _DVI["m"] * _DVI["lin_cut"]


def linear_to_davinci(x: torch.Tensor) -> torch.Tensor:
    p = _DVI
    log_part = (_log2(x + p["a"]) + p["b"]) * p["c"]
    lin_part = x * p["m"]
    return torch.where(x > p["lin_cut"], log_part, lin_part)


def davinci_to_linear(y: torch.Tensor) -> torch.Tensor:
    p = _DVI
    log_part = torch.pow(2.0, y / p["c"] - p["b"]) - p["a"]
    lin_part = y / p["m"]
    return torch.where(y > _DVI_logcut, log_part, lin_part)


# ─────────────────────────────────────────────────────────────────────────────
#  Registry — keyed by the format names used in rudra.config.FORMAT_NAMES
# ─────────────────────────────────────────────────────────────────────────────
LINEAR_TO_LOG = {
    "logc3": linear_to_logc3,
    "logc4": linear_to_logc4,
    "slog3": linear_to_slog3,
    "vlog": linear_to_vlog,
    "log3g10": linear_to_log3g10,
    "davinci": linear_to_davinci,
}

LOG_TO_LINEAR = {
    "logc3": logc3_to_linear,
    "logc4": logc4_to_linear,
    "slog3": slog3_to_linear,
    "vlog": vlog_to_linear,
    "log3g10": log3g10_to_linear,
    "davinci": davinci_to_linear,
}

# Human-readable curve names (as used in model_map / config) → registry key.
CURVE_NAME_TO_KEY = {
    "ARRI LogC4": "logc4",
    "ARRI LogC3": "logc3",
    "Sony S-Log3": "slog3",
    "Panasonic V-Log": "vlog",
    "RED Log3G10": "log3g10",
    "DaVinci Intermediate": "davinci",
}


def encode_linear(linear: torch.Tensor, curve_key: str) -> torch.Tensor:
    """Encode scene-linear → log code for the named curve key."""
    fn = LINEAR_TO_LOG.get(curve_key)
    if fn is None:
        raise KeyError(f"Unknown log curve key: {curve_key!r}. Known: {sorted(LINEAR_TO_LOG)}")
    return fn(linear)


def decode_to_linear(code: torch.Tensor, curve_key: str) -> torch.Tensor:
    """Decode log code → scene-linear for the named curve key."""
    fn = LOG_TO_LINEAR.get(curve_key)
    if fn is None:
        raise KeyError(f"Unknown log curve key: {curve_key!r}. Known: {sorted(LOG_TO_LINEAR)}")
    return fn(code)
