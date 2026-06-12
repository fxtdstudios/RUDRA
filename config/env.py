"""Environment variable access layer — single point of env-var interaction.

Every module in Radiance should read env vars through this module rather than
calling os.environ directly. This makes env vars discoverable, testable, and
prevents scattered magic-string lookups.
"""
from __future__ import annotations

import os
from typing import MutableMapping, Optional


class ENV:
    """Well-known environment variable names used across Radiance.

    Usage:  value = os.environ.get(ENV.RADIANCE_TURBO_DECODER, "")
    """

    RADIANCE_TURBO_DECODER = "RADIANCE_TURBO_DECODER"
    RADIANCE_CACHE_SIZE = "RADIANCE_CACHE_SIZE"
    RADIANCE_LICENSE_KEY = "RADIANCE_LICENSE_KEY"
    OCIO = "OCIO"

    # Internal flags that must be set before OpenCV/OpenMP-backed imports.
    KMP_DUPLICATE_LIB_OK = "KMP_DUPLICATE_LIB_OK"
    OPENCV_IO_ENABLE_OPENEXR = "OPENCV_IO_ENABLE_OPENEXR"


RUNTIME_ENV_DEFAULTS: dict[str, str] = {
    ENV.KMP_DUPLICATE_LIB_OK: "TRUE",
    ENV.OPENCV_IO_ENABLE_OPENEXR: "1",
}


def configure_runtime_environment(
    environ: Optional[MutableMapping[str, str]] = None,
) -> MutableMapping[str, str]:
    target = os.environ if environ is None else environ
    for key, value in RUNTIME_ENV_DEFAULTS.items():
        target.setdefault(key, value)
    return target


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def get_env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")
