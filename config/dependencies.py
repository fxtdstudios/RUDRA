"""Runtime dependency checking — validates that required and optional packages exist."""
from __future__ import annotations

import importlib.util
import logging
import sys
from typing import Iterable, Optional, Sequence, Tuple

from radiance.config.constants import PACKAGE_NAME
from radiance.config.env import ENV


class DependencySpec:
    """A runtime dependency and the feature area it unlocks."""

    def __init__(
        self,
        module_name: str,
        display_name: str,
        feature: str,
        install_hint: str,
        required: bool = False,
    ):
        self.module_name = module_name
        self.display_name = display_name
        self.feature = feature
        self.install_hint = install_hint
        self.required = required


CORE_DEPENDENCIES: Tuple[DependencySpec, ...] = (
    DependencySpec("torch", "torch", "tensor processing", "pip install torch", True),
    DependencySpec("numpy", "numpy", "array processing", "pip install numpy", True),
    DependencySpec("PIL.Image", "Pillow", "image I/O", "pip install Pillow", True),
    DependencySpec("aiohttp", "aiohttp", "async HTTP server", "pip install aiohttp", True),
)

OPTIONAL_DEPENDENCIES: Tuple[DependencySpec, ...] = (
    DependencySpec("OpenEXR", "OpenEXR", "EXR file support", "pip install OpenEXR"),
    DependencySpec("transformers", "transformers", "Depth Anything V2", "pip install transformers"),
    DependencySpec("colour", "colour-science", "advanced color science", "pip install colour-science"),
    DependencySpec("defusedxml", "defusedxml", "secure CDL XML parsing", "pip install defusedxml"),
)


def module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def missing_dependencies(
    dependencies: Iterable[DependencySpec],
) -> Tuple[DependencySpec, ...]:
    return tuple(spec for spec in dependencies if not module_available(spec.module_name))


def validate_runtime_dependencies(
    logger: Optional[logging.Logger] = None,
    core_dependencies: Sequence[DependencySpec] = CORE_DEPENDENCIES,
    optional_dependencies: Sequence[DependencySpec] = OPTIONAL_DEPENDENCIES,
) -> bool:
    active_logger = logger or logging.getLogger(PACKAGE_NAME)
    missing_core = missing_dependencies(core_dependencies)
    missing_optional = missing_dependencies(optional_dependencies)

    for spec in missing_core:
        active_logger.error(
            "Missing required dependency %s for %s. Install with: %s",
            spec.display_name,
            spec.feature,
            spec.install_hint,
        )

    if missing_optional:
        active_logger.info("Optional Radiance dependencies not installed:")
        for spec in missing_optional:
            active_logger.info(
                "  %s: %s (install: %s)",
                spec.display_name,
                spec.feature,
                spec.install_hint,
            )
    else:
        active_logger.debug("All optional Radiance dependencies are available")

    return not missing_core
