"""RUDRA delivery layer — torch-free mastering, metadata, and benchmarking.

Modules (import them directly; this package pulls in nothing heavy):
  - colorspace: exact primary conversions (Rec.709/2020, P3-D65, AP0/AP1)
  - exr:        dependency-free uncompressed EXR write/read (ACES container)
  - aces:       ACES 2065-1 EXR export + OCIO v2 config generation
  - metadata:   Dolby Vision L1 / HDR10+ dynamic metadata, MaxCLL/MaxFALL
  - controls:   artist grade controls (EV, regions, qualifiers, knee, peak)
  - bench:      PU21-PSNR / CVVDP paired benchmark harness
  - cli:        the ``rudra`` command tying it together

Everything here runs with numpy alone, by design: render nodes and colorist
machines should not need a CUDA torch install to master, measure, or remux
what the models produced.
"""

from . import colorspace, exr, aces, metadata, controls, bench  # noqa: F401

__all__ = ["colorspace", "exr", "aces", "metadata", "controls", "bench"]
