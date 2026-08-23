"""RUDRA training scripts package.

This __init__ exists so `training/` is a REGULAR package, not a namespace
package: namespace packages lose the import race to any installed module
that happens to be called `training` (common in ML environments — the
comfyui env shadowed it and broke `tests/test_sdr2hdr.py` collection,
2026-08-23). A regular package at sys.path[0] always wins.

Scripts here are still run directly (`python training/<script>.py`); this
file deliberately imports nothing so it cannot slow that down or create
import cycles.
"""
