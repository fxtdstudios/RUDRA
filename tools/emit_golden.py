"""Every golden file the native tests read, from the Python they port.

One entry point for principle P4 (docs/NATIVE_ARCHITECTURE.md): run it after
changing any Python a native module ports, commit what changes, and CI runs it
again on every push so a Python change that moves a number fails the native
tests instead of drifting silently.

    python tools/emit_golden.py              # every emitter
    python tools/emit_golden.py decode core  # just these

Emitters, in order:
    core       tools/emit_core_golden.py       baseline, curve, tile weights
    composite  tools/emit_composite_golden.py  composite, master chain, measure
    decode     tools/emit_decode_golden.py     still decode fixtures
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
EMITTERS = {
    "core": "emit_core_golden.py",
    "composite": "emit_composite_golden.py",
    "decode": "emit_decode_golden.py",
}


def main(argv: list[str]) -> int:
    names = argv or list(EMITTERS)
    unknown = [n for n in names if n not in EMITTERS]
    if unknown:
        print(f"unknown emitter(s): {', '.join(unknown)}; known: {', '.join(EMITTERS)}")
        return 2
    for name in names:
        print(f"== {name}")
        saved = sys.argv
        sys.argv = [EMITTERS[name]]
        try:
            runpy.run_path(str(TOOLS / EMITTERS[name]), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"{name} failed with exit code {e.code}")
                return int(e.code) if isinstance(e.code, int) else 1
        finally:
            sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
