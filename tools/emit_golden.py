"""Every golden file the native tests read, from the Python they port.

One entry point for principle P4 (docs/NATIVE_ARCHITECTURE.md): run it after
changing any Python a native module ports, commit what changes, and CI runs it
again on every push so a Python change that moves a number fails the native
tests instead of drifting silently.

    python tools/emit_golden.py              # every emitter
    python tools/emit_golden.py decode core  # just these
    python tools/emit_golden.py --except viewer   # all but these

Emitters, in order:
    actions    tools/emit_actions_golden.py    the Studio's menus, actions, keys, shortcut sheet
    copy       tools/emit_copy_golden.py       the Studio page's clipboard texts (needs Playwright)
    core       tools/emit_core_golden.py       baseline, curve, tile weights
    catalog    tools/emit_catalog_golden.py    the Studio server's model discovery (find_checkpoint, /api/checkpoints)
    composite  tools/emit_composite_golden.py  composite, master chain, measure
    decode     tools/emit_decode_golden.py     still decode fixtures
    delivery   tools/emit_delivery_golden.py   grade, HDR10/HLG, sidecars, EXR/ACES/OCIO
    fit        tools/emit_fit_golden.py        the Studio's preview downscale (max_side 1600)
    layout     tools/emit_layout_golden.py     the Studio page's layout, words and states (needs Playwright)
    master     tools/emit_master_golden.py     the Studio's master of three stills
    qc         tools/emit_qc_golden.py         QC checks and report text
    queue      tools/emit_queue_golden.py      queue state files and refusals
    render_plan tools/emit_render_plan_golden.py the Studio server's master targets and refusals
    readouts   tools/emit_readouts_golden.py   the Studio page's probe, Frame panel and bars (needs Playwright)
    scopes     tools/emit_scopes_golden.py     the Studio page's scope drawings and rasters (needs Playwright)
    sequence   tools/emit_sequence_golden.py   sequence open: names, order, messages
    video      tools/emit_video_golden.py      video probe, input contract and clock (needs ffmpeg)
    session    tools/emit_session_golden.py    the Studio page's state, params() and undo (needs Playwright)
    viewer     tools/emit_viewer_golden.py     the browser Studio's viewer (needs Playwright)
    viewport   tools/emit_viewport_golden.py   the Studio's fit, zoom and pan layout (needs Playwright)
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
EMITTERS = {
    "actions": "emit_actions_golden.py",
    "catalog": "emit_catalog_golden.py",
    "copy": "emit_copy_golden.py",
    "core": "emit_core_golden.py",
    "composite": "emit_composite_golden.py",
    "decode": "emit_decode_golden.py",
    "delivery": "emit_delivery_golden.py",
    "fit": "emit_fit_golden.py",
    "layout": "emit_layout_golden.py",
    "master": "emit_master_golden.py",
    "qc": "emit_qc_golden.py",
    "queue": "emit_queue_golden.py",
    "readouts": "emit_readouts_golden.py",
    "render_plan": "emit_render_plan_golden.py",
    "scopes": "emit_scopes_golden.py",
    "sequence": "emit_sequence_golden.py",
    "session": "emit_session_golden.py",
    "video": "emit_video_golden.py",
    "viewer": "emit_viewer_golden.py",
    "viewport": "emit_viewport_golden.py",
}


def main(argv: list[str]) -> int:
    if argv[:1] == ["--except"]:
        skip = argv[1:]
        names = [n for n in EMITTERS if n not in skip]
        unknown = [n for n in skip if n not in EMITTERS]
    else:
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
