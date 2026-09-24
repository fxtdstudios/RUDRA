"""Golden render plans for the native Deliver tab (Phase 3 step 9).

The oracle is ui/server.py master_targets, the paths /api/master/plan hands
back and run_master writes: the folder resolved, the name checked, the image
or the numbered sequence, and every refusal with its message (a relative
folder, a bad name, a bad range, an image of more than one frame, a file or
a sidecar that exists already). Paths are recorded under <root>, the
temporary folder the cases run in; the native test rebuilds the same files.

    python tools/emit_render_plan_golden.py   # writes native/tests/golden/render_plan/plans.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "native" / "tests" / "golden" / "render_plan"

# (name, params with <root>, files to create first under <root>)
CASES = [
    ("image", {"render_dir": "<root>/out", "render_name": "master"}, []),
    ("image_default_name", {"render_dir": "<root>/out"}, []),
    ("sequence_three", {"render_dir": "<root>/out", "render_name": "shot_010", "render_mode": "sequence",
                        "frame_start": 1, "render_count": 3}, []),
    ("sequence_from_zero", {"render_dir": "<root>/out", "render_name": "a.b-c_d", "render_mode": "sequence",
                            "frame_start": 0, "render_count": 2}, []),
    ("sequence_big_start", {"render_dir": "<root>/out", "render_name": "m", "render_mode": "sequence",
                            "frame_start": 1234567, "render_count": 2}, []),
    ("folder_with_dots", {"render_dir": "<root>/out/../out/./deep", "render_name": "master"}, []),
    ("folder_spaces", {"render_dir": "  <root>/out  ", "render_name": "  master  "}, []),
    ("relative_folder", {"render_dir": "renders/out", "render_name": "master"}, []),
    ("empty_folder", {"render_dir": "", "render_name": "master"}, []),
    ("bad_name_space", {"render_dir": "<root>/out", "render_name": "my master"}, []),
    ("bad_name_slash", {"render_dir": "<root>/out", "render_name": "a/b"}, []),
    ("bad_name_dotdot", {"render_dir": "<root>/out", "render_name": ".."}, []),
    ("empty_name", {"render_dir": "<root>/out", "render_name": "   "}, []),
    ("count_zero", {"render_dir": "<root>/out", "render_mode": "sequence", "render_count": 0}, []),
    ("start_negative", {"render_dir": "<root>/out", "render_mode": "sequence", "frame_start": -1}, []),
    ("too_many", {"render_dir": "<root>/out", "render_mode": "sequence", "render_count": 100001}, []),
    ("past_the_end", {"render_dir": "<root>/out", "render_mode": "sequence", "frame_start": 99999999,
                      "render_count": 2}, []),
    ("image_of_two", {"render_dir": "<root>/out", "render_count": 2}, []),
    ("exr_exists", {"render_dir": "<root>/out", "render_name": "master"}, ["out/master.exr"]),
    ("sidecar_exists", {"render_dir": "<root>/out", "render_name": "master"}, ["out/master.json"]),
    ("sequence_one_exists", {"render_dir": "<root>/out", "render_name": "s", "render_mode": "sequence",
                             "frame_start": 1, "render_count": 3}, ["out/s.000002.json"]),
]


def main() -> int:
    from ui.server import master_targets

    results = []
    for name, params, files in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            for f in files:
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                (root / f).write_text("")
            p = {k: (v.replace("<root>", root.as_posix()) if isinstance(v, str) else v) for k, v in params.items()}
            try:
                paths = [Path(x).as_posix().replace(root.as_posix(), "<root>") for x in master_targets(p)]
                results.append({"name": name, "params": params, "files": files, "paths": paths})
            except ValueError as exc:
                msg = str(exc).replace(root.as_posix(), "<root>")
                results.append({"name": name, "params": params, "files": files, "error": msg})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "plans.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "ui/server.py master_targets", "cases": results}, f, indent=1)
        f.write("\n")
    print(f"render_plan: {len(results)} cases -> {OUT / 'plans.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
