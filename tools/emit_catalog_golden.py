"""Golden model discovery for the native checkpoint manager (Phase 3 step 10).

The oracle is ui/server.py: find_checkpoint (which model a bare start loads)
and loadable_models (what /api/checkpoints lists, in order). Each case lays
out training roots (RUDRA_CHECKPOINT_ROOTS, a run folder per checkpoint with
an age in hours) and the repo's checkpoints/ folder with its models.json, in
a temporary folder, and records what the server picks and lists. The native
catalog (engine/model_catalog) gets the same layout as model packages, one
folder per checkpoint whose manifest names it as its source file, and must
pick and list the same ones.

    python tools/emit_catalog_golden.py   # writes native/tests/golden/catalog/cases.json
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = REPO / "native" / "tests" / "golden" / "catalog"

REG = {"default": "shadow_v1.pt", "models": [
    {"file": "shadow_v1.pt", "kind": "sdr2hdr", "title": "v5 + shadow gate (shipped)", "note": "the paper's"},
    {"file": "shadow_s2.pt", "kind": "sdr2hdr", "title": "shadow gate, seed 2", "note": "best on PU21"},
    {"file": "image_v5.pt", "kind": "sdr2hdr", "title": "v5 backbone, no gate", "note": ""},
    {"file": "temporal_v1.pt", "kind": "temporal", "title": "temporal refiner", "note": "not an SDR2HDRNet"},
]}

# (name, training roots: [[{run, file, age}]], repo registry or None, repo files)
CASES = [
    ("repo_default", [], REG, ["shadow_v1.pt", "shadow_s2.pt", "image_v5.pt", "temporal_v1.pt"]),
    ("repo_default_missing", [], REG, ["image_v5.pt", "shadow_s2.pt", "temporal_v1.pt"]),
    ("repo_only_temporal", [], REG, ["temporal_v1.pt"]),
    ("repo_unlisted_files", [], REG, ["other.pt", "image_v5.pt"]),
    ("repo_no_registry", [], None, ["shadow_v1.pt"]),
    ("training_newest", [[{"run": "a", "file": "best.pt", "age": 5}, {"run": "b", "file": "best.pt", "age": 1},
                          {"run": "c", "file": "best.pt", "age": 3}]], REG, ["shadow_v1.pt"]),
    ("training_shipped_wins", [[{"run": "a", "file": "best.pt", "age": 1},
                                {"run": "b", "file": "shipped_v7.pt", "age": 9}]], REG, ["shadow_v1.pt"]),
    ("training_newest_shipped", [[{"run": "a", "file": "shipped_x.pt", "age": 4},
                                  {"run": "b", "file": "shipped_y.pt", "age": 2},
                                  {"run": "c", "file": "best.pt", "age": 0.5}]], REG, []),
    ("training_other_names_ignored", [[{"run": "a", "file": "last.pt", "age": 1}]], REG, ["shadow_s2.pt"]),
    ("first_root_with_anything", [[], [{"run": "r", "file": "best.pt", "age": 7}],
                                  [{"run": "s", "file": "best.pt", "age": 1}]], REG, ["shadow_v1.pt"]),
    ("nothing_anywhere", [[]], None, []),
]


def main() -> int:
    import ui.server as server

    results = []
    for name, training, reg, files in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "repo"
            (repo / "checkpoints").mkdir(parents=True)
            now = time.time()
            roots = []
            for i, runs in enumerate(training):
                root = base / f"train{i}"
                root.mkdir()
                roots.append(root)
                for r in runs:
                    p = root / r["run"] / r["file"]
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(b"pt")
                    t = now - r["age"] * 3600
                    os.utime(p, (t, t))
            if reg is not None:
                (repo / "checkpoints" / "models.json").write_text(json.dumps(reg), encoding="utf-8")
            for f in files:
                (repo / "checkpoints" / f).write_bytes(b"pt")
            server.REPO = repo
            server.REGISTRY = repo / "checkpoints" / "models.json"
            # An empty RUDRA_CHECKPOINT_ROOTS means hdrdata/checkpoints, which
            # does not exist here: the same as no training roots.
            os.environ["RUDRA_CHECKPOINT_ROOTS"] = os.pathsep.join(str(r) for r in roots)
            chosen = server.find_checkpoint(None)
            listed = server.loadable_models()
            pick = None
            if chosen is not None:
                if chosen.parent == repo / "checkpoints":
                    pick = {"root": len(training), "run": None, "file": chosen.name}
                else:
                    idx = next(i for i, r in enumerate(roots) if chosen.parent.parent == r)
                    pick = {"root": idx, "run": chosen.parent.name, "file": chosen.name}
            results.append({"name": name, "training": training, "registry": reg, "files": files,
                            "chosen": pick, "default": server.registry()["default"],
                            "listed": [{"file": m["file"], "title": m.get("title", ""), "note": m.get("note", "")}
                                       for m in listed]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cases.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "ui/server.py find_checkpoint and loadable_models", "cases": results}, f, indent=1)
        f.write("\n")
    print(f"catalog: {len(results)} cases -> {OUT / 'cases.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
