"""Golden video predictions for the native pipeline (Phase 4, step 3).

The oracle is rudra/video.py Predictor.predict and ShadowSmoother on the
shipped checkpoint, eager PyTorch on CPU: the SDR canonicalised from the
clip's transfer and range, converted to Rec.2020 when it is not, the 36 x 64
area thumbnail the cut detector compares, the per-frame shadow weight and
residual scale, the smoothed weight, and the frame run whole or in tiles with
the video path's own linear feathers.

The inputs are the video goldens' clips (native/tests/golden/video/clips),
decoded with the Python's own decoder command and read_frame; the raw 16-bit
frames are saved so the native test starts from the same bytes on any ffmpeg.
Four sequences cover the settings: tiled with a hard cut half way (two clips
back to back) and retention 0.5, whole-frame BT.2020 with no smoothing, RGB
with heavy retention and cuts off, and alpha ProRes in tiles that do not
divide the frame. The pure parts are recorded on their own too: torch's area
resize on odd shapes and torch.linspace as the feathers use it.

    python tools/emit_video_predict_golden.py   # writes native/tests/golden/video_predict/
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

VIDEO = REPO / "native" / "tests" / "golden" / "video"
OUT = REPO / "native" / "tests" / "golden" / "video_predict"
CHECKPOINT = REPO / "checkpoints" / "sdr2hdr_shadow_v1.pt"

# name: (segments of (clip, first frame, count)), tile size, overlap, retention, cut threshold
SEQUENCES = {
    "tiled_cut": ([("h264_709.mp4", 0, 4), ("bars_709.mp4", 0, 3)], 32, 8, 0.5, 0.15),
    "whole_2020": ([("bt2020.mkv", 0, 3)], 0, 64, 0.0, 0.15),
    "rgb_retained": ([("rgb_png.mov", 0, 3)], 512, 64, 0.9, 1.0),
    "alpha_odd_tiles": ([("prores4444_alpha.mov", 0, 2)], 24, 4, 0.25, 0.15),
}
AREA_CASES = [((3, 50, 90), (36, 64)), ((3, 100, 7), (36, 64)), ((3, 36, 64), (36, 64)), ((3, 20, 30), (36, 64))]
LINSPACE = [1, 2, 3, 4, 7, 8, 12, 32, 64]


def decoded_frames(clip: str) -> tuple[list[np.ndarray], dict]:
    """The clip through the Python's decoder command and read_frame."""
    from rudra.video import read_frame
    decode = json.loads((VIDEO / "decode.json").read_text(encoding="utf-8"))
    case = next(c for c in decode["cases"] if c["clip"] == clip)
    index = json.loads((VIDEO / "index.json").read_text(encoding="utf-8"))
    ok = next(c for c in index["cases"] if c["clip"] == clip and c["args"] == case["args"])["ok"]
    cmd = [shutil.which("ffmpeg"), *[str(VIDEO / "clips" / clip) if a == clip else a for a in case["command"][1:]]]
    channels = 4 if ok["alpha"] else 3
    size = ok["width"] * ok["height"] * channels * 2
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    while True:
        data = read_frame(process.stdout, size)
        if not data:
            break
        frames.append(np.frombuffer(data, dtype="<u2").reshape(ok["height"], ok["width"], channels).copy())
    process.wait()
    return frames, ok["contract"]


def main() -> int:
    import torch
    import torch.nn.functional as F
    from rudra.video import Predictor, ShadowSmoother

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames").mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(1)   # one accumulation order

    sequences = []
    for name, (segments, tile, overlap, retention, threshold) in SEQUENCES.items():
        predictor = Predictor(CHECKPOINT, "cpu", tile, overlap)
        smoother = ShadowSmoother(retention, threshold)
        # The raw gate for the record, before smoothing.
        raw_weights = []
        real_gate = predictor.model.predict_shadow_weight

        def gate(x, *a, **k):
            w = real_gate(x, *a, **k)
            raw_weights.append(None if w is None else float(w.item()))
            return w
        predictor.model.predict_shadow_weight = gate
        # The canonical Rec.2020 frame and its thumbnail, as predict made them.
        seen = {}
        real_interp = F.interpolate

        def interp(x, size=None, **k):
            y = real_interp(x, size=size, **k)
            if tuple(size or ()) == (36, 64) and k.get("mode") == "area":
                seen["canonical"], seen["thumb"] = x[0].numpy().copy(), y[0].numpy().copy()
            return y
        F.interpolate = interp
        records = []
        for clip, first, count in segments:
            frames, contract = decoded_frames(clip)
            for i in range(first, first + count):
                decoded = frames[i]
                rgb = decoded[..., :3].astype(np.float32) / 65535
                hdr, weight, cut = predictor.predict(rgb, contract, smoother)
                stem = f"{name}_{len(records):02d}"
                np.save(OUT / "frames" / f"{stem}.canonical.npy", seen["canonical"])
                np.save(OUT / "frames" / f"{stem}.thumb.npy", seen["thumb"])
                np.save(OUT / "frames" / f"{stem}.in.npy", decoded)
                np.save(OUT / "frames" / f"{stem}.hdr.npy", np.ascontiguousarray(hdr, dtype=np.float32))
                records.append({"clip": clip, "frame": i, "contract": contract, "input": f"{stem}.in.npy",
                                "hdr": f"{stem}.hdr.npy", "canonical": f"{stem}.canonical.npy",
                                "thumb": f"{stem}.thumb.npy", "raw_weight": raw_weights[-1],
                                "shadow_weight": weight, "cut": bool(cut)})
        F.interpolate = real_interp
        sequences.append({"name": name, "tile_size": tile, "overlap": overlap, "retention": retention,
                          "cut_threshold": threshold, "frames": records})

    rng = np.random.default_rng(7)
    area = []
    for n, (shape, size) in enumerate(AREA_CASES):
        x = rng.random(shape, dtype=np.float32)
        y = F.interpolate(torch.from_numpy(x)[None], size=size, mode="area")[0].numpy()
        np.save(OUT / "frames" / f"area_{n}.in.npy", x)
        np.save(OUT / "frames" / f"area_{n}.out.npy", np.ascontiguousarray(y))
        area.append({"input": f"area_{n}.in.npy", "output": f"area_{n}.out.npy", "size": list(size)})
    linspace = {str(n): [float(v) for v in torch.linspace(.001, 1, n).tolist()] for n in LINSPACE}
    linspace_down = {str(n): [float(v) for v in torch.linspace(1, .001, n).tolist()] for n in LINSPACE}

    with open(OUT / "index.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"oracle": "rudra/video.py Predictor.predict and ShadowSmoother, eager PyTorch %s on CPU"
                   % torch.__version__, "checkpoint": CHECKPOINT.name, "sequences": sequences, "area": area,
                   "linspace_up": linspace, "linspace_down": linspace_down}, f, indent=1)
        f.write("\n")
    print(f"video_predict: {sum(len(s['frames']) for s in sequences)} frames in {len(sequences)} sequences -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
