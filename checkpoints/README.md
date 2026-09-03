# checkpoints/

Every model this project trained on the SDR→HDR path, committed so the paper's
tables can be reproduced without asking anyone for a file. 38 MB total. The
training pairs and HDR sources stay out of the repo, as the root README says.

`models.json` is the registry. `ui/server.py` reads it to decide what to load,
which is why the folder is described rather than scanned: newest-file picks
whatever was copied last, and one of the files here is not an `SDR2HDRNet` at
all.

## What is here

| file | params | what it is |
|---|---:|---|
| **`sdr2hdr_shadow_v1.pt`** | 1,217,318 | **the shipped model.** v5 backbone + trained shadow gate, seed 20260901. What the paper reports. |
| `sdr2hdr_shadow_s2.pt` | 1,217,318 | same recipe, seed 2. Best of the three on clean PU21, worst on clean CVVDP. |
| `sdr2hdr_shadow_s3.pt` | 1,217,318 | same recipe, seed 3. Middle on every measure. |
| `sdr2hdr_image_v5.pt` | 1,196,197 | the backbone alone, no gate. What shipped before. |
| `sdr2hdr_image_v6.pt` | 4,770,117 | the capacity ablation, 64 base channels. |
| `sdr2hdr_temporal_v1.pt` | — | the temporal refiner. **Not an `SDR2HDRNet`**; the viewer filters it out by its `kind` in the registry. Unevaluated: 4 validation and 5 test clips of one scene each. |

Each `.pt` has a `<name>.config.json` beside it — the *training* config (seed,
lr, objective, which backbone it started from), kept for provenance. It is not
the architecture spec; that lives inside the payload.

`sdr2hdr_shadow_v1.pt` is
`6b7f73f0b44a6edc3e117373f7397ae3a8df12cb398ec4815597b6e2ebeb63f9`.

## Loading one

The files are **payload dicts**, not bare state dicts — `{"model", "step",
"best", "config"}` — and the config to build from is the one **inside** the
payload:

```python
import torch
from rudra.sdr2hdr import SDR2HDRNet

payload = torch.load("checkpoints/sdr2hdr_shadow_v1.pt",
                     map_location="cpu", weights_only=False)
model = SDR2HDRNet.from_config(payload.get("config", {}))
model.load_state_dict(payload["model"], strict=True)
model.eval()
```

That is `training/export_bench_pairs.py:load_model` line for line. Build the
architecture by hand instead and a checkpoint written before some flag existed
will fail a strict load far from the change.

`ui/server.py` and `training/export_bench_pairs.py` both take `--checkpoint`
and handle all of the above. The server also accepts a bare name from this
folder, so `--checkpoint sdr2hdr_image_v6.pt` works from a fresh clone, and
`GET /api/checkpoints` lists what is loadable.

## What they score

429 held-out frames, gain over the analytic inverse-ACES baseline:

| model | clean dB | clean JOD | hard dB | hard JOD |
|---|---:|---:|---:|---:|
| shadow_v1 | +0.07 | +0.113 | +1.24 | +0.389 |
| shadow_s2 | +0.71 | +0.068 | +0.96 | +0.308 |
| shadow_s3 | +0.45 | +0.089 | +1.18 | +0.358 |
| image_v5 | −3.00 | −0.046 | +1.43 | +0.443 |
| image_v6 | −2.75 | +0.004 | +0.96 | +0.343 |

The three gate seeds are the only configurations here positive on all four
measures. `shadow_v1` is the default because it has the best clean CVVDP, not
because it is the average: it is the *worst* of the three on clean PU21. The
root README has the mean and spread, and that is what the paper reports.

## Checking them

```bash
sha256sum -c checkpoints/SHA256SUMS
pytest tests/test_committed_checkpoint_2026_09_03.py
```

The test loads every model in the registry, runs a frame through it, and
compares the parameter counts to what `models.json` and the paper claim. That
last check exists because the registry said v6 had 4,772,485 parameters and the
file has 4,770,117 — the paper had copied the wrong figure too, and nothing
compared either claim to the tensors until this ran.

## Reproducing them

```bash
# the backbone: ~100k steps
python training/train_sdr2hdr.py --mode image --manifest <manifest> \
    --output-dir <dir>/image --steps 100000 --best-metric composite_gain

# the gate on top of it: ~15 minutes on one 4080
python training/train_shadow_gate.py --manifest <manifest> \
    --init-checkpoint <dir>/image/best.pt --output-dir <dir>/shadow --seed 20260901
```
