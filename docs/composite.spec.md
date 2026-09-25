# The composite: specification

Status: revision 1, 23 Sep 2026. Normative for every implementation.

`picture = composite(sdr, fields, scalars, params)` is a pure function. The
network runs once per frame and produces the fields; every user control acts
here, after it, so a slider never runs the network
(docs/NATIVE_ARCHITECTURE.md section 4).

The function exists once as this spec and as three implementations, all held
to the same goldens:

| Implementation | Precision | Where | Role |
|---|---|---|---|
| `SDR2HDRNet.forward` tail + `rudra/delivery`, `rudra/anchor.py`, `rudra/chroma.py`, `rudra/grain.py` | fp32 / fp64 | Python | the oracle (P4) |
| `native/core` `composite.cpp`, `master.cpp`, `measure.cpp` | fp32 / fp64 | C++ | CPU path, reference for the shader |
| `ui/compositor.js` (GLSL ES), `native/render/shaders/composite.frag` (GLSL 440, through QRhi) | fp16/fp32 | GPU | the live viewer |

Goldens: `tools/emit_composite_golden.py` writes
`native/tests/golden/composite/` from the Python functions themselves;
`native/tests/test_composite.cpp` checks the C++ against them.

---

## 1. Inputs

| Symbol | Shape | Meaning |
|---|---|---|
| `s` | 3 x H x W | SDR display code values, sRGB-encoded, clamped to [0, 1] |
| `r` | 3 x H x W | log-domain residual, the frame's residual scale already folded in |
| `mh`, `ms` | 1 x H x W | highlight and shadow masks in [0, 1] |
| `w_s` | scalar | per-frame shadow weight (1 without a shadow gate) |
| `c` | 1 + K | per-frame curve parameters (`{0}` without a curve head) |
| `L`, `M`, `ev_c` | scalars | model constants: `log_scale` (16), `max_hdr` (4.0), `corpus_ev` (-1) |

Units: the network convention is linear Rec.709, 1.0 = 10 000 nits. Scene
linear is 1.0 = 203 nits (diffuse white).

## 2. Interactive composite (fp32)

Per pixel, per channel `k`, in float32:

1. Baseline: `b_k = invACES(srgb_to_linear(s_k)) * 2^-ev_c * 203/10000 * 2^curve(s_k; c)`.
   `invACES` is the larger root of the Narkowicz fit with the input clamped to
   [0, 0.995]; `curve` is the piecewise-linear CurveHead correction in log2
   (native/core baseline.cpp, rudra/sdr2hdr.py).
2. Luma of the SDR: `y = 0.2126 s_0 + 0.7152 s_1 + 0.0722 s_2` (Rec.709).
3. Priors: `hp = sigmoid(24 (y - 0.82))`, `sp = sigmoid(24 (0.10 - y)) * w_s`.
4. Gate by recovery mode: all `max(hp, sp)`, highlights `hp`, shadows `sp`,
   off `0`; then `g = gate * strength`.
5. Log prediction: `p_k = clamp(log1p(b_k L) + r_k g, 0, log1p(M L))`, and
   `pred_k = expm1(p_k) / L`.
6. Preserve outside the masks (default on): `pred_k = b_k + max(mh, ms) (pred_k - b_k)`.
7. Region EV, only when some band has `ev != 0` (section 3), in nits:
   `out_k = clamp(pred_k * 10000 * G, 0, M * 10000) / 10000`.

The residual scale is not applied here: `predict_fields` folds it into `r`,
which is what keeps this function free of model heads.

## 3. Region EV (fp64, weights fp32)

For a pixel in absolute nits `n`:

* `Y = max(sum_k max(n_k, 0) * w_k, 1e-6)` with `w = (0.2627, 0.6780, 0.0593)`
  rounded to float32 (rudra/hdr10.py stores them as a float32 array).
* For each band `(lo, hi, ev)` with `ev != 0`, softness `S = max(soft, 1e-3)` stops:
  `rise = clamp((log2 Y - (log2 lo - S)) / S, 0, 1)`,
  `fall = clamp(((log2 hi + S) - log2 Y) / S, 0, 1)`,
  `m = smoothstep(min(rise, fall))`, cast to float32.
* `G = 2^(sum ev * m)`, where each `ev * m` product is float32 (a Python float
  times a float32 array, NEP 50) and the sum is float64.

Default bands (neutral): highlights 400 to 2 000, speculars 2 000 to 8 000,
shadows 0.05 to 12 nits, all at 0 EV. Neutral bands are a no-op, bit for bit.

## 4. Master chain (fp64)

What `ui/server.py _render_master` writes, in this order:

1. `n = network * 10000` (float64).
2. Region EV (section 3), then `clip(n, 0, M * 10000)`, only when graded.
3. Anchor to the SDR (default on, knee 0.9, softness 0.04), rudra/anchor.py:
   * `target = 203 * sum_k srgb_to_linear(s_k) w'_k`, `actual = sum_k n_k w'_k`,
     `w' = (0.2627, 0.6780, 0.0593)` in float64.
   * `want = (target + 1e-4) / (actual + 1e-4)`; `code = max_k s_k`.
   * `hold = median(want over the band)` when the band
     `knee - soft < code < knee + soft and actual > 1e-9` holds at least 64
     pixels, else `median(want)` over the frame. Median is numpy's: the mean
     of the two middle values for an even count.
   * `ramp = smoothstep((code - (knee - soft)) / (2 soft))`,
     `gain = want (1 - ramp) + hold ramp`, and `gain = want` where
     `code <= knee - soft`; non-finite gain becomes 1.
   * `n_k *= gain`, one scalar per pixel, never per channel.
4. Carry the source chroma (default on, knee 0.99, softness 0.01), rudra/chroma.py:
   * `src_k = 203 srgb_to_linear(s_k)`,
     `carried_k = src_k (Y_n + 1e-6) / (Y_src + 1e-6)` with the float64 weights.
   * `ramp` as in the anchor on `code = max_k s_k`, cast to float32, then
     Gaussian-blurred: sigma 2, OpenCV's kernel (17 taps, float32 taps
     normalised in double), replicate borders.
   * `n_k = carried_k (1 - ramp) + n_k ramp`, where `1 - ramp` is float32;
     non-finite becomes 0.
5. Settle the highlight grain (default on, at the anchor's knee 0.9 and
   softness 0.04), rudra/grain.py:
   * `ramp` as in the anchor on `code = max_k s_k` (float64).
   * `flat`: for each of `max_k s_k` and `sum_k s_k w'_k`, the spread
     `sqrt(max(box(x^2) - box(x)^2, 0))` over a 7x7 box (OpenCV boxFilter,
     BORDER_REFLECT); the larger of the two, in codes (`* 255`), through
     `smoothstep((3 - spread) / 2)`: 1 under one code, 0 from three.
   * `Y = max(sum_k n_k w'_k, 0)`; `settled = G(flat Y) / G(flat)` where
     `G(flat) > 1e-6`, else `Y`; `G` is a sigma-2 Gaussian (OpenCV's float64
     kernel, 17 taps, BORDER_REFLECT).
   * `Y' = Y + ramp flat (settled - Y)`, `n_k *= Y' / Y` where `Y > 1e-6`
     (else 1); non-finite becomes 0. One scalar per pixel: hue is kept.
6. Scene linear: `(n / 203)` cast to float32.
7. Container: ACES 2065-1 converts from the source primaries (default
   Rec.709; the network never changes primaries) to AP0 with
   `M = inv(NPM_dst) * Bradford(white_src -> white_dst) * NPM_src`, matrices in
   float64 from the chromaticities, pixels widened to float64 for the product
   and written as float32. The scene-linear container writes step 5 as is.

## 5. Measurements (fp64)

`analyze_frame(n)` (rudra/delivery/metadata.py): `nan -> 0`, `+inf -> 10000`,
then `max(., 0)`. MaxRGB per pixel; min, mean and max of it; per-channel
maxima (MaxSCL); percentiles 1, 5, 10, 25, 50, 75, 90, 95, 99.98 by numpy's
linear method; a 64-bin histogram of `log2(max(MaxRGB, 1e-4))` over [-14, 21]
with numpy's edge rule, normalised by the pixel count. MaxCLL and MaxFALL are
the ceilings of the brightest frame's MaxRGB peak and of its mean.

The Studio's QC numbers (ui/server.py `measure`) on the untiled composite and
its baseline: peaks, headroom in stops (peak ratio and, inside each mask > 0.5,
mean MaxRGB ratio), RMS departure from the baseline in stops, p99 and median
over all channel values, share of values above 203 x (1 + 1e-6) and above
1 000 nits, and the mask coverage. The port returns raw values; the page rounds.

## 6. Tolerances

| Check | Bound | Why |
|---|---|---|
| C++ composite vs `predict_image` | rtol 2e-4, atol 1e-6 (network units) | the golden is fed the fields `predict_fields` returned, which agree with the full forward to ~3e-6 before `expm1` |
| Region EV, anchor | rtol 1e-12 / 1e-10 | same operations in double |
| Chroma carry | rtol 1e-5, atol 1e-6 nits | OpenCV sums the float32 blur in a SIMD order |
| Highlight grain | rtol 1e-9, atol 1e-9 nits | OpenCV's box and Gaussian filters sum in their own order |
| Scene linear | exact | one division and a cast |
| AP0 | rtol 1e-7 | a 3x3 in double, then float32 |
| Matrices | 1e-13 absolute | inverse by cofactors vs LAPACK |
| analyze_frame | exact (histogram, MaxSCL, max), 1e-12 (percentiles) | |
| Studio measure | half the page's rounding step | the oracle is the rounded dict |
| GPU shader vs C++, RGBA32F target (day 8) | atol 1e-6, rtol 2e-4 (network units) | GPU `exp`, `log`, `pow` are not correctly rounded; measured 2.1e-6 worst on llvmpipe |
| GPU shader vs C++, RGBA16F target (day 8) | 2 half-float ulp | the viewer renders in fp16; measured 1 ulp on llvmpipe |

## 7. Not in the composite

Display mapping (the viewer's SDR simulation, scRGB, EDR or HDR10 encoding,
docs/NATIVE_ARCHITECTURE.md section 6.3) and file encoding (half-float EXR,
PQ) happen after it and are specified with the render and deliver layers.
