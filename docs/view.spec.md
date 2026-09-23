# The viewer: specification

Status: revision 3, 24 Sep 2026 (Phase 2 steps 2, 5 and 9). Normative for
every implementation. Revision 2 added the HDR output paths (section 9),
revision 3 the viewport (section 10); the guides follow (step 11).

The viewer takes the two float pictures the composite produces, the
reconstruction and its analytic baseline (docs/composite.spec.md), and turns
them into what the artist sees and reads: the picture on the glass, the probe,
the measurements and the scopes. None of it changes a pixel of the master.

| Implementation | Where | Role |
|---|---|---|
| `ui/compositor.js` (`DISPLAY`, `REDUCE`, `COPY`, `probe`, `sample`, `sourceIndex`) and `ui/app.js` (`adopt`, `computeStats`, `buildScopes`, `drawVector`) | browser | the oracle (P4) |
| `native/core` `view.cpp`, `scopes.cpp` | C++ | CPU reference |
| `native/render/shaders/display.frag` and the reduction passes, through QRhi | GPU | the live viewer |

Goldens: `tools/emit_viewer_golden.py` runs the oracle files unmodified in
headless Chromium and writes `native/tests/golden/viewer/`;
`native/tests/test_viewer.cpp` checks the C++ against them.

---

## 1. Inputs and units

| Symbol | Shape | Meaning |
|---|---|---|
| `M` | 3 x H x W, fp32 | the reconstruction, network units (1.0 = 10 000 nits), the composite target `model` |
| `B` | 3 x H x W, fp32 | the analytic baseline, the composite target `base` (the composite with `baselineOnly`) |
| `mh`, `ms` | H x W | highlight and shadow masks, the halves `/api/frame` sent, decoded exactly |
| `s8` | 3 x H x W, uint8 | the SDR pixels the page composites from |

`P = 10000` nits per network unit. Rec.2020 luma weights
`w = (0.2627, 0.6780, 0.0593)`, applied to the linear values as they are.

**Orientation.** Every array is in image order, row 0 the top of the picture,
end to end: the uploads, both composite targets, the probe, the sample and the
readbacks. The display pass is the one place that flips: it reads image row
`H - 1 - j` into framebuffer row `j`, because the framebuffer puts row 0 at the
bottom. A readback of the float targets is therefore never flipped, and a
readback of the presented picture is flipped once to be in image order.

## 2. The display pass

Parameters: `view` (0 image, 1 false colour, 2 difference), `displayNits`
(the view peak, default 203), `show` (`model` or `baseline`), `wipe` (off, or
a position in [0, 1]), `wipeHalfWidth` (0.0012), `diffGain` (2 000 nits).

For the output pixel at column `x` (of `W`), in image row `y`:

1. `u = (x + 0.5) / W`. The wipe is on when `wipe >= 0`.
2. The source: `S = B` when the wipe is off and `show = baseline`, else `S = M`.
   With the wipe on the right-hand side is always the reconstruction.
3. The picture sample: `h = B[y, x]` when the wipe is on and `u < wipe`,
   else `h = S[y, x]`.
4. The view:
   * image: `c = linear_to_srgb(clamp(h * P / displayNits, 0, 1))`, per channel.
     Exposure and a hard clip, no tone curve.
   * false colour: `c = zone(dot(h, w) * P)` (the table below), independent
     of `displayNits`.
   * difference: `d = dot(|S[y, x] - B[y, x]|, w)`,
     `v = clamp(log2(1 + d P) / log2(1 + max(diffGain, 1)), 0, 1)`,
     `c = v * (0.95, 0.62, 0.28)`. It reads `S` and `B` whatever the wipe.
5. The wipe handle: when the wipe is on and `|u - wipe| < wipeHalfWidth`,
   `c = 1 - c`.
6. Encode: `linear_to_srgb(x) = 12.92 x` for `x <= 0.0031308`, else
   `1.055 x^(1/2.4) - 0.055`, on `x` clamped to [0, 1]. The 8-bit picture is
   `round(255 c)`, the unsigned-normalised conversion of the framebuffer.

False-colour zones, by Rec.2020 luminance in nits `n`, first match:

| `n` below | colour (linear, as written to the framebuffer) | reads as |
|---|---|---|
| 0.1 | (0.169, 0.122, 0.239) | below the shadow floor |
| 1 | (0.184, 0.294, 0.561) | deep shadow |
| 10 | (0.184, 0.561, 0.722) | shadow |
| 100 | (0.200, 0.627, 0.416) | mid tones |
| 160 | (0.604, 0.655, 0.698) | just under diffuse white |
| 250 | (0.914, 0.929, 0.945) | diffuse white (203) |
| 1 000 | (0.910, 0.765, 0.290) | highlight |
| 4 000 | (0.910, 0.529, 0.227) | bright highlight |
| 10 000 | (0.816, 0.263, 0.184) | specular |
| otherwise | (0.780, 0.290, 0.780) | past the PQ peak |

The zone colours are not sRGB-encoded: false colour writes them straight to
the framebuffer.

## 3. The probe

`probe(x, y)`: `px = clamp(round(x), 0, W - 1)`, `py = clamp(round(y), 0, H - 1)`
(round half up), then for each of `M` and `B` the texel at `(px, py)` in image
order: `r, g, b = texel * P` and `nits = dot(texel, w) * P`. No flip.

## 4. Reductions

Over `m = max(R, G, B)` of a composite target, in fp32 on the GPU, by a ladder
of 2x2 passes to one texel (the first pass takes the channel maximum):

* `peak = max(m) * P`, exact: a maximum loses nothing.
* `mean = sum(m) * P / (W H)`, the sum in the ladder's tree order: level by
  level, each output texel `(x, y)` adds, in fp32, the source texels
  `(2x, 2y)`, `(2x+1, 2y)`, `(2x, 2y+1)`, `(2x+1, 2y+1)` that exist, rows in
  image order, until one texel is left. Evaluated in that order the sum is
  the same number on every implementation.

`MaxCLL = ceil(peak(M))`, `MaxFALL = ceil(mean(M))`. Never from the sample:
a downsample would miss the one specular pixel MaxCLL is about.

## 5. The sample and its index

Distributional numbers and the scopes run on a point sample capped at 768 on
the long side: `k = min(1, 768 / max(W, H))`, `sw = max(1, round(W k))`,
`sh = max(1, round(H k))`, in double, round half up. Sample pixel `(x, y)` takes source pixel

    sx = min(W - 1, floor((2x + 1) W / (2 sw)))
    sy = min(H - 1, floor((2y + 1) H / (2 sh)))

in integer arithmetic, on the GPU (`COPY`) and on the CPU (`sourceIndex`) alike,
so the masks read at the index always belong to the pixel sampled. The index
is `sy W + sx`. (Before 24 Sep 2026 the GPU took a filtered lookup and the CPU
a floating-point floor, and they disagreed where the quotient is an integer.)

## 6. Measurements (`computeStats`)

On the sample, `n = sw sh`. Per sample pixel, `l = max(R, G, B) * P` of `M` and
`lb` the same of `B` (the page's `luma`: the channel maximum, not a luma).
Arithmetic in double on the fp32 texels; `l` is stored as fp32. Per channel value `v` of `M` in nits (`3n` values):

* `above_diffuse_white_pct = 100 #(v > 203 (1 + 1e-6)) / 3n`,
  `above_1000_nits_pct = 100 #(v > 1000) / 3n`.
* A 2 048-bin histogram of `t = (log2 clamp(v, 0.05, 4000) - log2 0.05) / (log2 4000 - log2 0.05)`,
  bin `clamp(floor(2048 t), 0, 2047)`. `pct(p)`: the first bin `k` whose
  cumulative count reaches `3n p / 100` gives `2^(log2 0.05 + (k + 0.5) span / 2048)`;
  none gives 4 000. `p99_nits = pct(99)`, `median_nits = pct(50)`.

Using the index into the full-resolution masks:

* `headroom_highlight_stops`: over sample pixels with `mh > 0.5`,
  `log2(max(mean l, 1e-6) / max(mean lb, 1e-6))`; NaN when there are none.
  `headroom_shadow_stops` the same with `ms`.
* `departure_rms_stops = sqrt(mean over the sample of log2((l + 1e-4) / (lb + 1e-4))^2)`.

From the reductions: `peak_nits`, `baseline_peak_nits`, `maxcll`, `maxfall`,
`headroom_stops = log2(max(peak, 1e-6) / max(baseline_peak, 1e-6))`.

At full resolution, once per frame (`adopt`): `highlight_mask_pct` and
`shadow_mask_pct`, the share of pixels with `mh > 0.5` and `ms > 0.5`; and
`clipped`, the share with any SDR channel at 254 or above.

## 7. Scopes (`buildScopes`, `drawVector`)

Waveform and histogram, on `l` over the sample (`w x h` = `sw x sh`), with
`v = clamp(l, 0.05, 4000)`:

* 230 columns, column `min(229, floor(230 x / w))`; 512 bins per column on
  `t = (log10 v - log10 0.05) / log10(80000)`, bin `min(511, floor(512 t))`.
* Per column the 2, 25, 50, 75 and 98 percentiles: the first bin `k` at which
  the cumulative count reaches `total p / 100` (`total` the column count, or 1)
  gives `(k + 0.5) / 512`; a percentile never reached gives 1.
* A 76-bin histogram on `(log2 v - log2 0.05) / (log2 4000 - log2 0.05)`,
  bin `min(75, floor(76 t))`, normalised by `max(1, highest bin)`.

Vectorscope, 256 x 256, on `M` over the sample: `r, g, b` in nits,
`Y = dot((r, g, b), w)`; pixels with `Y < 0.02` are skipped. With
`q = max(Y, 1)`, `cb = (b - Y) / (2 (1 - 0.0593)) / q`,
`cr = (r - Y) / (2 (1 - 0.2627)) / q`; the pixel lands at
`(round(128 + (cb 128) 0.92), round(128 - (cr 128) 0.92))`, evaluated in
double in that order (round half up), and outside the square is skipped. The image: with `A` the
count and `peak` the highest count, `a = (ln(1 + A) / ln(1 + max(peak, 1)))^0.6`
(0 where `A = 0`), colour `(158, 242, 255) a` and alpha `min(255, 300 a)`,
each rounded half to even into 8 bits.

## 8. Tolerances

| Check | Bound | Why |
|---|---|---|
| browser composite vs `composite.cpp` (step 1) | 5e-5 + 1e-5 \|ref\| (network units), measured 3e-6 | GPU `exp`, `log`, `pow` are not correctly rounded |
| display pass, C++ vs browser (step 3) | 1 code in 8 bits; zone index exact; measured: 3 of 290 k values 1 code off, the rest equal | the GPU's `pow` in `linear_to_srgb` |
| display pass, shader vs C++ (step 4) | 1 code in 8 bits; measured on llvmpipe: equal in every byte | the same |
| probe, index, sample | exact | texel reads and integers |
| peak (step 6) | exact against the same target | a maximum |
| sum (step 6) | exact when evaluated in the ladder's order; measured: equal to the browser's and the shader's | the order is specified, so no bound is needed |
| waveform, histogram, vectorscope bins (step 8) | exact; measured: equal, and every `computeStats` number too | integer counts on the same sample |
| HDR paths, shader vs C++, RGBA32F (step 5) | 1e-5 + 2e-4 \|ref\|; measured 2.8e-5 relative on llvmpipe | GPU `pow`, amplified by PQ's exponent of 78.84 |
| HDR paths, shader vs C++, RGBA16F (step 5) | 2 half-float ulp; measured 1 | the swapchain format |
| viewport vs the browser's layout (step 9) | scale and readout exact; rectangle within 1/64 px; pan within 1e-3 px | Chromium lays out in 1/64 px and reads transforms back in float32 |
| the viewer window's swapchain vs `core/view.cpp` on `composite.cpp` (step 9) | 1 code in 8 bits, either texel where a pixel centre falls on a texel edge (fractional device pixel ratios); measured 1 on llvmpipe at ratios 1 to 2 | the picture is held in RGBA16F between the display pass and the blit |

## 9. HDR output paths

Section 2 is the SDR path, the browser Studio's view. On an HDR swapchain the
same pass writes absolute light instead (ADR-005: the peak comes from the
display and nothing is tone-mapped behind the artist's back). A target is
`(path, primaries, peak, unit)`:

| Path | Swapchain (QRhi) | Primaries | Value written for `n` nits |
|---|---|---|---|
| SDR | `SDR` | Rec.709 (sRGB) | section 2 |
| scRGB | `HDRExtendedSrgbLinear`, scene-referred (Windows, Linux Vulkan) | Rec.709 | `n / 80` |
| HDR10 | `HDR10` | Rec.2020 | `PQ(n)`, ST 2084 of `n / 10000` |
| EDR | `HDRExtendedDisplayP3Linear`, or `HDRExtendedSrgbLinear` display-referred (macOS) | Display P3, or Rec.709 | `n / 203`: 1.0 is the SDR white, held at diffuse white |

`peak` is what the display reports now (maximum luminance on Windows; on
macOS the live EDR headroom times 203, re-read every frame).

**Primaries.** The composite keeps the primaries of its input, Rec.709 for
an sRGB still (docs/composite.spec.md: "the network never changes
primaries"), so the picture is converted from those source primaries to the
swapchain's with `rgb_to_rgb_matrix` (no adaptation: all four are D65). The
SDR path writes the values as they are, which is the same statement for a
Rec.709 source. (Revision 1 of NATIVE_ARCHITECTURE.md 6.3 assumed a Rec.2020
working space here; the composite never had one.)

**Image view.** Per channel in the source primaries,
`n_k = clamp(h_k P, 0, ceiling)` with `ceiling = min(displayNits, peak)`:
the view peak slider still sets a ceiling, but it clips at that luminance
instead of exposing to it. Then `o = M_source->target n`, rows summed in order
0, 1, 2 in fp32, and each `o_k` is encoded for the path. Because the
Rec.709 to P3 and Rec.709 to Rec.2020 matrices have non-negative rows summing
to one, clipping before the conversion keeps every output channel within the
ceiling. The wipe handle writes `ceiling - n_k`.

**Overlays.** False colour and the difference view are graphics, not light:
their section 2 value `c` (the handle inverting it to `1 - c`) is shown at the
SDR white, `n_k = 203 srgb_to_linear(c_k)` in Rec.709, then converted with
`M_709->target` and encoded as above.

## 10. The viewport

Where the picture lands in the viewer, in the viewer's logical pixels, origin
top-left (`core/viewport.hpp`; the oracle is ui/style.css with ui/app.js
`fitScale`, `applyViewport` and `zoomAbout`, laid out by Chromium itself,
tools/emit_viewport_golden.py). The viewer has a 14-pixel padding on each side.

* **Fit.** `fit = min(1, (W - 28) / fw, (H - 28) / fh)`: the whole frame
  inside the padded viewer, never above 1:1. The frame is `fw fit` by
  `fh fit`, centred.
* **Zoomed.** A scale `s` and a pan `(px, py)`: the frame at `fw s` by
  `fh s`, centred on `(W / 2 + px', H / 2 + py')` where `px'` is the pan
  rounded to 0.1 pixel (the transform is written with `toFixed(1)`).
* **Wheel.** One notch multiplies the scale by 1.12 (or divides), clamped
  to [0.05, 32], keeping the frame point under the cursor still: with `c`
  the cursor minus the frame's current centre and `k = to / from`,
  `pan = (pan - c) k + c`. From fit, `from` is the fit scale.
* **Middle drag** pans; from fit it first becomes a zoom at the fit scale.
  **Double click** returns to fit; 1:1 is scale 1, pan 0.
* **Readout** `round(100 s)` percent. **Pixel under the pointer**
  `floor((x - left) / width fw)`, and the same in y; outside the frame,
  none. **Wipe from the pointer** `clamp((x - left) / width, 0, 1)`.

The window multiplies by its device pixel ratio. Placement follows the
Studio in logical pixels; "actual pixels" and the zoom readout are in device
pixels (scale `1 / dpr` is 100 %), so 1:1 is one frame pixel per screen
pixel on a scaled display too. At and above 1:1 the
picture is sampled nearest, so a pixel is a pixel; below it, trilinear over
its mip chain. The surround is neutral grey #121212, drawn as a graphic at
the SDR white on an HDR swapchain.

Three Studio defects were found by laying the page out for this, and fixed
there on 24 Sep 2026: the zoom readout and the first wheel step used the
uncapped fit (350% claimed for a frame shown at 100%); a frame taller than
the viewer was never fitted (the canvas's `max-height:100%` refers to the
plate's auto height); and a zoomed frame wider than the viewer was scaled from
its fit size and sat at the grid track's start, so 1:1 was not 1:1 and the
first wheel tick threw the picture off centre.

## 11. Not in revision 3

The guides (step 11).
