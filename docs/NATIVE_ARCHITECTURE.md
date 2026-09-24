# RUDRA Native: architecture and design patterns

> 23 Sep 2026. Companion to `docs/DESKTOP_APP_PLAN.md` (stack, phases, cost).
> This document fixes **how** the native app is built: the layers, the types,
> the threading model, the patterns, the numerics, and the first ten days.
> Stack (revision 2 of this document): Qt 6 (Widgets), **QRhi** over Metal /
> Direct3D 12 / Vulkan with an OpenGL fallback, C++20, **LibTorch + ONNX
> Runtime**, OpenEXR, OpenColorIO, FFmpeg with hardware decode. Target UI: the
> "Pro direction" boards on the design canvas. No platform is second-class: every
> OS gets native GPU rendering, compute, HDR output and accelerated inference.

---

## 0. The one idea the whole design rests on

The browser Studio already made the decision that matters, in `ui/server.py`
`run_frame()`:

> The fields the head produces (log residual and the two masks) do not depend
> on recovery_mode, residual_strength or the grade.

So the product is two pure functions and a cache between them:

```
fields    = infer(model, sdr_frame)                 # expensive, once per frame, GPU
picture   = composite(sdr_frame, fields, params)    # cheap, every slider move
```

`infer` runs once per (frame, model). `composite` is a pure function of its
inputs and runs every time a control moves, at display rate. Every design
decision below serves one of three goals:

1. keep `infer` off the UI's critical path,
2. make `composite` exist **once as a specification** and **twice as code**
   (GLSL for the viewer, compiled to every GPU API; C++ for masters and
   measurement) with a parity test
   that fails CI when they drift,
3. make the native result provably equal to the Python result.

---

## 1. Principles

| # | Principle | Consequence in code |
|---|---|---|
| P1 | **Units and colour encodings are types, not comments.** | `Image<Space>` tags; `Nits`, `Stops`, `SceneLinear` strong types; the 203-nit diffuse white defined once. A PQ buffer cannot be passed where scene-linear is expected; it does not compile. |
| P2 | **Pure stages, immutable data.** | Frames, fields and params are immutable values shared by `shared_ptr<const T>`. Threads exchange values, never mutate shared state. |
| P3 | **One composite, two backends, one parity test.** | `composite.spec.md` + `composite.cpp` + `composite.glsl`, tested pixel for pixel against each other and against Python goldens. |
| P4 | **Python is the oracle until retired.** | Every core module ships with golden files emitted by the Python code. No module merges without passing its golden test on all three OSes. |
| P5 | **Refuse, and say why.** | No silent fallbacks. Unknown colour tags, VFR, interlace, a model whose manifest does not match: typed error, shown to the user, same rules as `rudra.video`. |
| P6 | **The core knows nothing about Qt, the GPU API or the GPU vendor.** | `librudra` builds and tests headless on a CI box with no display. |
| P7 | **Every output is reproducible.** | Sidecar records model sha, params hash, app version, backend, precision. Same inputs, same bytes on CPU. |
| P8 | **Measure before optimising.** | Performance budgets are set from the Phase 0 spike on the RTX 4080, not guessed; Tracy zones on every stage from day one. |
| P9 | **Native on every OS, one source.** | One renderer on QRhi (Metal, D3D12, Vulkan, GL), one shader source compiled per API, one inference interface over several runtimes. Platform code lives only inside `render/` and `infer/` backends. |

---

## 2. Layers and the dependency rule

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ app/         Qt Widgets: windows, view models, design tokens     │  depends on ↓
 ├──────────────────────────────────────────────────────────────────┤
 │ engine/      Session, Scheduler, caches, jobs, undo              │
 ├───────────────┬───────────────┬───────────────┬──────────────────┤
 │ render/       │ infer/        │ media/        │ deliver/         │
 │ QRhi viewer:  │ LibTorch +    │ readers (hw   │ encoders, QC,    │
 │ Metal, D3D12, │ ONNX Runtime  │ decode),      │ queue            │
 │ Vulkan, GL    │ backends      │ writers (EXR) │                  │
 ├───────────────┴───────────────┴───────────────┴──────────────────┤
 │ core/  (librudra) types, colour, baseline, composite (CPU),      │
 │        measure, metadata, grade, sidecar, errors                 │
 ├──────────────────────────────────────────────────────────────────┤
 │ platform/    files, paths, hashing, threads, logging, clocks     │
 └──────────────────────────────────────────────────────────────────┘
 cli/  headless `rudra-native`: core + infer + media + deliver, no Qt
```

**Rule:** a layer may depend only on layers below it. Enforced three ways:
CMake `target_link_libraries(... PRIVATE ...)` so headers do not leak upward;
a CI script that fails if any file under `core/` includes `<Q`, `<rhi/` or a GPU API header,
and if any file outside `render/` includes `<rhi/`;
`cli/` must link without Qt.

**Why this shape (the Apple habit):** AppKit apps that age well keep the model
layer free of the UI framework. It is what lets the CLI, the tests, every GPU
backend and a future ComfyUI/OFX plugin reuse the same core unchanged.

---

## 3. Domain model

All value types. Cheap to copy or shared immutably. Hashable where they feed
a cache key.

### 3.1 Colour and units (`core/color`)

```cpp
enum class Transfer  { Srgb, Bt1886, Gamma24, Linear, Pq, Hlg };
enum class Primaries { Rec709, Rec2020, P3D65, Ap0, Ap1 };
enum class Range     { Full, Limited };

struct ColorEncoding {            // what the numbers in a buffer mean
    Transfer  transfer;
    Primaries primaries;
    Range     range;
    float     nits_per_unit;      // 203 for scene-linear, 10000 for PQ-normalised
    bool operator==(const ColorEncoding&) const = default;
};

namespace space {                  // compile-time tags for the common ones
    struct SdrDisplay;             // sRGB/Rec.709 display-referred, [0,1]
    struct SceneLinear709;         // linear, Rec.709 primaries, 1.0 = 203 nits
    struct SceneLinear2020;
    struct DisplayPq2020;
}

template <class Space> class Image;   // RGB float32 planar or interleaved, owns pixels
struct Nits  { float v; };  struct Stops { float v; };
inline constexpr Nits kDiffuseWhite{203.f};
```

`Image<Space>` conversions exist only as named functions
(`to_scene_linear(const Image<SdrDisplay>&, Baseline)`), so every colour
transform in the program is greppable and tested.

### 3.2 Pipeline values

| Type | Holds | Produced by | Notes |
|---|---|---|---|
| `SourceFrame` | SDR pixels as `Image<SdrDisplay>`, `ColorEncoding` of origin, timecode, frame index, content hash | `media::Reader` | hash = xxh3 of decoded pixels, the cache key root |
| `FrameScalars` | `residual_scale`, `shadow_weight` (and future heads) | `infer` frame pass | computed **once per frame** from the whole frame, never per tile |
| `Fields` | log residual RGB + highlight mask (fp16 RGBA), shadow mask (fp16 R), `FrameScalars`, model id | `infer` tile pass | invariant to every user control, exactly as today |
| `GradeParams` | mode, residual strength, preserve outside masks, region EV qualifiers, display peak, container | UI edits | immutable, versioned, `hash()`; serialised identically into project, sidecar and queue |
| `Composite` | `Image<SceneLinear2020>` fp32 | `composite()` | what masters and measurements read |
| `Measurements` | MaxCLL, MaxFALL, peak, P99, share above 1 000 nits, share clipped, headroom stops | `core::measure` | CTA-861.3 maths ported from `rudra/delivery/metadata.py` |

### 3.3 Documents and jobs

| Type | Role |
|---|---|
| `Shot` | a clip or an image sequence opened by path; frame count, rate (rational), encoding, per-frame clip statistics for the timeline lane |
| `Project` (`.rudraproj`, JSON, schema-versioned, relative paths) | shots, per-shot `GradeParams`, collections, render destinations |
| `MasterJob` / `DeliveryJob` | frozen copy of `GradeParams` + model sha + preset + output path |
| `Queue` | `queue.json` **byte-compatible with `rudra batch`**, same state file, same lock semantics |

---

## 4. Data flow

```
             media thread            infer thread (owns device session)   render thread (QRhi)
 path ──► Reader.decode(i) ──► SourceFrame ──► FramePass ──► TilePass ──► Fields ─┐
                 │                 │  (whole frame,      (512 px tiles,         │
                 │                 │   ≤512 px view +     overlap, blend)        │
                 │                 │   native stats)                             ▼
                 │                 └──────────────► upload SDR texture   upload fields
                 │                                                             │
                 │                           GradeParams (UI, every edit) ────►│
                 │                                                             ▼
                 │                         composite shader → view transform → HDR/SDR swapchain
                 │                                                             │
                 │                                           scopes (compute) ◄┘
                 ▼
          deliver thread:  SourceFrame + Fields + frozen GradeParams
                           → composite.cpp (fp32) → measure → EXR / ffmpeg pipe → QC → sidecar
```

- The viewer never waits for inference to redraw: a slider move re-runs only
  the composite shader over resident textures.
- The master never uses the GPU render path. It runs `composite.cpp` in fp32 so the
  file on disk is independent of driver, GPU and display.
- Measurements shown in the inspector come from `composite.cpp` on a
  downsampled copy for interactivity and from the full frame on demand
  ("Re-measure"), matching what the browser Studio does with `measure()`.

---

## 5. Design patterns, and why each one

### 5.1 Pure pipeline + content-addressed cache (render graph, lite)

Each stage is a pure function with a version number. Its result is cached
under `hash(inputs..., stage_version, model_sha)`.

```cpp
struct FieldsKey { Hash frame; Hash model; uint32_t infer_version; Precision p; };
LruCache<FieldsKey, shared_ptr<const Fields>> fields_cache{bytes_budget};
```

Why: seeking back is free, switching model is correct by construction (the key
changes), and a stale result can never be displayed for the wrong frame. This
is the Nuke/Resolve pattern scaled down to three stages; a general node graph
is **not** built, because RUDRA's pipeline is fixed.

Budget example, 4K frame: fields fp16 × 5 channels = 3840·2160·5·2 B ≈ 83 MB.
A 16-frame read-ahead is ~1.3 GB of host RAM; VRAM holds the current frame and
the next 2. Budgets are settings, sized at start-up from available memory.

### 5.2 Actors with bounded queues (threads own resources)

| Actor | Owns | Why it must be single-threaded |
|---|---|---|
| `MediaActor` | libav contexts, EXR readers | codec contexts are not thread-safe |
| `InferActor` | the backend session (LibTorch module or ONNX Runtime session) and its device stream | one session per device, deterministic ordering |
| `RenderThread` | the `QRhi` instance, swapchain, textures, pipelines | QRhi and every native API behind it are used from one thread |
| `DeliverActor` | ffmpeg child process, EXR writers | one output at a time per job, like `rudra batch` |
| UI thread | widgets, view models | Qt rule |

Messages are immutable values (`shared_ptr<const SourceFrame>` etc.) over
bounded single-producer/single-consumer queues. The bound **is** the read-ahead
window, so backpressure is free. No mutex guards pixel data anywhere.

This is Swift's actor model expressed in C++20: isolation by ownership, not by
locks.

### 5.3 Priority scheduler with generations and cancellation

Three priorities: **interactive** (the frame under the playhead) >
**read-ahead** > **background** (masters, deliveries, thumbnails, the timeline
clip lane). Every request carries a `generation` number; a seek increments it,
and results from an older generation are dropped on arrival instead of
displayed. Long jobs take a `std::stop_token`.

Why: the classic failure of scrubbing tools is showing frame 120's result on
frame 180 after a fast seek. Generations make that impossible without locking.

### 5.4 Strategy interfaces at every external boundary

```cpp
struct InferenceBackend {           // LibTorch{Cuda,Mps,Cpu}, Ort{DirectML,CoreML,Rocm,OpenVino,Cpu}
    virtual Result<FrameScalars> frame_pass(const Image<SdrDisplay>&) = 0;
    virtual Result<Fields>       tile_pass (const Image<SdrDisplay>&, const FrameScalars&, TilePlan) = 0;
    virtual BackendInfo          info() const = 0;    // device, precision, versions
};
struct Reader  { virtual Result<SourceFrame> decode(FrameIndex) = 0; /* ... */ };
struct Writer  { virtual Result<void> write(const Composite&, const FrameMeta&) = 0; };
struct ViewerBackend { /* upload, composite, display, scopes, readback */ };  // QRhi (all APIs)
```

Why: each has several implementations **in v1**, not someday. Inference runs
on LibTorch where it is fastest (CUDA, MPS) and on ONNX Runtime everywhere else
(DirectML covers any DX12 GPU on Windows, Core ML the Apple GPU and Neural
Engine, ROCm and OpenVINO AMD and Intel on Linux). `BackendSelector` ranks what
the machine offers, runs the golden self-test on the winner, and falls back
down the list on failure. Swapping or adding one (AOTInductor, TensorRT) must
not touch the engine.

| Machine | Chosen by default | Fallbacks |
|---|---|---|
| Windows + NVIDIA | LibTorch CUDA | ORT DirectML, CPU |
| Windows + AMD / Intel | ORT DirectML | CPU |
| Mac, Apple Silicon | LibTorch MPS | ORT Core ML, CPU |
| Linux + NVIDIA | LibTorch CUDA | CPU |
| Linux + AMD | ORT ROCm | CPU |
| Linux + Intel | ORT OpenVINO | CPU |

### 5.5 Presets as data (registry)

HDR10, HLG, ProRes 422/422 HQ/4444 and EXR containers are JSON profiles
(`deliver/profiles/*.json`) ported from `rudra/delivery/profiles.py`: codec,
pixel format, colour tags, filter graph, QC rules. A `PresetRegistry` loads
them. Adding a preset is a data change reviewed against the Python profile,
not new C++.

### 5.6 Command pattern for edits, undo for free

Every user edit is a `GradeCommand { before: GradeParams, after: GradeParams }`
pushed on a `QUndoStack` per shot. Because `GradeParams` is an immutable value,
undo is a pointer swap and the composite simply re-runs. Slider drags coalesce
into one command (`mergeWith`).

### 5.7 MVVM at the Qt boundary

Widgets never call the engine. Each panel has a view model (`QObject`) that
exposes engine state as properties and signals, and turns user intent into
commands.

```
Widget  ⇄  ViewModel (QObject, UI thread)  →  Engine (actors, any thread)
                 ▲                                   │
                 └──── queued signal with value ◄────┘
```

Why: the Pro design and the dense "Full" workspace become two sets of widgets
over the **same** view models. Screenshot tests drive view models directly.

### 5.8 `Result<T>` and typed errors

`std::expected<T, Error>` throughout `core`, `infer`, `media`, `deliver`.
`Error` carries a code, a user-facing sentence and a technical detail. No
exceptions cross a module boundary (LibTorch's exceptions are caught at the
`infer` edge and converted). The UI renders `Error` the same way everywhere,
which is how "refuse, and say why" stays consistent.

### 5.9 Model package with capability negotiation

A model is a folder, not a file:

```
sdr2hdr_shadow_v1/
  model.ts              TorchScript module: frame_pass(), tile_pass()
  model.frame.onnx      ONNX graph of frame_pass (opset pinned)
  model.tile.onnx       ONNX graph of tile_pass
  manifest.json         contract version, corpus_ev, log_scale, max_hdr,
                        heads present {gate, shadow_gate, ...}, tile/overlap, opset,
                        source .pt sha256, torch version, export date
  golden/               16 frames: inputs and expected outputs
  LICENSE               the non-commercial weights licence
```

The engine reads `manifest.json` and enables only what the model declares
(for example, no shadow-weight UI if the model has no shadow gate). A
`contract` major version mismatch is a refusal. This matters now: an
uncommitted change on `main` adds a new model head
(`tests/test_curve_head_2026_09_23.py`); the native app must accept v1 and v2
models without code forks.

---

## 6. Computer-vision and numerics

### 6.1 Export: move every tensor op that affects numbers into the model

`predict_residual_scale` and `predict_shadow_weight` downscale with
`F.interpolate(mode="area")` to ≤512 px for pooled features but read
statistics from the **native** frame (the chroma high-frequency evidence is
destroyed by an area resize: 1.13 sd vs 0.60 sd with the sign inverted, per
the 29 Aug measurement in `rudra/sdr2hdr.py`). Re-implementing that resize or
those statistics in C++ is a parity risk with no upside. So the export wraps
them:

```python
class Exported(torch.nn.Module):
    def frame_pass(self, sdr_native):        # (1,3,H,W) full frame
        return residual_scale, shadow_weight # scalars, same code as today
    def tile_pass(self, sdr_tile, residual_scale, shadow_weight):
        return log_residual_rgb, highlight, shadow
```

The same wrapper is exported twice, `torch.jit` for LibTorch and
`torch.onnx.export` (dynamic H and W) for ONNX Runtime, and both are checked
against eager PyTorch on the goldens before the package is written. If an op
exports badly to ONNX (the area resize is the one to watch), it is rewritten in
the wrapper as an equivalent average pool, and that rewrite is itself checked
against eager.

C++ does only what has no numerical content: tiling geometry, overlap
blending with the **same window** `training/infer_sdr2hdr.py::predict_fields`
uses, and memory layout. The analytic baseline (`sdr_to_baseline_hdr`) is
computed inside the model too, for the tile pass; C++ ports it separately in
`core` only for the composite and for "Baseline" view, with a golden test.

### 6.2 Precision policy

| Stage | Precision | Reason |
|---|---|---|
| network | fp32 default; fp16 opt-in, labelled in UI and sidecar. Core ML and DirectML may run fp16 internally: their tolerance is recorded per backend | parity first, speed second |
| fields storage | fp16, clamped to `MAX_FIELD_MAGNITUDE` exactly as `run_frame` | same as the browser path, halves memory |
| composite (CPU and shader) | fp32 | log-domain add then exp: fp16 loses highlights |
| masters | computed fp32, written EXR half (float optional) | as today |
| measurements | fp32 over the full composite | MaxCLL is a max: one wrong pixel changes it |

### 6.3 Colour pipeline

- **Input decode** mirrors `rudra.video` exactly: tags decide transfer,
  primaries, matrix, range; missing tags require explicit overrides; the same
  rejections. Phase 1 uses libavfilter with the same `zscale` arguments the
  Python path passes, so parity is by construction. A hand-written float
  YUV→RGB path can replace it later, behind a golden test.
- **Working space:** scene-linear, 1.0 = 203 nits, Rec.2020 for the composite
  (Rec.709 input primaries converted with the exact matrix from
  `rudra/delivery/colorspace.py`).
- **View transforms:**
  - SDR out: today's PQ simulation (display-peak exposure + clip), identical
    maths to `compositor.js` `DISPLAY`.
  - HDR out, one shader with a per-swapchain output stage:

    | Swapchain (QRhi) | OS / API | Encoding | Value written |
    |---|---|---|---|
    | `HDRExtendedSrgbLinear` | Windows D3D12; Linux Vulkan where offered | scRGB: linear Rec.709, 1.0 = 80 nits, negatives allowed | `M709←src · nits / 80` |
    | `HDR10` | Windows D3D12 (option) | PQ, Rec.2020, 10-bit | `PQ(M2020←src · nits)` |
    | `HDRExtendedDisplayP3Linear` | macOS Metal (EDR) | linear Display P3, 1.0 = the display's SDR white | `MP3←src · nits / 203` |

    `src` is the composite's primaries, which are the input's (Rec.709 for an
    sRGB still): the network never changes primaries. Revision 1 of this
    table converted from Rec.2020, which the composite never was; the
    normative version is docs/view.spec.md section 9.

    Peak is clamped to what the display reports: max luminance on Windows,
    the live EDR headroom on macOS (it changes with the brightness slider, so
    it is re-read every frame). The pipe bar shows the path and the headroom.
  - Masters: ACES 2065-1 (AP0) or linear Rec.2020 via OpenColorIO, with the OCIO
    config the Python writes (`rudra aces --ocio`) so Resolve and Nuke read the
    same thing.

### 6.4 Scopes and analysis

- Waveform, RGB histogram, vectorscope: QRhi compute shaders with atomic bins
  over a decimated grid (230 columns × 76 bins, as `server.py scopes()`),
  log-nits axis, on every backend. Only the OpenGL fallback below 4.3 uses the
  CPU path from a downsampled readback.
- **Timeline clip lane** (Pro design): per-frame SDR clipped share and crushed
  share, computed by the media actor at decode time (cheap, no network), stored
  per shot, so the lane fills as the shot is read.

### 6.5 Temporal

Frames are independent; the v02 temporal line is closed. The only temporal
state is the optional scalar shadow-gate smoothing with cut reset, ported from
`rudra.video`, and it lives in `DeliverActor` and the viewer's playback path
only when enabled.

### 6.6 Performance budgets (set in Phase 0, recorded here)

| Path | Budget | Measured |
|---|---|---|
| composite + view, 1080p, GPU | ≤ 4 ms | **composite + display pass 0.240 ms (D3D12), 0.248 (D3D11), 0.244 (Vulkan), 0.219 (OpenGL)**, RTX 4080 SUPER, GPU timestamps, 24 Sep; the composite alone 0.118 ms |
| composite + view, 4K, GPU | ≤ 12 ms | **composite + display pass 1.156 ms (D3D12), 1.154 (D3D11), 1.167 (Vulkan), 1.030 (OpenGL)**; the composite alone 0.49 ms |
| inference, 1080p, RTX 4080, LibTorch CUDA fp32 / fp16 | measure | fp32 172 ms untiled, 296 ms tiled 512/64 (RTX 4080 SUPER, fields in host memory); fp16/bf16 not built yet |
| inference, 1080p, RTX 4080, ORT DirectML fp32 | measure | 150 ms untiled, 489 ms tiled 512/64; CPU for reference: LibTorch 2.5 s, ONNX Runtime 3.2 s |
| inference, 1080p, Apple M-series, LibTorch MPS / ORT Core ML | measure | [Phase 0] |
| viewer measurements and scopes, CPU, after a slider settles (off the render thread) | ≤ 16 ms | 21 ms at 1080p and 23 ms at 4K (768 x 432 sample) plus 5 ms vectorscope on two 2.1 GHz cloud cores (29 and 7 ms on one): the per-pixel work runs in up to eight chunks whose integer counts merge exactly, the order-dependent sums stay sequential, and the result is bit-identical to the browser's still (`rudra-native bench-scopes`); the desktop number comes from `NATIVE_GATE_A.ps1` |
| first frame after open (warm) | ≤ 2 s | decode plus one inference: 150 to 172 ms of inference at 1080p on the RTX 4080 SUPER (above); end to end from the app on Windows open |
| scrub to cached frame | ≤ 1 display frame | synchronous: a cached frame is delivered inside `FrameEngine::show()` (0.01 ms in the engine test); the upload and passes are the composite + view row |

Instrumentation: Tracy zones on every actor message and GPU timer queries on
every render pass (QRhi GPU timestamps), from the first commit.

Read of the numbers (23 Sep 2026): the composite is 35 times inside its 1080p
budget and 24 times inside its 4K budget, so every control runs at display
rate with room for the view pass and scopes. Inference is the only cost that
matters and it is paid once per frame, not per slider move: 150 to 170 ms at
1080p untiled on the RTX 4080 SUPER in true fp32. Tiling costs 1.7x on CUDA and
3.3x on DirectML (per-call overhead on twelve 512 tiles), so tiles are for
memory, not for speed, as the Studio already treats them. A reduced-precision
path (fp16/bf16, own tolerance, never the default) is the first speed item in
Phase 1.

---

## 7. Threading model, summarised

| Thread | Runs | May block on |
|---|---|---|
| UI | widgets, view models, undo | nothing, ever |
| Render | QRhi upload, composite, display, scopes | vsync |
| Media | decode, clip statistics, thumbnails | disk, codec |
| Infer | frame pass, tile pass | GPU |
| Deliver | composite.cpp, measure, write, ffmpeg pipe, QC | disk, child process |
| Pool (N cores) | CPU composite tiles for masters, EXR compression | nothing shared |

Cross-thread communication: bounded queues and queued Qt signals carrying
values. Shared mutable state: none except the caches, which are internally
synchronised and store immutable values.

---

## 8. Persistence and interchange

| File | Format | Compatible with |
|---|---|---|
| `.rudraproj` | JSON, `schema` field, relative paths | native only; migration functions per schema bump |
| sidecar `.json` | same keys as the Python sidecar, plus `backend`, `precision`, `app_version` | Python readers ignore unknown keys |
| `queue.json` + `.state.json` | identical to `rudra batch` | run or resume by either tool |
| settings | `QSettings` | per user |
| caches | `<appData>/RUDRA/cache/` (thumbnails, clip lanes), keyed by content hash | safe to delete |

---

## 9. Errors, logging, diagnostics

- `spdlog` structured JSON logs to `<appData>/RUDRA/logs`, one file per session.
- A **Diagnostics** window: backend info, model manifest, GPU/driver, display
  HDR state, ffmpeg capability probe, last 200 log lines, "Copy report".
- Crash reports stay local (minidump on Windows, `.ips` on macOS); nothing is
  uploaded.
- Every refusal (`Error`) is logged with its code; the code list is documented
  so support can map a screenshot to a cause.

---

## 10. Testing

| Level | Tool | What it proves | Tolerance |
|---|---|---|---|
| Unit | GoogleTest | each function's contract | exact / 1 ulp |
| Golden parity | GoogleTest reading `tests/golden/` emitted by `pytest --emit-golden` | C++ core == Python | per module, table in `tests/golden/TOLERANCES.md` |
| Model parity | `rudra-native diff` on 429 bench frames | LibTorch == eager PyTorch | log-space max abs ≤ 1e-5 (CPU fp32) |
| Composite parity | readback on **each** QRhi backend vs `composite.cpp` | viewer == master on every API | ≤ 2 half ulp |
| Backend parity | each `InferenceBackend` on the 429 bench frames | every runtime == eager PyTorch | per backend, in `TOLERANCES.md` |
| Integration | CLI end to end: clip in, HDR10 out, QC | whole path | QC pass + sidecar keys |
| Queue interop | Python writes queue, C++ resumes it, and back | compatibility | state identical |
| UI | Qt Test + screenshot diff of Pro boards | layout, states | perceptual diff threshold |
| Performance | Google Benchmark in CI on fixed frames | no regressions | ±10% gate |

The golden emitter is the linchpin: it turns the existing Python test suite
into the native specification.

---

## 11. Repository layout and targets

```
native/
  CMakeLists.txt  CMakePresets.json  vcpkg.json
  platform/   rudra_platform   (files, hash, log, threads)
  core/       rudra_core       (types, colour, baseline, composite.cpp, measure, metadata, sidecar)
  media/      rudra_media      (libav reader, EXR/PNG/TIFF sequences, EXR writer)
  infer/      rudra_infer      (InferenceBackend, LibTorch + ONNX Runtime impls, BackendSelector, tiling)
  render/     rudra_render     (ViewerBackend on QRhi, shaders/*.glsl → .qsb, scopes, HDR swapchains)
  deliver/    rudra_deliver    (presets, ffmpeg pipe, QC, queue)
  engine/     rudra_engine     (Session, Scheduler, caches, actors, undo)
  app/        RUDRA            (Qt Widgets, view models, tokens → QSS)
  cli/        rudra-native
  tests/      unit, golden, parity, ui, bench
  design/     tokens.json (from the Pro boards), icons
tools/
  export_model.py        writes the model package (§5.9)
  emit_golden.py         pytest plugin: writes tests/golden/
```

Design tokens (colours, radii, type scale, spacing from the Pro direction)
live in `design/tokens.json` and generate both QSS and a C++ header, so the
look has one source.

---

## 12. What is next: the first ten working days

Goal: answer the two go/no-go questions of Phase 0 and leave a skeleton every
later phase builds on.

| Day | Deliverable | Done when | Status |
|---|---|---|---|
| 1 | `tools/export_model.py`: `frame_pass` + `tile_pass` as TorchScript **and** ONNX, `manifest.json`, 16 golden frames | both reloaded graphs match eager on the 16 frames, max abs ≤ 1e-6 | **done** 23 Sep: TorchScript max \|d\| 0.0, ONNX 5.9e-5 on the shipped model; an all-heads model also passes |
| 2 | `native/` CMake + vcpkg skeleton (Qt with Shader Tools, ONNX Runtime, LibTorch), empty targets, dependency-rule check, CI on Windows/macOS/Linux | three green builds | **done** 23 Sep: 18 GoogleTests green; Qt shell builds and starts; CI in `.github/workflows/native.yml` |
| 3 | `rudra_infer`: `InferenceBackend`, LibTorch CPU + ORT CPU, tiling + overlap blend | `rudra-native diff` runs on 1 frame on both | **done early** 23 Sep: LibTorch and ORT CPU backends; the tiler is bit-exact with `predict_fields` |
| 4 | Model parity on the 429 bench frames: CPU fp32 on both runtimes, then CUDA, DirectML, MPS, Core ML | **Gate A:** ≤ 1e-5 log-space on CPU for both; every GPU backend's delta recorded | **Windows done** 23 Sep, RTX 4080 SUPER: LibTorch CPU 1.2e-7, LibTorch CUDA 1.4e-5 (true fp32, TF32 off; `gpu_fp32` bound 5e-5 + 1e-5 \|ref\|), ONNX Runtime CPU 5.8e-5, DirectML 3.3e-6; the 429-frame bench, MPS and Core ML open |
| 5 | QRhi HDR spike: a bare `QRhi` window on Windows (D3D12, `HDRExtendedSrgbLinear`) and macOS (Metal, `HDRExtendedDisplayP3Linear`), 1 000-nit patch; Linux Vulkan probed | **Gate B:** patch measured above SDR white on Windows and on an XDR display | **Windows passes** 23 Sep: `rudra-hdr-probe` (Qt 6.8, QRhi) on an RTX 4080 SUPER and an ASUS PA279CRV (418-nit peak, SDR white 240): D3D12 scRGB 203 / 1 000 / 2 000 nits exact, D3D12 HDR10 202.9 / 998.9 / 1 991.8 (10-bit PQ), D3D11 scRGB exact; the SDR fallback reports FAIL. XDR Mac open |
| 6 | `core/color` types, `ColorEncoding`, `Image<Space>`, units; baseline port + golden | golden passes on 3 OSes |  |
| 7 | `composite.spec.md` written from `compositor.js`, `composite.cpp`, golden vs browser Studio readback | exact on 5 test frames | **done** 23 Sep: [`composite.spec.md`](composite.spec.md); `composite.cpp` matches `predict_image` on 5 mode/strength/preserve cases on 2 frames (rtol 2e-4), Region EV and the whole master chain to AP0 match `_render_master` stage by stage; browser readback moves to day 8 with the shader |
| 8 | composite shader in GLSL 440 compiled by `qsb`, running in the spike window on D3D12, Metal, Vulkan and GL; readback parity with `composite.cpp` | ≤ 2 half ulp on every backend | **Windows done** 23 Sep: `render/shaders/composite.frag`, `GpuCompositor`, `rudra-gpu-parity`; on an RTX 4080 SUPER D3D12 (fp32 3.0e-6), D3D11, Vulkan and OpenGL (1.7e-6) all pass with fp16 at 1 half ulp, 12 cases each; llvmpipe passes too. Metal open |
| 9 | `measure()` port + MaxCLL/MaxFALL golden; Tracy + QRhi GPU timestamps; first budget numbers in §6.6 | table filled | **port done** 23 Sep: `analyze_frame`, MaxCLL/MaxFALL and the Studio `measure()` match the Python; timing in place: `rudra-native bench` (inference) and `rudra-gpu-parity --bench` (composite pass, QRhi GPU timestamps), run by both gate scripts; §6.6 numbers from the Windows run; Tracy open |
| 10 | Review: ADRs signed, budgets and backend matrix recorded, go/no-go | decision written into `STATUS.md` | **done** 23 Sep: GO for Phase 1 on Windows; macOS conditional on its three gates (`STATUS.md`, line F) |

If Gate A fails, the fix is in the export (usually a traced branch or a
dtype, or for ONNX an op that needs rewriting in the wrapper); nothing else
starts until it passes. If Gate B fails on one OS, that OS ships SDR-out in v1
and HDR-out follows; the renderer does not change.

---

## 13. Decisions, recorded as ADRs

Status after the Phase 0 review, 23 Sep 2026. Accepted means the Phase 0
evidence supports it; accepted, not yet exercised means nothing in Phase 0
could test it and it stands until Phase 1 does.

| ADR | Status | Evidence |
|---|---|---|
| 001 | accepted | both graphs exported and verified; TorchScript bit-exact, ONNX 5.8e-5 |
| 002 | accepted | Qt 6.8.3 builds the probe and the GPU composite on MSVC (VS 2026 Build Tools), GCC and, for the Windows headers, MinGW |
| 003 | accepted | the composite passes readback parity on D3D12, D3D11, Vulkan and OpenGL; Metal open |
| 004 | accepted | four Windows backends pass Gate A; TF32 off by default on CUDA |
| 005 | accepted | scRGB and HDR10 carry 1 000 and 2 000 nits to the swapchain; the SDR fallback reports FAIL |
| 006 | accepted, not yet exercised | decode is Phase 1 |
| 007 | accepted | contract 1.0 read and verified by `rudra-native`; `gpu_fp32` added without a major bump |
| 008 | accepted | `queue.json` state written byte-identical by both sides and resumed across them (Phase 1 step 9) |
| 009 | accepted | Windows DX12 on an RTX 4080 SUPER measured; the Apple Silicon row waits on the Mac run |
| 010 | accepted | the viewer is that window: composite, display and blit on its own swapchain, passing its readback on llvmpipe; the glass run is `NATIVE_GATE_B.ps1` |


1. **ADR-001** Model package carries TorchScript and ONNX; AOTInductor and
   TensorRT are later speed backends, never the only format.
2. **ADR-002** Qt 6 LTS (6.8 or later) with Qt Shader Tools; LGPL dynamic
   linking.
3. **ADR-003** Renderer on QRhi: D3D12 on Windows, Metal on macOS, Vulkan on
   Linux, OpenGL fallback; all QRhi use confined to `render/`.
4. **ADR-004** Two inference runtimes behind `InferenceBackend`: LibTorch
   (CUDA, MPS, CPU) as reference, ONNX Runtime (DirectML, Core ML, ROCm,
   OpenVINO) for universal GPU coverage; `BackendSelector` order in §5.4.
5. **ADR-005** HDR output encodings per swapchain (§6.3), peak from the display,
   never tone-mapped silently.
6. **ADR-006** Decode through libavfilter + `zscale` for parity in v1; hardware
   decode enabled only after it matches software decode on the test clips.
7. **ADR-007** Model package format and contract versioning (§5.9).
8. **ADR-008** `queue.json` and sidecar stay byte-compatible with Python.
9. **ADR-009** Supported hardware: Apple Silicon only on macOS; DX12 GPUs on
   Windows; the Linux GPU and compositor matrix for HDR.
10. **ADR-010** (accepted, Phase 2) The viewer is a `QWindow` with its own QRhi
   swapchain, embedded with `createWindowContainer`; not `QRhiWidget`, whose
   backing-store composite is SDR (section 15).

---

## 14. Phase 1: librudra, the next fifteen working days

Goal (docs/DESKTOP_APP_PLAN.md section 6): every Python module the product
depends on exists in C++ and passes its golden test on three OSes, ending in a
master EXR rendered with no Python on the machine. Same rules as Phase 0: the
Python is the oracle, each module lands with its goldens and tests, the README
is ticked in the same commit.

Already ported in Phase 0: baseline, tiling, composite, Region EV, anchor,
chroma carry, colour-space matrices, `analyze_frame`, MaxCLL/MaxFALL, Studio
`measure`.

| # | Deliverable | Oracle | Done when | Days | Status |
|---|---|---|---|---|---|
| 1 | One golden harness: `tools/emit_golden.py` runs every emitter; CI re-emits on all three OSes and runs the native tests against fresh arrays | the emitters | a Python change that moves a number fails CI | 0.5 | **done** 23 Sep: `tools/emit_golden.py` (core, composite, decode, delivery, master, qc, queue, sequence); CI re-emits on Linux and fails on any drift in the exact goldens; since step 11 the `core` job re-emits on all three OSes |
| 2 | Still decode in `media/`: PNG 8/16-bit (grey, alpha, palette), JPEG, TIFF 8/16/float, BMP, WebP; bit depth, padded-16-bit detection and distinct codes reported; scene-linear float refused | `rudra/decode.py` `decode_sdr` | same float pixels, bits and refusals on a fixture set of every format and depth | 2 | **done** 23 Sep: `media/still.cpp` on OpenCV imgcodecs, the decoder the Python uses; 17 fixtures bit-exact, JPEG included, across OpenCV 4.6 (C++) and 4.13 (Python). EXR dropped: the Python refuses it too |
| 3 | Grade controls: exposure, highlight desaturation, shoulder to peak, `apply_grade`, `itm_strength_map` | `rudra/delivery/controls.py` | golden arrays, rtol 1e-12 | 1 | **done** 24 Sep: `core/grade.cpp`; four grades within 2e-7 of the float32 output (numpy's SIMD `exp` is the only difference), the strength map exact |
| 4 | HDR10 and profiles: PQ OETF/EOTF, `master_to_peak`, `master_to_pq`, delivery profiles | `rudra/hdr10.py`, `delivery/profiles.py` | golden arrays; PQ codes exact at 10 and 12 bit | 1 | **done** 24 Sep: `core/hdr10.cpp`, every profile incl. HLG; 12-bit PQ codes exact on 407 levels; float PQ within 2e-5 of code (numpy's float32 `power` is 1 ulp off where glibc is correctly rounded, and PQ's 78.84 exponent amplifies it: a fiftieth of a 10-bit step) |
| 5 | Metadata writers: `detect_shots`, `l1_per_shot`, Dolby Vision generate JSON, HDR10+ JSON, the RUDRA sidecar | `delivery/metadata.py` | byte-identical JSON on a multi-shot fixture | 1.5 | **done** 24 Sep: `core/metadata.cpp`, `deliver/sidecars.cpp`, `platform/pyjson` (Python's `json.dumps` and `repr`) and numpy's pairwise sum; all three sidecars byte-identical on a 12-frame, 3-shot sequence |
| 6 | EXR and ACES writers on OpenEXR: half pixels, chromaticities, provenance attributes, AP0 container | `delivery/exr.py`, `delivery/aces.py` | pixels exact after the half cast; every header attribute equal when read back by Python | 2 | **done** 24 Sep: `deliver/exr.cpp` writes the EXR directly, like the Python (no OpenEXR library needed); half, float, RGBA, ACES AP0 and ACEScg files and the OCIO config byte-identical; float16 rounding exact on 1 012 values incl. subnormals and overflow |
| 7 | **`rudra-native master <package> <image> --out x.exr`**: decode, infer, composite, master chain, measure, EXR, sidecar | `ui/server.py` `_render_master` | the same EXR within 1 half-float ulp and the same sidecar numbers as the Studio master, on three stills, no Python installed | 2 | **done** 24 Sep: `cli/master.cpp`, checked by `rudra-native master-check` (a ctest): three stills (8-bit PNG, 16-bit PNG, JPEG; default, graded linear, highlights-only unanchored) within 1 half ulp on LibTorch and ONNX Runtime (at most 0.8% of values one ulp off), every EXR header attribute equal, sidecars equal and byte-identical in two of three (the third's peak rounds 3235.5 vs 3235.6) |
| 8 | QC: `check_frame`, thresholds file, report text | `rudra/qc.py` | same pass/fail and identical report text on the fixtures | 1.5 | **done** 24 Sep: `deliver/qc.cpp`; six fixtures walk every status (PASS, FAIL, UNMEASURED) of all ten checks and both verdicts; statuses equal, values within 1e-9 (1e-3 for the two blur-based ones, OpenCV's SIMD blur), report text byte-identical |
| 9 | Queue: `queue.json` read, write, lock, atomic save, resume, artifact digests | `rudra/batch.py` | a queue started by Python resumes in C++ and the other way round | 1.5 | **done** 24 Sep: `deliver/queue.cpp`; a fresh run with a failing job and its `--retry-failed` resume write the Python's state files byte for byte, a Python-started state resumes here to the Python's final state, nine malformed queues refused with the same messages, one runner per queue (`busy`) |
| 10 | Sequence open by path: frame folders and numbering rules (video frames wait for libav in Phase 4) | `ui/sequence.py` | same frame list and order on the fixture folders | 1 | **done** 24 Sep: `media/sequence.cpp`; same names, order and messages as `Sequence.open` on 14 cases (natural order across powers of ten, case, leading zeros, dot files, quoted and messy paths, empty and frameless folders); video recognised and refused until Phase 4 |
| 11 | Review: CI green on Windows, macOS, Linux; Mac runs from Phase 0 folded in; Phase 1 exit written into `STATUS.md` | | every module passes its golden on three OSes | 1 | **in progress** 24 Sep: the `core` CI job now re-emits every golden with each runner's Python and builds with OpenCV on Windows (vcpkg, cached), macOS (Homebrew) and Linux; goldens are byte-exact in git (`.gitattributes`) and every Python writer the port matches (sidecars, OCIO config, queue state, golden indexes) writes LF on every OS, so Windows compares the same bytes. Waiting on the first three-OS run and the Mac runs |

New third-party code, none in the public headers (principle P6): OpenCV
core and imgcodecs in `media/src` (`RUDRA_WITH_OPENCV`), chosen over separate
PNG, JPEG and TIFF libraries because it is the decoder `rudra/decode.py` calls,
so the two decode the same bytes to the same floats, quirks included (palette
expansion, grey-alpha, 16-bit TIFF). No OpenEXR: the Python writes EXR itself and
the port does the same, byte for byte. OCIO
waits until a module needs it; nothing in this list does.

Order: 1 first, then 2 to 6 in any order, 7 as soon as 2 and 6 land (it is the
milestone that proves the port end to end), then 8 to 11.

## 15. Phase 2: the QRhi viewer, the next twenty working days

Goal (docs/DESKTOP_APP_PLAN.md section 6): the picture, the probe and every
measurement the browser Studio shows, drawn by QRhi on D3D12, Metal, Vulkan and
OpenGL, into an HDR swapchain where the display has one. Same rules as Phases
0 and 1, with one change of oracle: the viewer ports `ui/compositor.js` and the
scope code in `ui/app.js`, so those files, run unchanged in a headless browser,
are the oracle. The Python stays the oracle for anything numeric it also does
(composite, measure), and the two must already agree.

Already in hand from Phase 0: the composite shader and `GpuCompositor`
(readback parity on four Windows backends), `rudra-hdr-probe` (scRGB, HDR10,
the SDR fallback), `ViewerBackend` and `OutputPath` in `render/`.

| # | Deliverable | Oracle | Done when | Days | Status |
|---|---|---|---|---|---|
| 1 | Browser oracle: `tools/emit_viewer_golden.py` drives `ui/compositor.js` and the scope functions of `ui/app.js`, unmodified, in headless Chromium (Playwright, SwiftShader WebGL2 with float targets) on the composite goldens' frames | `ui/compositor.js`, `ui/app.js` | deterministic across runs; its composite readback matches `composite.cpp` within the `gpu_fp32` bound, closing the browser readback left open in Phase 0 step 7 | 1.5 | **done** 24 Sep: `tools/emit_viewer_golden.py`, `tests/test_viewer.cpp`; two frames from the shipped checkpoint (80x48, and 900x40 so `sample()` downsamples), six composites and seven views, deterministic across runs; the browser composite matches `composite.cpp` within 3e-6 (bound 5e-5 + 1e-5\|ref\|), peak and mean too. It found a Studio defect, fixed in `ui/compositor.js`: `sample()` chose its texel with a filtered lookup and `sourceIndex()` in doubles, and they disagreed on rows where (2y+1)h/(2 sh) is an integer, pairing one pixel with its neighbour's mask; both now use the same integer expression |
| 2 | `docs/view.spec.md` written from the `DISPLAY` shader, `present()`, `probe()`, `sample()` and the scopes: the one V flip, wipe and handle, false-colour zones, log difference ramp, exposure plus hard clip, sRGB encode, sample grid | the same files | every constant and branch in them has a line in the spec | 0.5 | **done** 24 Sep: [`view.spec.md`](view.spec.md) revision 1: orientation, the display pass and its zones, probe, reductions, sample and index, `computeStats`, `buildScopes` and the vectorscope, with evaluation order where it decides a rounding |
| 3 | `core/view.cpp`: CPU reference of the display pass for all three views and the wipe | browser canvas readback | within 1 8-bit code of the browser; false-colour zone index exact; handle columns exact | 1.5 | **done** 24 Sep: `core/view.cpp`; 14 views on two frames (image at 203 and 1 000 nits, baseline, false colour, difference, two wipes) equal to the browser canvas in every byte but 3 of 290 k, which are 1 code off; zones and handle columns exact |
| 4 | `display.frag` in GLSL 440 through `qsb`, run by `rudra-gpu-parity` after the composite | `core/view.cpp` | within 1 8-bit code on D3D12, D3D11, Vulkan, OpenGL (and Metal); llvmpipe in CI | 1.5 | **Linux done** 24 Sep: `render/shaders/display.frag`, `GpuCompositor::view`, seven views per frame in `rudra-gpu-parity`; OpenGL on llvmpipe equal to `core/view.cpp` in every byte (CI runs it). D3D12, D3D11 and Vulkan run with `NATIVE_GATE_B.ps1`; Metal with `native_gate_b.sh` |
| 5 | HDR output from the display pass: scRGB, HDR10 PQ, EDR, with the Rec.2020 to Rec.709 / P3 matrices and the view peak taken from the display instead of a clip at SDR white (ADR-005); SDR keeps today's exposure and clip | `core/view.cpp` extended, `rudra-hdr-probe` patches | readback within 2 half ulp per path; Gate B re-run through the real display pass on the PA279CRV | 2 | **Linux done** 24 Sep: [`view.spec.md`](view.spec.md) section 9, `core/view.cpp` and `display.frag` for scRGB, HDR10 and EDR (P3 and Rec.709), `GpuCompositor::view_values`; 40 cases per run in `rudra-gpu-parity`, on llvmpipe fp32 within 2.8e-5 relative and fp16 within 1 half ulp; unit tests against `pq_oetf` and the gamut matrices. Found that 6.3 converted from a Rec.2020 working space the composite never had; it now converts from the source primaries. The glass re-run of Gate B through this pass needs the viewer window and moves to step 9 |
| 6 | GPU reductions: the `REDUCE` ladder as QRhi passes for peak, mean and MaxFALL | `measure()` (Python), browser `peakNits` / `meanNits` | peak exact, mean within fp32 summation bounds, on every backend | 1 | **done** 24 Sep: `render/shaders/reduce.frag`, `GpuCompositor::reduce`, `core/view.cpp` `reduce_ladder` (the ladder's own fp32 order); peak and sum equal to the browser's bit for bit, MaxCLL and MaxFALL too, and the shader equal to the CPU ladder on llvmpipe. Windows and Metal runs with the gate scripts |
| 7 | Probe: one texel from each float target, nits by Rec.2020 luma, no flip | browser `probe()`, `composite.cpp` | equal at a fixed point set including the four corners | 0.5 | **done** 24 Sep: `core/view.cpp` `probe_pixel`; eight points per frame, corners included, equal to the browser's `probe()` bit for bit. On the GPU it is a one-texel read of the target, done by the viewer (step 9) |
| 8 | Sample and scopes: the 768-side point sample and its source index, the waveform quantiles and histogram, the vectorscope, ported to `core/scopes.cpp` and run on the sample | browser `sample()`, `buildScopes`, `drawVector` accumulation | sample indices exact; waveform, histogram and vectorscope bins exact; the scopes cost recorded (compute shaders only if the CPU path misses its budget) | 1.5 | **done** 24 Sep: `core/scopes.cpp`; the sample grid and index, all 14 `computeStats` numbers, the waveform quantiles, the histogram and the vectorscope image equal the browser's bit for bit on every stored case, and the mask coverage too. Cost at 1080p (768 x 432 sample) on a 2.1 GHz cloud core: 32 ms for the measurements and scopes, 7 ms for the vectorscope; they run off the render thread, after a slider settles, as in the Studio. Compute shaders only if the Windows measurement (step 12) misses a 16 ms budget |
| 9 | The viewer in the Qt shell: a `QWindow` with its own QRhi swapchain inside the widget tree (`createWindowContainer`), fit, 1:1, zoom about the cursor, pan, wipe drag, view switching with no recomposite, resize, device loss | `fitScale` and `zoomAbout` in `ui/app.js` (goldens of viewport maths) | viewport maths equal to the browser's; a still is judged in the app exactly as in the Studio; Gate B re-run through the real display pass on the PA279CRV (scRGB and HDR10) | 3 | **done** 24 Sep. Linux: `render/viewer_window.cpp` (composite, display and blit on its own swapchain, the HDR format from the display, device loss rebuilt from host copies), `core/viewport.cpp` equal to the Studio's own layout (`tools/emit_viewport_golden.py`); `rudra-viewer-check` reads the swapchain back at fit, 2x and 1:1 within 1 code of `core/view.cpp` at device pixel ratios 1, 1.25, 1.5 and 2. Windows, RTX 4080 SUPER and PA279CRV: Gate B through the real display pass passes on D3D12 and D3D11 scRGB (203 exact, 1 000 and 2 000 clipped at the display's 418 nits) and on Vulkan scRGB (418 from DXGI). The first Windows runs found three things, all fixed: the parity check did not allow for texel ties at a 150 % display scale, and Qt's Vulkan swapchain reports a placeholder 1 000-nit peak, now replaced by the DXGI value on Windows; and on D3D12, D3D11 and Vulkan a swapchain readback completes only when its frame slot comes round again, so each window check compared a case with the one before it (reproduced on lavapipe, now waited for with `QRhi::finish`, and lavapipe Vulkan runs in CI). "Actual pixels" is device 1:1. Three Studio viewport defects fixed on the way (view.spec.md section 10) |
| 10 | Frame path: decode, an engine inference job with generations and cancellation, field upload, composite, present; stills and sequences (Phase 1 step 10) with fields cached per frame | Phase 1 modules | scrubbing a 240-frame folder never shows a stale frame; latency recorded | 3 | **done** 24 Sep: `engine/frame_engine.cpp`, the InferActor: one worker owns the backend; `show()` bumps the generation, cancels older queued work, queues the frame and 12 after it; results are cached (32 frames, LRU, the frame on screen never evicted) and delivered only while current. A 240-frame scrub faster than inference never delivers another frame's pixels or fields (fakes tagged by value; ThreadSanitizer clean); with the real package on CPU over the 17 decode fixtures, every delivered frame equals a direct decode, the refused float TIFF is reported as itself, the way back is all cache hits. The app opens a still or a folder through it (comma, full stop, Home, End, Space at 24 fps). Latency at 1080p is measured with step 12 |
| 11 | Guides (new, no Studio oracle): title and action safe, aspect masks, centre cross, specified in `view.spec.md` | the spec | CPU reference test; drawn identically on every backend | 1 | **done** 24 Sep: [`view.spec.md`](view.spec.md) section 11, `core/guides.cpp`, drawn by the blit in device pixels; `rudra-viewer-check` holds the window's readback to the CPU reference with all guides on and a 2.39 and a 4:3 mask, at fit, 2x and 1:1, device pixel ratios 1 and 1.5 (within 1 code); the app's View > Guides (G, Shift+G) |
| 12 | Backend matrix and budgets: everything above on D3D12, D3D11, Vulkan, OpenGL, Metal; composite plus view at 1080p against the 4 ms budget in 6.6 | | matrix and numbers recorded here and in 6.6 | 1.5 | **in progress** 24 Sep: the matrix above; `rudra-gpu-parity --bench` times composite plus display per slider move and `rudra-native bench-scopes` the CPU measurements, both run by the gate scripts; 6.6 filled where measured. Waiting on the next Windows gate runs and the Mac |
| 13 | Review: Phase 2 exit written into `STATUS.md` | | probe and measurements equal the browser Studio's on every backend | 0.5 | |

Backend matrix for the viewer (step 12), from the gate scripts and CI:

| Backend | Composite (fp32 / fp16) | Display pass (SDR, 7 views) | HDR paths (40 cases) | Reductions | Window readback (fit, 2x, 1:1, guides) | Gate B through the viewer |
|---|---|---|---|---|---|---|
| OpenGL, llvmpipe (Linux, CI) | pass | exact | pass (2.8e-5, 1 ulp) | exact | pass at device pixel ratios 1 to 2 | n/a: no HDR swapchain |
| D3D12, RTX 4080 SUPER | pass (fp32 3.0e-6, fp16 1 ulp) | 1 code | pass, HDR10 near black within 1/20 of a 10-bit code (1.4e-5) | exact | stalled at the 03:17 run (below); re-run | **pass**, scRGB, 203 exact, clipped at 418 |
| D3D11, RTX 4080 SUPER | pass (1.7e-6, 1 ulp) | 1 code | as D3D12 | exact | **pass** at device pixel ratio 1.5 | **pass**, scRGB, as D3D12 |
| Vulkan, RTX 4080 SUPER | pass (1.7e-6, 1 ulp) | 1 code | as D3D12 | exact | **pass** at device pixel ratio 1.5 | **pass**, scRGB, the display's 418 nits from DXGI (Qt reports a placeholder 1 000) |
| Vulkan, lavapipe (Linux, CI) | pass | exact | pass | exact | pass at device pixel ratios 1 and 1.5 | n/a: no HDR swapchain |
| OpenGL, RTX 4080 SUPER | pass (1.7e-6, 1 ulp) | 1 code | as D3D12 | exact | **pass** at device pixel ratio 1.5 | n/a: Qt's OpenGL swapchain is SDR on Windows |
| Metal, Apple Silicon | open | open | open | open | open | open (the XDR Mac run) |

The window readback failures on Windows (24 Sep, 03:08 and 03:14 runs) were
the check, not the viewer: the dumps show the placed rectangle off the
intended scale and pan in every failing case (1:1 at 75.7 device pixels wide
instead of 80), so the check window was being scrolled or clicked while it
ran. The check now ignores input (`ViewerWindow::set_input_enabled`) and
records the viewport it saw. With that, the 03:17 run passes on D3D11,
Vulkan and OpenGL within 1 code. On D3D12 the check wrote no report: it hit
the old 30 s limit. With a 90 s limit it stops at the eleventh case (wipe 0.37
at 2x) after ten passes, the window exposed and the event loop alive: the
grab was asked for and no frame followed. The viewer now counts update
requests, presented frames and failed `beginFrame` calls, the check asks
again for a frame once a second while a grab waits (`nudges`, per case in the
report), and a `TIMEOUT` report carries the counters (`stall`), so the next
run says whether D3D12 drops the update request or refuses the frame. It
drops it: the 03:50 run passes all 18 cases within 1 code, each one after
exactly two nudges. That is a viewer defect, not a check one (in the app, a
slider move that never shows), so the viewer now arms a 50 ms timer with
every update it asks for and draws the frame itself if the request never
arrives; `fallback_frames` counts those, per case in the report. On Linux it
never fires.
Found on the way: on Vulkan the window's surface outlived the
`QVulkanInstance` it was made from, and the check crashed on exit after
writing its report (lavapipe, exit 139); the viewer now destroys its platform
window before the instance goes.

Why a `QWindow` and not `QRhiWidget` (ADR-010, proposed): `QRhiWidget` draws
into a texture that the widget backing store composites, and that path is SDR
on every platform today. Gate B already proved the `QWindow` swapchain path
carries scRGB and HDR10 to the glass, so the viewer is that window, embedded in
the widget layout; the widgets around it stay ordinary Qt Widgets.

Why the scopes on the CPU first: they read the 768-side sample the browser
already reads (at most 590 k pixels, a few milliseconds), the result must be
bin-exact with the browser, and QRhi compute is unavailable on the OpenGL
fallback below 4.3. A compute path is added only if step 8's measurement says
the CPU path misses its budget.

Order: 1 and 2 first (the oracle and the spec), then 3 to 8 in order of
dependency (3 before 4 and 5; 6, 7 and 8 need only the composite targets), 9
and 10 together, 11, then 12 and 13.

## 16. Phase 3: the Qt UI, the next fifteen working days

Goal (docs/DESKTOP_APP_PLAN.md section 6): the full Studio workflow in the
native app with no Python installed. Open a package and a shot, scrub it,
reconstruct and grade it, compare, probe, measure and master it, and every
number and file matches what the browser Studio gives for the same actions.
Same rules as before, with the page itself as the oracle for behaviour.
`ui/index.html` names the actions and panels, `ui/app.js` holds the state and
what each control does, `ui/theme.css` holds the colours, and `ui/server.py`
holds the master's naming and parameters. Layout follows section 2.5 of the
desktop plan and the Pro-direction boards.

In hand from Phase 2: the viewer window (with its wipe, flip, zoom, pan and
guides), the frame engine, the display pass in every encoding, and the
measurements and scopes as data, all equal to the Studio's.

| # | Deliverable | Oracle | Done when | Days | Status |
|---|---|---|---|---|---|
| 1 | Theme: QSS generated at build time from the custom properties in `ui/theme.css`; IBM Plex Mono and Plex Sans Condensed embedded (OFL) | `ui/theme.css` | every colour in the QSS is a `theme.css` token (a build-time test), the surround is R = G = B | 1 | **done** 24 Sep: `app/theme/studio.qss.in` names the tokens as the page's CSS does (`var(--panel)`) and `cmake/rudra_theme.cmake` fills them from the `:root` of `ui/theme.css` at build time (no Python, no Qt; an unknown token fails the build); `app/theme.cpp` loads it with a palette from the same tokens under Fusion. `test_theme`: every colour in the sheet is one `ui/theme.css` uses, every grey R = G = B, the only hues the theme's named ones (plus its active-control ink and slider key), every background neutral, the fonts the tokens' first families, no colour literal in the app's C++. IBM's own TTFs (Sans Condensed 400 to 700, Mono 400 and 500, unmodified, OFL): the files carry per-weight legacy family names, so `RUDRA --theme-check` verifies at run time that every weight resolves under the typographic family; it does on Linux (fontconfig), and the Qt shell CI job and `NATIVE_GATE_B.ps1` run it. The first Windows run registered all six faces but looked them up as `IBM Plex Sans Condensed\r`: the generated token file had CRLF there; the generator now strips CR from its inputs and writes LF on every OS, the readers trim, and `test_theme` fails on a CR |
| 2 | Actions: all 33 `data-act` ids as `QAction`s with the Studio's shortcuts and menus; the native menubar on macOS, in-window elsewhere | `ui/index.html` `data-act`, `SHORTCUTS` in `ui/app.js` | a test extracts both from the page and compares them to the app's actions | 1 | |
| 3 | Session model (`engine/session`): the page's `state` in C++, with undo and redo as commands (5.6) where the page pushes undo | `ui/app.js` `state`, `params()`, `pushUndo` | the same scripted actions give byte-identical `params()` JSON (goldens from the page, headless) | 1.5 | |
| 4 | Main window: media rail (drop zone, shot list, open by path), viewer toolbar (compare, layer, probe, guides, zoom, HDR badge), transport (prev, play, next, timecode, scrub), right rail (scopes; Reconstruct, Grade, Deliver tabs; frame measurements), pipe bar; the simple and full workspaces | `ui/index.html`, the Pro-direction boards | every panel present and wired; a side-by-side screenshot review against the boards | 2.5 | |
| 5 | Scope widgets: waveform, histogram and vectorscope painted from `ScopeData` as `drawScopes` and `drawVector` draw them (zone colours, gridlines, the dashed 203 line, the clip band) | `ui/app.js` `drawScopes`, `drawVector` | a raster of the page's scope SVG and the widget's paint agree within 2 codes on 99 % of pixels | 1.5 | |
| 6 | Reconstruct and Grade panels: mode, strength, preserve, view peak, the three-band Region EV editor (the `#regions` drag behaviour), anchor and chroma carry, container and primaries | `ui/app.js` handlers | each control drives the composite, the view and `params()` as the page does (the step 3 goldens) | 2 | |
| 7 | Probe: the floating box and the rail panel at the cursor (`probeAt`, `showProbePanel`): nits, stops, baseline and model values, masks, the SDR codes | `ui/app.js` | text identical to the page's for the same pixel | 0.5 | |
| 8 | Frame measurements panel (`showMetrics`), the clip bar, the pipe bar's warning | `ui/app.js` `showMetrics`, `paintClipBar`, `updatePipe` | text identical to the page's for the same frame | 0.5 | |
| 9 | Deliver tab: a master EXR of the frame or the whole sequence as an engine background job with progress and cancel; `render_master` moves out of the CLI into `deliver` so the app and the CLI share it | `ui/server.py` `/api/master/plan` and `/api/master` | the same paths and names as the plan, and the same EXR and sidecar bytes as the Studio for a 3-frame sequence (within the master goldens' bound) | 2 | |
| 10 | Checkpoint manager and first run: find, verify and switch model packages and the runtime and device (the page's `#ckpt`, `#device`) without a restart; the first-run wizard runs the viewer's HDR card on the display it opens on | `/api/model`, `/api/checkpoints` | switching packages mid-session keeps the session; the wizard reports the display's real peak | 1.5 | |
| 11 | The rest of the page: the sheets (shortcuts, about, copy metrics, scopes and delivery), drop to open, recent shots, settings kept between runs | `ui/app.js` | every remaining `data-act` works | 0.5 | |
| 12 | Review: the whole workflow on Windows with no Python on the machine, scripted and by hand; Phase 3 exit in `STATUS.md` | | open, scrub, grade, compare, probe and master a 240-frame folder with the numbers matching the Studio's | 0.5 | |

Order: 1 to 3 first (look, actions and state are what everything else binds
to), then 4, then 5 to 8 in any order, then 9 and 10 (they need the engine
jobs), 11, 12.

The one structural change: step 9 moves the master pipeline, which is in
`cli/master.cpp` today, into a library the app can call. It goes to
`deliver/` (the stages are core and deliver already), and the CLI keeps only
argument parsing, so the layer rule stays as it is.
