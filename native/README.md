# RUDRA native

The desktop RUDRA for Windows, macOS and Linux: C++20, Qt 6, a QRhi viewer on
Direct3D 12, Metal, Vulkan or OpenGL, and inference through LibTorch or ONNX
Runtime. No Python at runtime. Design: [`docs/NATIVE_ARCHITECTURE.md`](../docs/NATIVE_ARCHITECTURE.md);
plan: [`docs/DESKTOP_APP_PLAN.md`](../docs/DESKTOP_APP_PLAN.md); the composite
it implements: [`docs/composite.spec.md`](../docs/composite.spec.md).

The Python in this repo is the oracle. Every native module is tested against
arrays the Python it replaces writes (principle P4), so the app and the paper
cannot quietly disagree.

## Layout

```
platform   Result<T>, hashes, .npy reader, running a program       (no deps)
core       colour types, Image<Space>, baseline, tiling, manifest,
           composite, gamut, master chain, measurements           (platform)
infer      InferenceBackend: LibTorch, ONNX Runtime, the tiler    (core)
media      still decode (OpenCV imgcodecs), sequence open by path;
           video probe, decode on a pipe, the PNG spool, alpha    (core)
render     the QRhi viewer: GpuCompositor and ViewerWindow (rudra_render_gpu);
           probe/ holds rudra-hdr-probe, rudra-gpu-parity and
           rudra-viewer-check                                     (core, Qt)
deliver    EXR/ACES/OCIO writers, metadata sidecars, QC, queue;
           video mastering, the encoder command, video QC         (core)
video      convert_video: the predictor, the pipeline end to end
           (Phase 4); no Qt, so rudra-native links it (infer, media, deliver)
engine     FrameEngine: the InferActor, generations, cancellation,
           read-ahead and the frame cache; actions: the Studio's
           menus, keys and commands; session: its grading state,
           undo and params(); measure: computeStats               (below)
cli        rudra-native: version | info | diff | bench | master | master-check |
           master-compare | video
           (never Qt, never render)
app        the Qt application: main_window.cpp and layout.cpp (rudra_app_ui),
           theme.cpp (the look of ui/theme.css; fonts/ holds IBM
           Plex, OFL), tests/ (rudra_app_tests, offscreen Qt)      (everything)
tests      GoogleTest against the goldens in tests/golden/
```

A layer includes only headers from layers below it; Qt, QRhi, GPU APIs and
the inference runtimes appear only where the architecture puts them.
`python tools/check_layers.py` enforces it, in CI too.

## Build

Headless core and tests (any OS, CMake 3.24+, a C++20 compiler):

```
cd native
cmake --preset core
cmake --build --preset core
ctest --preset core
```

Options:

| Option | Default | What |
|---|---|---|
| `RUDRA_BUILD_TESTS` | ON | the GoogleTest suite |
| `RUDRA_BUILD_CLI` | ON | `rudra-native` |
| `RUDRA_WITH_LIBTORCH` | OFF | LibTorch backend (`CMAKE_PREFIX_PATH` to LibTorch or `torch.utils.cmake_prefix_path`) |
| `RUDRA_TORCH_ROOT` | empty | import LibTorch or a pip torch folder directly, without TorchConfig: a CUDA torch then needs no CUDA toolkit to build |
| `RUDRA_WITH_OPENCV` | OFF | still decode in `media/` (OpenCV core + imgcodecs, the decoder `rudra/decode.py` uses) |
| `RUDRA_WITH_ONNXRUNTIME` | OFF | ONNX Runtime backend (`ONNXRUNTIME_ROOT` with `include/`, `lib/`) |
| `RUDRA_BUILD_APP` | OFF | the Qt shell (Qt 6.4+); `RUDRA --theme-check out.json` reports the fonts, weights and style it resolved, `RUDRA --grab out.png` saves the window |
| `RUDRA_BUILD_RENDER` | OFF | `rudra_render_gpu`: the QRhi composite (Qt 6.6+ with Qt Shader Tools) |
| `RUDRA_BUILD_HDR_PROBE` | OFF | `rudra-hdr-probe` and `rudra-gpu-parity` (turns `RUDRA_BUILD_RENDER` on) |
| `RUDRA_TEST_PACKAGE` | empty | a model package: adds its golden frames to `ctest` |

## Model package

`python tools/export_model.py checkpoints/sdr2hdr_shadow_v1.pt` writes
`dist/models/sdr2hdr_shadow_v1/`: `model.ts` (TorchScript), `model.frame.onnx`
and `model.tile.onnx`, `manifest.json` (contract 1.0), golden frames and the
licence. The package is written only if every graph matches eager PyTorch on
the golden frames at the manifest's tolerances.

```
rudra-native info dist/models/sdr2hdr_shadow_v1
rudra-native diff dist/models/sdr2hdr_shadow_v1 --runtime all --device cpu
rudra-native bench dist/models/sdr2hdr_shadow_v1 --runtime libtorch --device cuda --size 1920x1080
rudra-native master dist/models/sdr2hdr_shadow_v1 plate.png --out plate.exr [--params '{"container": "linear"}']
rudra-native master-check dist/models/sdr2hdr_shadow_v1 native/tests/golden/master
rudra-native video dist/models/sdr2hdr_shadow_v1 clip.mp4 --output master_hdr10.mp4 [--format hlg|prores422|prores422hq|prores4444] [--audio copy|aac|none] [--runtime ...] [--device ...]
```

`master` is the Studio's Master button without the Studio: decode, fields,
composite, Region EV, anchor, chroma carry, measure, ACES (or linear) EXR and
its sidecar. `--params` takes the Studio's master parameters as JSON
(`recovery_mode`, `strength`, `preserve_outside`, `regions`, `anchor`,
`carry_chroma`, `source_space`, `container`, ...). Needs `RUDRA_WITH_OPENCV`.

`video` is `rudra video` (rudra/video.py) without Python: the same options,
checks, report and QC, with ffmpeg and ffprobe on PATH (libx265, prores_ks and
zscale). The master is published beside its `.json` report only after QC.

## Goldens

| Script | Writes | Tested by |
|---|---|---|
| `tools/emit_core_golden.py` | `tests/golden/core/`: baseline, curve, tile weights | `test_core.cpp` |
| `tools/emit_composite_golden.py` | `tests/golden/composite/`: composite, Region EV, anchor, chroma, AP0, master chain, measurements | `test_composite.cpp` |
| `tools/emit_decode_golden.py` | `tests/golden/decode/`: 17 image fixtures and their decoded floats | `test_decode.cpp` |
| `tools/emit_delivery_golden.py` | `tests/golden/delivery/`: grades, PQ/HLG, a 3-shot sidecar set, EXR/ACES files, the OCIO config | `test_delivery.cpp` (files compared byte for byte) |
| `tools/emit_master_golden.py` | `tests/golden/master/`: three stills and the Studio's masters of them | `rudra-native master-check` (a ctest) |
| `tools/emit_qc_golden.py` | `tests/golden/qc/`: six reconstructions, their QC reports and report text | `test_qc.cpp` |
| `tools/emit_queue_golden.py` | `tests/golden/queue/`: a queue project, its state after a run and a resume, nine refusals | `test_queue.cpp` (state compared byte for byte) |
| `tools/emit_sequence_golden.py` | `tests/golden/sequence/`: folder layouts and what `Sequence.open` made of them | `test_sequence.cpp` |
| `tools/emit_viewer_golden.py` | `tests/golden/viewer/`: the browser Studio's composites, views, probes, reductions, metrics and scopes on two frames (headless Chromium, Playwright) | `test_viewer.cpp` |
| `tools/emit_viewport_golden.py` | `tests/golden/viewport/`: where the Studio's own CSS layout puts the frame through fit, zoom and pan scripts | `test_viewport.cpp` |
| `tools/export_model.py` | the package's `golden/` | `rudra-native diff` |

`python tools/emit_golden.py` runs every emitter; re-run it and commit when the
Python it reads changes. CI does the same and fails on drift.

## Gates

**Gate A, model parity.** `scripts/NATIVE_GATE_A.ps1` on a Windows machine
exports the package, builds with LibTorch and ONNX Runtime (DirectML), and
runs the golden frames on every backend the box has: LibTorch CPU and CUDA,
ONNX Runtime CPU and DirectML, taking LibTorch from the Python's own torch (no
CUDA toolkit needed). `-BenchDir` adds the bench frames. Report in
`reports/`. Both Windows gate scripts need Visual Studio 2022 or 2026 (or the
Build Tools) with the C++ tools; `-InstallBuildTools` installs the Build Tools
with winget when none is found.

**Gate B, HDR out.** `rudra-hdr-probe` opens a QRhi window with a test card
(patches at 100, 203, 400, 600, 1 000 and 2 000 nits and a log ramp to
10 000), reads the swapchain back and reports what reached it:

```
scripts/NATIVE_GATE_B.ps1          # Windows: D3D12 scRGB and HDR10, D3D11 scRGB
scripts/native_gate_b.sh           # macOS: Metal EDR (Display P3, sRGB); Linux: Vulkan, GL
FRAMES=0 scripts/native_gate_b.sh  # keep the window open and look; Esc quits
```

The Windows script lists the displays first; `-Screen N` opens the probe on
another one when the HDR display is not the primary. When a display reports
HDR luminance but the swapchain falls back to SDR, the report says HDR is off
for that display.

PASS means the swapchain carried the 1 000-nit patch at least a stop above SDR
white; the glass is then checked by eye or meter. On an SDR swapchain it
reports SDR and FAIL rather than passing a clipped card.

**The Phase 3 exit (step 12).** `RUDRA --workflow-check report.json --package <pkg> --frames <folder> [--backend onnxruntime/directml] [--out <dir>]` runs the Studio workflow in the window and writes a JSON report (open, scrub, grade, compare, measure, master); `rudra-native master-compare <pkg> report.json --runtime <r> --device <d>` holds its masters to the CLI's. On Windows, `.\scripts\NATIVE_PHASE3_EXIT.ps1 -Frames <folder>` builds, deploys and runs both with no Python on the PATH.

**Model packages in the app (Phase 3 step 10).** `RUDRA` with no arguments opens the package used last, else the one the catalog picks from: `RUDRA_PACKAGE_ROOTS` (folders separated by `;` on Windows, `:` elsewhere), the folders added in File > Model packages, `<app>/models`, and the per-user `models` folder. `python tools/export_model.py <ckpt> --out <folder>` writes a package there with `models.json` beside it. The first time a package runs on a backend its golden frames are run (the same check as `rudra-native diff`) and a package that drifts is refused. The first start shows the display check (Help > Check the display and the model) once.

**The viewer window (Phase 2 step 9).** `rudra-viewer-check --api <api>`
opens the real viewer window, puts a golden frame in as fields and reads its
swapchain back at fit and at 2x: every picture pixel within 1 code of
`core/view.cpp` on `core/composite.cpp`, the surround exact. `--card` is Gate
B through the real display pass: a card of 10 to 2 000 nits on the HDR
swapchain, each patch at its own luminance up to the display's peak and
clipped above it. Both gate B scripts run both on every API. A grab whose
frame does not come is asked for again once a second (`nudges` in the
report); a run that stalls writes a `TIMEOUT` report with the viewer's frame
counters. `NATIVE_GATE_B.ps1` also builds the app and runs
`RUDRA --theme-check`: Plex must resolve under its family at every weight.

**Day 8, GPU composite parity.** `rudra-gpu-parity --api <api>` renders the
composite shader offscreen on this GPU for every case in the composite
goldens and reads it back: into RGBA32F against `composite.cpp` (atol 1e-6,
rtol 2e-4) and into RGBA16F, the viewer's format, within 2 half-float ulp.
Then the display pass (`display.frag`, Phase 2 step 4): seven views of each
frame into RGBA8 against `core/view.cpp`, within 1 code ("view codes" in the
gate table is the worst difference), and the HDR paths (scRGB, HDR10, EDR,
step 5) into RGBA32F and RGBA16F. Both gate B scripts run it on every API the machine has, with `--bench`: one
composite pass, and a composite plus display pass (a slider move), timed at
1080p and 4K (QRhi GPU timestamps). `rudra-native bench-scopes` times the
viewer's CPU measurements and scopes; Gate A runs it. Gate A
times inference at 1080p on every backend that passed (`rudra-native bench`).

## Status

| Piece | State |
|---|---|
| Model package export, TorchScript and ONNX | done: TorchScript bit-exact with eager, ONNX within tolerance |
| Core types, baseline, tiling | done: bit-exact with the Python |
| LibTorch and ONNX Runtime CPU backends, tiler | done: Gate A passes on CPU |
| GPU execution providers (CUDA, DirectML, Core ML, ROCm, OpenVINO) | CUDA and DirectML pass; Core ML, ROCm, OpenVINO written, not yet run |
| Composite, Region EV, master chain, AP0 | done: against `predict_image` and the `_render_master` stages |
| Measurements (MaxRGB stats, MaxCLL/MaxFALL, Studio QC) | done |
| Gate A | Windows passes on all four: LibTorch CPU and CUDA, ONNX Runtime CPU and DirectML |
| Gate B | Windows passes: D3D12 scRGB and HDR10, D3D11 scRGB on a 418-nit HDR display; XDR Mac open |
| Composite shader in GLSL 440, readback parity | done on Windows: D3D12, D3D11, Vulkan, OpenGL (fp16 1 half ulp); Metal open |
| Budgets (NATIVE_ARCHITECTURE.md 6.6) | recorded: composite 0.11 ms 1080p, 0.51 ms 4K; inference 150 ms (DirectML) and 172 ms (CUDA) at 1080p fp32 |
| Phase 0 | closed 23 Sep 2026: GO on Windows, macOS conditional on MPS/Core ML, Metal EDR and Metal parity runs |
| Still decode | done: bit-exact with `rudra/decode.py` on 17 fixtures |
| Grade, HDR10/HLG, metadata, EXR/ACES/OCIO | done: sidecars, EXRs and OCIO config byte-identical with the Python |
| `rudra-native master` | done: Studio-identical master (1 half ulp, same header and sidecar) on LibTorch and ONNX Runtime |
| QC, queue, sequence open | done: same QC report text, queue state byte-identical and resumable across Python and C++, same frame order and messages |
| Phase 1 (librudra) | steps 1 to 10 of 11 done (`NATIVE_ARCHITECTURE.md` section 14) |
| Phase 2 (QRhi viewer) | steps 1 to 11 of 13 done: the browser Studio as oracle, `docs/view.spec.md`, the display pass SDR and HDR, the reduction ladder, probe, measurements and scopes equal to the browser's, the viewer window in the app (Gate B through it passes on Windows), the frame path through the engine, and the guides (section 15) |
| Phase 3 (Qt UI) | the window in the Pro-direction look (`app/theme/pro.css`, Geist; the boards on the RUDRA Studio Desktop canvas), previews at the Studio's 1600 (`fit_max_side`); steps 1 to 11 of 12 done, step 12 scripted and passing on Linux (`RUDRA --workflow-check`, `rudra-native master-compare`, `scripts/NATIVE_PHASE3_EXIT.ps1` for the Windows run): the rest of the page (`core/js_json` and `core/copy_texts`, the page's clipboard texts byte for byte, `test_copy_texts`; the sheets, drop to open, Open recent and settings kept between runs, `AppCopy`, `AppOpen`, `AppSettings`); the checkpoint manager and the first run (`engine/model_catalog`, the server's model discovery for packages, `test_model_catalog`; `infer/self_test`, the goldens on first load; `app/model_dialogs`; `core/hdr_card`, the card and what the display can show); the Deliver tab's master EXR as a background job with progress and a stop (`deliver/master`, the Studio server's render plan and refusals, `test_render_plan`; `engine/master_job`, `test_master_job`), the master moved out of the CLI so the app and `master-check` share it; the probe and the frame measurements (`core/readouts`, the page's read-outs word for word; `engine/measure`, computeStats on a worker); the Reconstruct and Grade panels with the Region EV editor (`app/region_editor`, the page's gestures replayed on the widgets); the scope widgets (`core/scope_draw`, the page's scope SVG element for element; `app/scope_widgets`); the main window as the page lays it out (`app/layout.cpp`, checked against the page rendered headless in five states); the session with undo, `params()` byte for byte with the page (`engine/session`, `test_session`); the Studio's actions, menus and keys (`engine/actions`, `test_actions`, and the real menubar in `rudra_app_tests`); the style sheet generated from `ui/theme.css` (`cmake/rudra_theme.cmake`, the `rudra_theme` target), checked by `test_theme` (every colour the theme's, greys neutral, no colour in the app's C++), Plex embedded and checked at run time by `--theme-check`, which the Qt shell CI job and `NATIVE_GATE_B.ps1` run (section 16) |
| Video decode, encode | Phase 4 (section 17): steps 1 to 7 of 12 done: the probe, clock and input contract equal to `rudra/video.py` on 25 cases (`media/video_probe`, `platform/process`), the decoder's command and frames equal to the Python's (`media/video_decode`), and the video predictor with its shadow smoother within 2.7e-6 of eager PyTorch with the same cuts (`core/video_predict`, `video/predictor`), and the mastering and 16-bit PNG spool with MaxCLL and MaxFALL as the Python computes them (`deliver/video_master`, `media/png16`), the encoder's command and publishing (`deliver/video_encode`), export QC with the alpha check (`deliver/video_qc`, `media/video_alpha`, `video/qc`), and `rudra-native video` end to end, HDR10 and HLG masters byte-identical to the Python's (`video/convert`) |
