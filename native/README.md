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
platform   Result<T>, hashes, .npy reader                         (no deps)
core       colour types, Image<Space>, baseline, tiling, manifest,
           composite, gamut, master chain, measurements           (platform)
infer      InferenceBackend: LibTorch, ONNX Runtime, the tiler    (core)
media      still decode (OpenCV imgcodecs); video in Phase 4       (core)
render     the QRhi composite (rudra_render_gpu); probe/ holds
           rudra-hdr-probe and rudra-gpu-parity                   (core, Qt)
deliver    encode and write (interfaces)                          (core)
engine     jobs, generations, priorities                          (below)
cli        rudra-native: version | info | diff                    (never Qt, never render)
app        the Qt application                                     (everything)
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
| `RUDRA_BUILD_APP` | OFF | the Qt shell (Qt 6.4+) |
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
```

## Goldens

| Script | Writes | Tested by |
|---|---|---|
| `tools/emit_core_golden.py` | `tests/golden/core/`: baseline, curve, tile weights | `test_core.cpp` |
| `tools/emit_composite_golden.py` | `tests/golden/composite/`: composite, Region EV, anchor, chroma, AP0, master chain, measurements | `test_composite.cpp` |
| `tools/emit_decode_golden.py` | `tests/golden/decode/`: 17 image fixtures and their decoded floats | `test_decode.cpp` |
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

**Day 8, GPU composite parity.** `rudra-gpu-parity --api <api>` renders the
composite shader offscreen on this GPU for every case in the composite
goldens and reads it back: into RGBA32F against `composite.cpp` (atol 1e-6,
rtol 2e-4) and into RGBA16F, the viewer's format, within 2 half-float ulp.
Both gate B scripts run it on every API the machine has, with `--bench`: one
composite pass timed at 1080p and 4K into RGBA16F (QRhi GPU timestamps). Gate A
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
| Phase 1 (librudra) | steps 1 and 2 of 11 done (`NATIVE_ARCHITECTURE.md` section 14) |
| Video decode, encode, engine, viewer, app | Phase 1 onward |
