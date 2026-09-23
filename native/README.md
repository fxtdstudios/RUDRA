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
media      decode (interfaces)                                    (core)
render     the QRhi viewer; probe/ is Gate B's rudra-hdr-probe    (core, Qt)
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
| `RUDRA_WITH_ONNXRUNTIME` | OFF | ONNX Runtime backend (`ONNXRUNTIME_ROOT` with `include/`, `lib/`) |
| `RUDRA_BUILD_APP` | OFF | the Qt shell (Qt 6.4+) |
| `RUDRA_BUILD_HDR_PROBE` | OFF | `rudra-hdr-probe` (Qt 6.6+ with Qt Shader Tools) |
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
```

## Goldens

| Script | Writes | Tested by |
|---|---|---|
| `tools/emit_core_golden.py` | `tests/golden/core/`: baseline, curve, tile weights | `test_core.cpp` |
| `tools/emit_composite_golden.py` | `tests/golden/composite/`: composite, Region EV, anchor, chroma, AP0, master chain, measurements | `test_composite.cpp` |
| `tools/export_model.py` | the package's `golden/` | `rudra-native diff` |

Re-run the script and commit when the Python it reads changes.

## Gates

**Gate A, model parity.** `scripts/NATIVE_GATE_A.ps1` on a Windows machine
exports the package, builds with LibTorch and ONNX Runtime (DirectML), and
runs the golden frames on every backend the box has: LibTorch CPU and CUDA,
ONNX Runtime CPU and DirectML. `-BenchDir` adds the bench frames. Report in
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

PASS means the swapchain carried the 1 000-nit patch at least a stop above SDR
white; the glass is then checked by eye or meter. On an SDR swapchain it
reports SDR and FAIL rather than passing a clipped card.

## Status

| Piece | State |
|---|---|
| Model package export, TorchScript and ONNX | done: TorchScript bit-exact with eager, ONNX within tolerance |
| Core types, baseline, tiling | done: bit-exact with the Python |
| LibTorch and ONNX Runtime CPU backends, tiler | done: Gate A passes on CPU |
| GPU execution providers (CUDA, DirectML, Core ML, ROCm, OpenVINO) | written; Gate A on GPU runs on the Windows box |
| Composite, Region EV, master chain, AP0 | done: against `predict_image` and the `_render_master` stages |
| Measurements (MaxRGB stats, MaxCLL/MaxFALL, Studio QC) | done |
| Gate B probe | built and verified on the SDR fallback; waiting on a Windows HDR display and an XDR Mac |
| Composite shader in GLSL 440, readback parity | next (day 8) |
| Decode, encode, engine, viewer, app | Phase 1 onward |
