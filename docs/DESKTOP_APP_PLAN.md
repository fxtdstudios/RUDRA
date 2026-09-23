# RUDRA Studio Desktop: native cross-platform plan

> 23 Sep 2026, revision 3. Phase 0 closed: GO for Phase 1 on Windows (Gates A and B pass, budgets recorded); macOS conditional on its Mac runs (`STATUS.md`, line F; `native/README.md`).
> Stack: **Qt 6, QRhi (Metal / Direct3D 12 / Vulkan, OpenGL fallback), C++20,
> LibTorch + ONNX Runtime**.
> No Python at runtime. Design: `docs/NATIVE_ARCHITECTURE.md`. Mockup: the
> "RUDRA Studio Desktop" design canvas.
> Revision 1 (Electron + Python sidecar) and revision 2 (raw OpenGL, LibTorch
> only) are superseded: raw OpenGL left macOS at GL 4.1 with no compute and no
> HDR, and LibTorch alone left every non-NVIDIA GPU on the CPU.

## 1. What we are building

A native RUDRA Studio for Windows x64, Linux x64 and macOS arm64 that does
everything the browser Studio does (`ui/`, 5.1k lines of HTML/JS/CSS over
`ui/server.py`) plus what only the CLI does today (`rudra video`, `rudra batch`),
with no Python interpreter shipped or required:

1. Footage opened by real path: native dialogs, OS drag and drop, nothing uploaded.
2. The same composite, false colour, difference, wipe, probe, scopes and pipe bar,
   rendered through **QRhi** on each OS's native GPU API: Metal on macOS,
   Direct3D 12 on Windows, Vulkan on Linux, OpenGL as the fallback.
3. SDR2HDRNet run through **LibTorch** (CUDA, MPS, CPU) from a TorchScript
   export, and through **ONNX Runtime** (DirectML, Core ML, ROCm, OpenVINO)
   from an ONNX export, so every GPU vendor is accelerated.
4. HDR10, HLG, ProRes 422 / 422 HQ / 4444 delivery with the queue, resume, QC
   and sidecar the Python implements.
5. Real HDR on the glass on all three: scRGB/HDR10 on Windows, EDR on macOS,
   Vulkan HDR on Linux where the compositor supports it; honest SDR-out otherwise.

Python stays the training stack, the export tool, and the **reference oracle**:
every C++ module is accepted only when it matches golden outputs produced by the
Python on the same inputs.

Out of scope: training in the app, model changes, mobile, web.

## 2. Architecture

```
┌──────────────────────── app/  (Qt 6 Widgets) ─────────────────────────┐
│ QMainWindow, QDockWidget rails, tabs, transport, pipe bar, QSS theme  │
│ native QMenuBar (macOS), QFileDialog, drag-drop QUrl, QShortcut table │
│ first-run wizard, checkpoint manager, render queue window             │
└───────┬───────────────────────┬──────────────────────────┬────────────┘
        │                       │                          │
┌───────▼────────┐   ┌──────────▼──────────┐   ┌───────────▼───────────┐
│ render/        │   │ infer/              │   │ deliver/              │
│ QRhi: Metal,   │   │ LibTorch: CUDA,MPS, │   │ libav decode (LGPL)   │
│ D3D12, Vulkan, │   │ CPU                 │   │ + hardware decode     │
│ GL fallback    │   │ ONNX Runtime: DML,  │   │ ffmpeg CLI encode     │
│ one GLSL → qsb │   │ CoreML, ROCm, OV    │   │ QC, sidecar, queue    │
│ HDR swapchains │   │ tiles, worker thread│   │ resume                │
└───────┬────────┘   └──────────┬──────────┘   └───────────┬───────────┘
        └───────────────────────┼──────────────────────────┘
                     ┌──────────▼───────────────────────────┐
                     │ core/  librudra, C++20, no Qt        │
                     │ decode, anchor, chroma, radiometry,  │
                     │ controls/regions, metadata (MaxCLL), │
                     │ exr (OpenEXR), aces (OpenColorIO),   │
                     │ qc, sidecar JSON, batch queue        │
                     └──────────────────────────────────────┘
        + cli/  headless `rudra` binary on librudra, diffable against the Python CLI
```

### 2.1 Model export (`tools/export_model.py`, Python, runs once per checkpoint)

- `SDR2HDRNet` is conv, `F.interpolate`, `sigmoid`, `cat`, `where`,
  `expm1`: all TorchScript-, `torch.export`- and ONNX-clean. The training-only
  paths (`binary_cross_entropy_with_logits`, loss helpers) are not in the
  exported graph.
- Export two entry points: `frame_pass` (the whole-frame heads that give
  `residual_scale` and `shadow_weight`) and `tile_pass` (the fields: log
  residual, highlight mask, shadow mask). `torch.jit.script` where control
  flow depends on mode, `trace` otherwise.
- Write one **model package** per checkpoint: `model.ts` (LibTorch),
  `model.onnx` (ONNX Runtime, opset pinned), `manifest.json` (contract version,
  source `.pt` sha256, torch and opset versions, `corpus_ev`, heads present),
  16 golden frames, the weights licence. Both graphs must match eager PyTorch
  on the goldens before the package is written.
- The app's first-run self-test re-runs the goldens on the chosen backend and
  refuses a model that drifts.
- Later, for speed: AOTInductor per GPU architecture, TensorRT engines built on
  the user's machine from the ONNX graph. Both are extra backends, not
  replacements.

### 2.2 Inference (`infer/`)

- `InferenceBackend` interface, two families behind it:

| Backend | Runtime | Windows | macOS (Apple Silicon) | Linux |
|---|---|---|---|---|
| **primary** | LibTorch | CUDA (NVIDIA) | MPS (Apple GPU) | CUDA (NVIDIA) |
| **universal** | ONNX Runtime | DirectML (any DX12 GPU: NVIDIA, AMD, Intel) | Core ML (GPU + Neural Engine) | ROCm / MIGraphX (AMD), OpenVINO (Intel) |
| always | either | CPU | CPU | CPU |

- The engine picks the best available at start-up and shows it in the status
  bar; the user can override. Every backend passes the same parity test on the
  429 bench frames before it is offered (tolerance per backend and precision in
  `tests/golden/TOLERANCES.md`).
- Worker thread owns the session/module and its device stream; the UI never
  blocks on it.
- 512 px tiles with overlap, blended with the same window as
  `training/infer_sdr2hdr.py::predict_fields`. Frame-level heads run once per
  frame, never per tile. fp32 default; fp16 opt-in, labelled in UI and sidecar.
- Read-ahead cache of decoded frames and their fields in front of the playhead.

### 2.3 Viewer (`render/`)

- Built on **QRhi**, Qt 6's rendering hardware interface (public API since
  Qt 6.6): one renderer, native API per OS.

| | Windows | macOS | Linux |
|---|---|---|---|
| QRhi backend | Direct3D 12 (D3D11 fallback) | Metal | Vulkan (OpenGL fallback) |
| Compute | yes | yes | yes |
| Float targets | RGBA16F / RGBA32F | same | same |

- Shaders: `ui/compositor.js` is GLSL ES 3.0 (`COMMON`, `COMPOSITE`,
  `DISPLAY`). Ported once to Vulkan-style GLSL 440 and compiled at build time
  by Qt's `qsb` tool into SPIR-V, HLSL, MSL and GLSL. Uniforms, units and maths
  unchanged.
- Composite target RGBA32F, so the probe and the on-screen measurements read
  the same numbers the master writes.
- Scopes (waveform, RGB histogram, vectorscope; nits, log axis) are compute
  shaders on every backend.
- `measure()` for masters and sidecars runs in librudra on the fp32 CPU
  composite, independent of GPU and driver.

### 2.4 HDR output

| OS | Swapchain | Value written | v1 |
|---|---|---|---|
| Windows | QRhi `HDRExtendedSrgbLinear` (scRGB, FP16) on D3D12; `HDR10` (PQ, 10-bit) as an option | scRGB: linear Rec.709, 1.0 = 80 nits, so `nits / 80` | **HDR out** when "Use HDR" is on |
| macOS | QRhi `HDRExtendedDisplayP3Linear` on Metal (EDR) | 1.0 = the display's SDR white; headroom read live from the screen, it changes with brightness | **HDR out** on XDR / Studio / HDR displays |
| Linux | QRhi Vulkan HDR swapchain where the compositor exposes it (KDE Plasma 6 on Wayland) | as Windows | **HDR out** where supported, verified in Phase 0 |
| any | SDR swapchain | today's PQ simulation | fallback, stated in the pipe bar |

The pipe bar states which path is live, and the "clipped on screen, not in the
master" warning uses the display's real headroom.

### 2.5 UI (`app/`)

- Qt Widgets, not QML: dense instrument UI, native menus, precise layout.
- QSS generated at build time from `ui/theme.css` custom properties, so the
  neutral surround (R = G = B), accent, gold and status colours stay one source.
  IBM Plex Mono and Plex Sans Condensed embedded (OFL).
- Every `data-act` id in `ui/index.html` becomes a `QAction` with the same
  shortcut; macOS uses the native menubar, Windows/Linux an in-window menubar
  in a frameless titlebar.
- Panels: Media rail (drop zone, shot list, probe), viewer toolbar
  (Compare, Layer, Probe, Guides, Zoom, HDR-out badge), transport with
  timecode, queue dock, right rail (scopes, Reconstruct / Grade / Deliver tabs,
  Frame measurements), pipe bar.

### 2.6 Delivery (`deliver/`)

- Decode: libavformat/libavcodec linked dynamically from an **LGPL** build,
  with hardware decode where present (NVDEC / D3D11VA on Windows, VideoToolbox
  on macOS, NVDEC / VAAPI on Linux) and software decode as the reference.
  Enforces the same input rules as `rudra.video` (progressive, square pixels,
  CFR, even dimensions, colour tags or explicit overrides; rejects HDR,
  interlaced, rotated, anamorphic, VFR).
- Encode: the **ffmpeg CLI** as a subprocess, fed 16-bit frames on a pipe
  (today: PNG spool). x265 is GPL; keeping it in a separate process keeps the
  app binary free of it. Presets, filters, tags and the `bt2020` vs `bt2020nc`
  handling ported from `rudra/delivery/video.py`. On macOS, ProRes can also be
  encoded with Apple's VideoToolbox ProRes encoder, which some delivery specs
  require; `prores_ks` stays the default elsewhere.
- QC before publish, same list as the CLI: dimensions, every timestamp, frame
  count, colour tags, HDR metadata, audio alignment, complete decode.
  Never overwrites. `.json` sidecar with checkpoint sha, settings, per-frame stats.
- Queue: `queue.json` format kept identical to `rudra.batch` so either tool can
  run or resume the other's queue. Atomic save + file lock as in `batch.py`.
- ffmpeg probe on start (`-encoders`, `-filters`: libx265, prores_ks, zscale)
  plus a 16-frame HDR10 self-test per ffmpeg binary hash, cached.

## 3. Port map

| Python / JS | Lines | C++ target | Test |
|---|---:|---|---|
| `rudra/decode.py`, `radiometry.py`, `anchor.py`, `chroma.py` | 392 | `core/decode`, `core/anchor`, `core/chroma` | golden arrays, max abs diff |
| `rudra/delivery/controls.py`, `colorspace.py`, `profiles.py` | 359 | `core/controls`, `core/colorspace` | golden arrays |
| `rudra/delivery/metadata.py`, `hdr10.py` | 320 | `core/metadata` | MaxCLL/MaxFALL exact |
| `rudra/delivery/exr.py`, `exr_io.py`, `aces.py` | 389 | `core/exr`, `core/aces` (OpenEXR, OCIO) | header attributes + pixels |
| `rudra/qc.py` | 213 | `core/qc` | same pass/fail on fixtures |
| `rudra/video.py`, `rudra/delivery/video.py` | 978 | `deliver/` | existing video test fixtures |
| `rudra/batch.py` | 169 | `core/queue` | cross-run: Python queue resumed by C++ and back |
| `ui/sequence.py`, `server.py` (measure, sequence, master) | ~600 | `core/sequence`, `core/master` | same master bytes |
| `ui/compositor.js` | 655 | `render/` GLSL 440 via `qsb` | probe values vs browser Studio, on every QRhi backend |
| `ui/app.js`, `shell.js`, `index.html`, CSS | ~3,000 | `app/` | UI smoke tests (Qt Test) |

About 3,300 lines of Python and 3,650 of JS/HTML/CSS. The existing pytest suite
becomes the generator of the golden files: a `pytest --emit-golden` pass writes
inputs and outputs to `tests/golden/`, GoogleTest reads them.

## 4. Build, packaging, runtime

- CMake presets + vcpkg manifest: Qt 6 (LTS, 6.8 or later) with Qt Shader Tools,
  OpenEXR, OpenColorIO, FFmpeg (LGPL features), nlohmann-json, spdlog,
  GoogleTest. LibTorch from the official archives (not vcpkg): cxx11 ABI on
  Linux, MSVC build on Windows, arm64 on macOS. ONNX Runtime from its official
  packages: DirectML build on Windows, Core ML on macOS, ROCm / OpenVINO builds
  on Linux.
- Shaders compiled at build time by `qsb` into one `.qsb` per shader holding
  SPIR-V, HLSL, MSL and GLSL; no runtime shader compilation.
- Compilers: MSVC 2022, clang (macOS), gcc 12+ (Linux). C++20.
- Deploy: `windeployqt` + NSIS or WiX; `macdeployqt` + DMG, Developer ID +
  notarization; `linuxdeploy` AppImage + .deb.
- **Compute packs.** Installers ship ONNX Runtime with the platform's
  universal provider (DirectML on Windows, Core ML on macOS) plus CPU, so every
  machine is GPU-accelerated from the first launch. LibTorch MPS ships in the
  macOS app. The NVIDIA pack (LibTorch CUDA + cuDNN, ~2.4 GB) downloads on
  first run when an NVIDIA GPU is found, resumable, sha-verified, versioned
  with the app. A Windows offline installer includes it.
- Checkpoints: model packages (`.ts` + `.onnx` + manifest) listed from `checkpoints/models.json`, verified
  against `SHA256SUMS`, golden-frame self-test on first load. The non-commercial
  weights licence (`checkpoints/LICENSE`) accepted before any download.
  `_invalid_*` never listed.
- Updates: app via a signed appcast (Sparkle on macOS, WinSparkle on Windows,
  AppImage zsync on Linux); compute packs only re-download when the LibTorch
  version changes.
- CI: GitHub Actions matrix (windows-2022, macos-14, ubuntu-22.04). Stages:
  pytest (emits golden) → CMake build → GoogleTest parity → Qt Test UI smoke →
  package → packaged smoke (open `ui/assets/cinematic_hdr_sunset.png` on CPU,
  master EXR, compare with the Python master).

## 5. Repository layout

```
native/
  CMakeLists.txt  CMakePresets.json  vcpkg.json
  core/      librudra (no Qt)
  infer/     InferenceBackend: LibTorch (CUDA, MPS, CPU), ONNX Runtime (DML, CoreML, ROCm, OpenVINO)
  render/    QRhi renderer, shaders/*.glsl compiled by qsb
  deliver/   libav decode, ffmpeg encode, QC
  app/       Qt Widgets, qss generated from ../../ui/theme.css
  cli/       headless rudra
  tests/     GoogleTest + Qt Test, reads ../../tests/golden
  packaging/ nsis, dmg, appimage, icons from ui/assets/rudra-mark.png
tools/export_model.py     writes model.ts + model.onnx + manifest + goldens
```

`rudra/`, `ui/`, `training/` stay as they are.

## 6. Phases

| # | Phase | Work | Exit gate | Est. |
|---|---|---|---|---|
| 0 | Export + spike | Model package export (TorchScript + ONNX); LibTorch and ONNX Runtime load it in a small C++ app; QRhi HDR window on Windows (D3D12 scRGB) and macOS (Metal EDR) | Both runtimes match Python on the 429 bench frames (CPU fp32, log space ≤ 1e-5); a 1,000-nit patch measured above SDR white on both OSes | 1.5 w |
| 1 | librudra | Port decode, anchor, chroma, controls, colorspace, metadata, exr, aces, qc, queue; golden emitter in pytest | Every module passes its golden test on all three OSes | 3 w |
| 2 | QRhi viewer | Shaders ported and compiled by `qsb`; composite, display, false colour, difference, wipe, probe, guides; compute scopes; HDR swapchains; D3D12, Metal, Vulkan, GL fallback | Probe and measurements equal the browser Studio's on every backend; readback parity with `composite.cpp` | 4 w |
| 3 | Qt UI | Main window, rails, tabs, transport, pipe bar, menus, shortcuts, QSS, first-run wizard, checkpoint manager | Full Studio workflow with no Python installed | 3 w |
| 4 | Deliver | libav decode with hardware decode, ffmpeg encode, VideoToolbox ProRes on macOS, five presets, QC, sidecar, queue window, resume | 3-clip queue killed mid-clip resumes; every output passes QC; C++ and Python resume each other's queue | 2.5 w |
| 5 | Package | Installers, signing, notarization, compute packs, updater, CI matrix | Signed builds install and update on all three | 1.5 w |
| 6 | Harden | Every inference backend through the parity test (CUDA, MPS, DirectML on NVIDIA/AMD/Intel, Core ML, ROCm, OpenVINO); 8K frames, long queues, VRAM exhaustion, missing GPU, low disk; HDR matrix (Windows GPUs, XDR, KDE Wayland) | No open P0; backend matrix published in the docs | 3 w |

About 18 weeks for one engineer, two more than revision 2: one for QRhi in the
viewer, one for the ONNX backends in hardening. Phase 0 is the go/no-go: if
model parity fails, the fix is in the export, before any porting starts.

## 7. Acceptance criteria for v1

- [ ] Installs and runs on Windows 11, Ubuntu 22.04, macOS 14+ (Apple Silicon) with no Python present.
- [ ] GPU-accelerated inference on NVIDIA, AMD, Intel and Apple GPUs, each backend passing the parity test.
- [ ] HDR out on Windows HDR displays and on macOS EDR displays; Linux where the compositor supports it.
- [ ] Master EXR from the native app matches the Python master within 1 half-float ULP on CPU.
- [ ] Every ported module passes its golden test on every OS in CI.
- [ ] Drag a 4K EXR folder from Explorer/Finder: opens by path, first frame under 2 s warm.
- [ ] All five video presets pass the same QC as the CLI; sidecar written; never overwrites.
- [ ] Queue survives quit, crash and reboot; `queue.json` interchangeable with `rudra batch`.
- [ ] Pipe bar reports HDR out or SDR out truthfully; on-screen clip warning uses real headroom.
- [ ] Weights licence accepted before download; model sha and golden self-test before first use.

## 8. Decisions for Ahmed

1. **Model format:** TorchScript + ONNX in every package (decided, rev 3); AOTInductor / TensorRT later as speed backends.
2. **Renderer:** QRhi with Metal / D3D12 / Vulkan and OpenGL fallback (decided, rev 3).
3. **Qt licence:** LGPL with dynamic linking (fine for a closed app if Qt is relinkable) or a commercial Qt licence.
4. **Browser Studio:** keep `ui/` as the reference implementation during the port (recommended), or retire it at v1.
5. **Distribution:** public releases, or FXTD-internal while the weights are non-commercial.

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| TorchScript export differs from eager | wrong reconstruction | Phase 0 parity gate on 429 frames before any porting |
| Two implementations drift (Python keeps changing) | app and paper disagree | golden files regenerated in CI on every commit; a failing parity test blocks merge |
| QRhi is newer than raw GL; its public API carries limited compatibility guarantees between Qt minors | a Qt upgrade touches `render/` | pin Qt LTS; all QRhi use confined to `render/` behind `ViewerBackend` |
| ONNX backends differ numerically (DirectML, Core ML fp16 paths) | small drift per vendor | per-backend tolerance in the parity test; fp32 where the provider allows; LibTorch stays the reference |
| Linux HDR depends on the desktop compositor | no HDR on some distros | detect at start-up, SDR-out with the warning |
| LibTorch CUDA size (~2.4 GB) | slow first run | CPU works immediately; resumable pack; offline installer |
| libav / ffmpeg version drift (the 7.x `bt2020` break) | bad encodes | per-binary self-test with full QC before first delivery |
| Port effort (~7k lines) underestimated | late v1 | librudra first, CLI diff against Python gives early signal |
