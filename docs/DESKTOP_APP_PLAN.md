# RUDRA Studio Desktop: native cross-platform plan

> 23 Sep 2026, revision 2. Proposal, not started. Stack: **Qt 6, OpenGL, C++20,
> LibTorch**. No Python at runtime. Mockup: the "RUDRA Studio Desktop" design
> canvas (architecture, main window, first run, deliver dialog, render queue).
> Revision 1 (Electron + Python sidecar) is superseded.

## 1. What we are building

A native RUDRA Studio for Windows x64, Linux x64 and macOS arm64 that does
everything the browser Studio does (`ui/`, 5.1k lines of HTML/JS/CSS over
`ui/server.py`) plus what only the CLI does today (`rudra video`, `rudra batch`),
with no Python interpreter shipped or required:

1. Footage opened by real path: native dialogs, OS drag and drop, nothing uploaded.
2. The same composite, false colour, difference, wipe, probe, scopes and pipe bar,
   rendered by OpenGL.
3. SDR2HDRNet run through **LibTorch** (C++), from a TorchScript export.
4. HDR10, HLG, ProRes 422 / 422 HQ / 4444 delivery with the queue, resume, QC
   and sidecar the Python implements.
5. Real HDR on the glass where OpenGL can do it (Windows), honest SDR-out elsewhere.

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
│ viewer/        │   │ infer/              │   │ deliver/              │
│ QOpenGLWidget  │   │ LibTorch            │   │ libav decode (LGPL)   │
│ GLSL 410/450   │   │ torch::jit::load    │   │ ffmpeg CLI encode     │
│ RGBA16F        │   │ tiles, CUDA stream  │   │ QC, sidecar, queue    │
│ scopes compute │   │ worker thread       │   │ resume                │
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

### 2.1 Model export (`tools/export_torchscript.py`, Python, runs once per checkpoint)

- `SDR2HDRNet` is conv, `F.interpolate`, `sigmoid`, `cat`, `where`,
  `expm1`: all TorchScript- and `torch.export`-clean. The training-only paths
  (`binary_cross_entropy_with_logits`, loss helpers) are not in the exported graph.
- Export the inference forward that `ui/server.py` calls (fields: residual,
  highlight mask, shadow mask), with `torch.jit.script` where control flow
  depends on mode, `trace` otherwise.
- Write `sdr2hdr_shadow_v1.ts` + `sdr2hdr_shadow_v1.ts.json` (source `.pt`
  sha256, export torch version, input contract, `corpus_ev`) and add both to
  `SHA256SUMS`.
- Emit golden tensors for 16 fixed bench frames; the app's first-run self-test
  re-runs them and refuses a model that drifts.
- Optional later: `torch.export` + AOTInductor package per GPU architecture for
  speed. TorchScript first because it loads on any LibTorch build of the same
  version with no per-arch compile.

### 2.2 Inference (`infer/`)

- One LibTorch version pinned across export and app (same minor as training).
- Worker `QThread` owns the module and a CUDA stream; UI never blocks on it.
- 512 px tiles with overlap, like `rudra.video` (`--tile-size 0` equivalent
  for untiled when VRAM allows). fp32 default, fp16 autocast as an option that
  is labelled in the status bar and sidecar.
- Read-ahead cache of N decoded frames and their fields in front of the
  playhead (today's server does the same).
- Devices: CUDA on Windows/Linux, CPU everywhere. MPS on macOS only after a
  parity test passes; until then macOS is CPU.

### 2.3 Viewer (`viewer/`)

- `ui/compositor.js` is already GLSL ES 3.0 (`COMMON`, `COMPOSITE`, `DISPLAY`
  programs, `uSdr`, `uFields`, `uShadow`, region uniforms). Port to GLSL 410
  core as one source with `#version` injected per platform; uniforms, units and
  maths unchanged.
- Textures RGBA16F; the composite target RGBA32F so the probe and measurement
  read exactly what the master writes.
- Scopes (waveform, RGB histogram, vectorscope, all nits, log axis): compute
  shaders on GL 4.3+ (Windows, Linux); CPU path on macOS (GL 4.1 has no compute).
- `measure()` (MaxCLL / MaxFALL per CTA-861.3, peak, P99, share above 1,000
  nits, share clipped) runs in librudra on the composited float buffer, the
  same code the master uses, as today.

### 2.4 HDR output with OpenGL

| OS | Path | v1 |
|---|---|---|
| Windows | `QSurfaceFormat` 16-bit float RGBA + scRGB linear colour space via WGL float pixel format; values above 1.0 reach the HDR display when "Use HDR" is on | **HDR out**, verified per driver in Phase 0 |
| macOS | GL deprecated, capped at 4.1; EDR only through the legacy NSOpenGL surface | **SDR out**, today's PQ simulation |
| Linux | no dependable HDR path for GL | **SDR out** |

The pipe bar states which one is live, and the "clipped on screen, not in the
master" warning uses the real output headroom. Keeping one GLSL source means a
Metal backend (via QRhi) can be added for macOS later without touching the maths.

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

- Decode: libavformat/libavcodec linked dynamically from an **LGPL** build.
  Enforces the same input rules as `rudra.video` (progressive, square pixels,
  CFR, even dimensions, colour tags or explicit overrides; rejects HDR,
  interlaced, rotated, anamorphic, VFR).
- Encode: the **ffmpeg CLI** as a subprocess, fed 16-bit frames on a pipe
  (today: PNG spool). x265 is GPL; keeping it in a separate process keeps the
  app binary free of it. Presets, filters, tags and the `bt2020` vs `bt2020nc`
  handling ported from `rudra/delivery/video.py`.
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
| `ui/compositor.js` | 655 | `viewer/` GLSL | probe values vs browser Studio |
| `ui/app.js`, `shell.js`, `index.html`, CSS | ~3,000 | `app/` | UI smoke tests (Qt Test) |

About 3,300 lines of Python and 3,650 of JS/HTML/CSS. The existing pytest suite
becomes the generator of the golden files: a `pytest --emit-golden` pass writes
inputs and outputs to `tests/golden/`, GoogleTest reads them.

## 4. Build, packaging, runtime

- CMake presets + vcpkg manifest: Qt 6 (LTS), OpenEXR, OpenColorIO, FFmpeg
  (LGPL features), nlohmann-json, spdlog, GoogleTest. LibTorch from the official
  archives (not vcpkg): cxx11 ABI on Linux, MSVC build on Windows, arm64 on macOS.
- Compilers: MSVC 2022, clang (macOS), gcc 12+ (Linux). C++20.
- Deploy: `windeployqt` + NSIS or WiX; `macdeployqt` + DMG, Developer ID +
  notarization; `linuxdeploy` AppImage + .deb.
- **Compute packs.** Installers ship LibTorch CPU (~200 MB) so the app works
  immediately. The CUDA 12.x pack (LibTorch CUDA + cuDNN, ~2.4 GB) downloads on
  first run into user data, resumable, sha-verified, versioned with the app.
  A Windows offline installer includes it.
- Checkpoints: `.ts` files listed from `checkpoints/models.json`, verified
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
  infer/     LibTorch wrapper
  viewer/    QOpenGLWidget, shaders/*.glsl
  deliver/   libav decode, ffmpeg encode, QC
  app/       Qt Widgets, qss generated from ../../ui/theme.css
  cli/       headless rudra
  tests/     GoogleTest + Qt Test, reads ../../tests/golden
  packaging/ nsis, dmg, appimage, icons from ui/assets/rudra-mark.png
tools/export_torchscript.py
```

`rudra/`, `ui/`, `training/` stay as they are.

## 6. Phases

| # | Phase | Work | Exit gate | Est. |
|---|---|---|---|---|
| 0 | Export + spike | TorchScript export; LibTorch loads it in a 50-line C++ app; GL window with RGBA16F scRGB on a Windows HDR monitor | CPU fp32 LibTorch vs Python on the 429 bench frames, max abs diff in log space ≤ 1e-5; 1,000-nit patch measured above SDR white | 1 w |
| 1 | librudra | Port decode, anchor, chroma, controls, colorspace, metadata, exr, aces, qc, queue; golden emitter in pytest | Every module passes its golden test on all three OSes | 3 w |
| 2 | GL viewer | Shaders ported; composite, display, false colour, difference, wipe, probe, guides; scopes; measure | Probe and measurements equal the browser Studio's on the same frame | 3 w |
| 3 | Qt UI | Main window, rails, tabs, transport, pipe bar, menus, shortcuts, QSS, first-run wizard, checkpoint manager | Full Studio workflow with no Python installed | 3 w |
| 4 | Deliver | libav decode, ffmpeg encode, five presets, QC, sidecar, queue window, resume | 3-clip queue killed mid-clip resumes; every output passes QC; C++ and Python resume each other's queue | 2.5 w |
| 5 | Package | Installers, signing, notarization, compute packs, updater, CI matrix | Signed builds install and update on all three | 1.5 w |
| 6 | Harden | 8K frames, long queues, VRAM exhaustion, missing GPU, low disk, driver matrix for Windows HDR, MPS parity if pursued | No open P0 | 2 w |

About 16 weeks for one engineer. Phase 0 is the go/no-go: if LibTorch parity
fails, the fix is in the export, before any porting starts.

## 7. Acceptance criteria for v1

- [ ] Installs and runs on Windows 11, Ubuntu 22.04, macOS 14+ with no Python present.
- [ ] Master EXR from the native app matches the Python master within 1 half-float ULP on CPU.
- [ ] Every ported module passes its golden test on every OS in CI.
- [ ] Drag a 4K EXR folder from Explorer/Finder: opens by path, first frame under 2 s warm.
- [ ] All five video presets pass the same QC as the CLI; sidecar written; never overwrites.
- [ ] Queue survives quit, crash and reboot; `queue.json` interchangeable with `rudra batch`.
- [ ] Pipe bar reports HDR out or SDR out truthfully; on-screen clip warning uses real headroom.
- [ ] Weights licence accepted before download; model sha and golden self-test before first use.

## 8. Decisions for Ahmed

1. **Model format:** TorchScript (recommended for v1, loads anywhere) or AOTInductor (faster, compiled per GPU arch and OS).
2. **macOS viewer:** GL 4.1 SDR-out in v1 (recommended), or a Metal backend now for EDR.
3. **Qt licence:** LGPL with dynamic linking (fine for a closed app if Qt is relinkable) or a commercial Qt licence.
4. **Browser Studio:** keep `ui/` as the reference implementation during the port (recommended), or retire it at v1.
5. **Distribution:** public releases, or FXTD-internal while the weights are non-commercial.

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| TorchScript export differs from eager | wrong reconstruction | Phase 0 parity gate on 429 frames before any porting |
| Two implementations drift (Python keeps changing) | app and paper disagree | golden files regenerated in CI on every commit; a failing parity test blocks merge |
| Windows HDR via GL float pixel format is driver-dependent | HDR out missing on some GPUs | per-driver verification; SDR-out fallback with the warning |
| macOS GL deprecated | future macOS drops it | single GLSL source, Metal backend as a planned follow-up |
| LibTorch CUDA size (~2.4 GB) | slow first run | CPU works immediately; resumable pack; offline installer |
| libav / ffmpeg version drift (the 7.x `bt2020` break) | bad encodes | per-binary self-test with full QC before first delivery |
| Port effort (~7k lines) underestimated | late v1 | librudra first, CLI diff against Python gives early signal |
