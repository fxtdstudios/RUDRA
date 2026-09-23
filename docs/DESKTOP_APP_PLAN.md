# RUDRA Studio Desktop: cross-platform plan

> 23 Sep 2026. Proposal, not started. Mockup: the "RUDRA Studio Desktop" design
> canvas (five boards: main window, first run, deliver dialog, render queue,
> architecture). Nothing in `ui/`, `rudra/` or `training/` changes behaviour
> until Phase 1 lands behind the existing tests.

## 1. What we are building

RUDRA Studio today is `python ui/server.py` plus a browser tab: 5.1k lines of
HTML/JS/CSS (`ui/app.js`, `ui/compositor.js`, `ui/shell.js`) in front of a
Python HTTP server that loads `SDR2HDRNet`, reads shots by path, and masters EXR.
Video delivery (`rudra video`, `rudra batch`) exists only on the command line;
the Deliver panel says so.

The desktop app is the same instrument, installed and double-clickable, on
Windows x64, macOS arm64 and Linux x64, with four things a browser tab cannot do:

1. **Real paths.** Native open/save dialogs and OS drag and drop hand the backend
   a path, so a 4K EXR sequence is read where it sits. Nothing is uploaded.
2. **Real HDR on the glass.** Where the OS and display allow it, the viewer
   emits values above SDR white instead of simulating PQ on an SDR canvas.
3. **Delivery wired in.** HDR10, HLG, ProRes 422/422 HQ/4444 from the Deliver
   panel, with the queue, resume, QC and sidecar that `rudra.batch` and
   `rudra.video` already implement.
4. **A managed runtime.** GPU detection, the right torch build, checkpoints
   verified against `checkpoints/SHA256SUMS`, FFmpeg capability probed, the
   non-commercial weights licence accepted once.

Out of scope: a mobile build, a web-hosted version, training from the app,
any change to the model or its numbers.

## 2. Decision: Electron + Python sidecar

| | Electron | Tauri 2 | PySide6 / Qt |
|---|---|---|---|
| Reuses `ui/` as is | yes | yes | no, rewrite 5.1k lines |
| One render engine on all three OSes | yes, Chromium everywhere | no: WebView2 / WKWebView / WebKitGTK | yes |
| HDR canvas output | Chromium WebGPU `rgba16float` + extended tone mapping, to verify in Phase 0 | differs per OS webview; WebKitGTK has none | possible via QRhi, but new code |
| Float textures / WebGL2 parity for `compositor.js` | identical to today's Chrome | varies | n/a |
| Installer size | ~110 MB shell | ~15 MB shell | ~80 MB |
| Size that matters | torch CUDA ~2.5 GB, dwarfs any shell | same | same |

**Electron.** The Studio is a measurement instrument: the same pixel must
render the same way on every OS, and the pipe bar's "clipped on screen, not in
the master" warning must be computed by one engine. Tauri's smaller shell is
irrelevant next to torch, and its per-OS webviews are exactly the variance an
instrument cannot have. Qt would mean rewriting a working UI.

The Python backend stays Python: `ui/server.py` becomes a sidecar process the
shell starts, watches and stops.

## 3. Architecture

```
┌──────────────── Electron main (Node) ────────────────┐
│ window, native menus, dialogs, drag-drop paths,       │
│ single instance, taskbar/dock progress, notifications │
│ power-save blocker during renders, auto-update        │
│ sidecar supervisor: spawn, handshake, health, restart │
└───────┬─────────────────────────────┬─────────────────┘
        │ preload (contextBridge)     │ spawn + stdout handshake
┌───────▼─────────────┐       ┌───────▼──────────────────────────┐
│ Renderer = ui/      │ HTTP  │ Sidecar = ui/server.py           │
│ app.js, compositor  │◄─────►│ 127.0.0.1:<ephemeral>, token     │
│ WebGL2 → WebGPU HDR │  SSE  │ /api/frame /api/sequence/*       │
│ window.rudraDesktop │       │ /api/master  + NEW /api/jobs,    │
└─────────────────────┘       │ /api/deliver, /api/health,       │
                              │ /api/events (SSE)                │
                              │ rudra.video · rudra.batch · torch│
                              └──────────────┬───────────────────┘
                                             │ subprocess
                                        ffmpeg / ffprobe
```

### 3.1 Sidecar contract (changes to `ui/server.py`)

- `--port 0 --host 127.0.0.1 --token-file <path>`: bind an ephemeral loopback
  port, print one JSON line `{"rudra":"ready","port":N,"pid":P,"device":"cuda"}`
  to stdout, then serve. The shell reads that line; no fixed 8422, no clash
  with a second copy.
- Every `/api/*` request carries `X-Rudra-Token`; anything else is 403. Keeps
  other local processes and web pages off the model.
- `GET /api/health` (loaded checkpoint, sha256, device, VRAM, ffmpeg caps).
- `POST /api/deliver`: one `rudra video` job, same option names as the CLI.
- `GET/POST /api/jobs`: read and edit a `queue.json`, run it through
  `rudra.batch.run_queue` in a worker thread, `retry_failed` exposed.
- `GET /api/events`: Server-Sent Events for job progress and log lines, fed by
  the `progress()` callback `run_queue` already has.
- `POST /api/shutdown`: finish the current frame write, then exit. The shell
  calls it on quit; SIGTERM / `TerminateProcess` after 10 s.
- `webbrowser.open` is skipped when `--no-browser` (desktop always passes it).
- Browser mode (`python ui/server.py`) keeps working unchanged: same page, same
  endpoints, token optional when bound to loopback without `--token-file`.

### 3.2 Preload API (`window.rudraDesktop`)

Narrow, typed, nothing else reaches Node:

```
pickFrames() -> string[]          pickFolder(kind) -> string
pickSaveTarget(preset) -> string  pathForDroppedFile(File) -> string
reveal(path)                      openExternal(url)   (allow-listed)
setProgress(0..1 | null)          notify(title, body)
displayInfo() -> {hdr, headroom, colorSpace}
onMenu(cb)  (native menu → the same data-act ids index.html uses)
```

`app.js` feature-detects `window.rudraDesktop`; when absent it behaves exactly
as today. The render-folder text field becomes a field plus a Browse button.

### 3.3 Menus

`index.html`'s `data-act` table is the single source. On macOS the HTML
menubar is hidden and a native `Menu` is built from the same ids (App menu,
File, Edit, Clip, Reconstruct, Measure, Deliver, Window, Help). On Windows and
Linux the existing in-page menubar stays and becomes the custom titlebar
(`titleBarOverlay`, window controls on the right). Shortcuts are unchanged;
Cmd replaces Ctrl on macOS.

### 3.4 HDR viewer path

- **Detect:** `matchMedia('(dynamic-range: high)')` plus Electron `screen`
  display info; report headroom in the pipe bar ("view PQ · Rec.2020 · HDR out
  · 1000 nits" vs "SDR out · PQ simulated").
- **Output:** a WebGPU canvas configured `rgba16float` with extended tone
  mapping and a Rec.2020-linear or display-P3 colour space, fed by the same
  composite `compositor.js` computes today. WebGL2 SDR stays the fallback and
  is the reference for the parity test.
- **Honesty rule:** if the display cannot show the frame's MaxCLL the existing
  warning fires, with the display's measured headroom in place of the slider
  value. Never tone-map silently.
- Windows needs "Use HDR" on; macOS uses EDR headroom (varies with
  brightness, so read it live); Linux ships SDR-out in v1 (Wayland HDR in
  Chromium is not dependable yet) and says so in the pipe bar.

## 4. Runtime and dependencies

### 4.1 Python and torch

- Ship **python-build-standalone 3.12** inside the app (~40 MB) plus **uv**.
- Ship a lockfile (`desktop/runtime/uv.lock`) generated from `pyproject.toml`.
- First run detects the GPU and installs one torch variant into the app's data
  dir, not into Program Files:
  - NVIDIA on Windows/Linux: CUDA 12.x wheel (~2.5 GB download, once).
  - macOS arm64: default wheel, `--device mps`.
  - Anything else: CPU wheel (~200 MB).
- An **offline installer** variant (Windows CUDA, ~3 GB) for studio machines
  without internet.
- Runtime is re-synced by `uv sync` only when the lockfile hash changes, so an
  app update that does not touch Python does not re-download torch.
- **MPS is new.** `ui/server.py` and `rudra.video` accept `cuda|cpu` today.
  Adding `mps` needs a parity test (MPS vs CPU on the 429 bench frames, max
  abs diff in log space) before it is offered; until then macOS is CPU.
- macOS Intel is not supported: torch stopped publishing x86_64 macOS wheels.

### 4.2 Checkpoints

- Checkpoint manager lists `checkpoints/models.json` entries and local files,
  verifies each against `SHA256SUMS` (or the Hub file's sha) before loading,
  shows the hash in the status bar (it already shows the path).
- Download from the Hugging Face repo with resume; stored in
  `<appData>/RUDRA/checkpoints`.
- The **non-commercial weights licence** (`checkpoints/LICENSE`) is shown on
  first run and must be accepted before any weights download. Code licence
  (Apache 2.0) and `NOTICE` in About.
- Quarantined runs (`checkpoints/_invalid_*`) are never listed.

### 4.3 FFmpeg

- **Probe, do not assume.** On start: `ffmpeg -version`, `-encoders`,
  `-filters`; required: `libx265`, `prores_ks`, `zscale`. Minimum version
  pinned. The 23 Sep `bt2020` vs `bt2020nc` break on 7.x is the reason: the
  probe runs the existing encode self-test (one 16-frame HDR10 clip, full QC)
  and caches the verdict per ffmpeg binary hash.
- **Not bundled in v1.** An x265-enabled build is GPL; shipping it inside an
  Apache-2.0 app obliges source distribution for it. v1 finds a system ffmpeg
  or offers a one-click download of a named GPL build into app data, with its
  licence shown. Stills and EXR mastering never need ffmpeg.

## 5. Security

- `contextIsolation: true`, `sandbox: true`, `nodeIntegration: false`, strict
  CSP (`default-src 'self' http://127.0.0.1:<port>`), no remote content.
- Loopback bind plus per-launch token (§3.1).
- `openExternal` allow-list (fxtdstudios.com, the GitHub repo, the Hub page).
- Code signing on every platform; unsigned Python in AppData is the most
  common antivirus false positive, so the embedded interpreter is signed too.

## 6. Packaging, updates, CI

| | Windows x64 | macOS arm64 | Linux x64 |
|---|---|---|---|
| Format | NSIS installer + offline variant | DMG | AppImage + .deb |
| Signing | Authenticode (Azure Trusted Signing or EV) | Developer ID + notarization | detached gpg |
| Updates | electron-updater, GitHub Releases | same | AppImage zsync; .deb via apt repo later |
| GPU runtime | CUDA 12.x | MPS (after parity) / CPU | CUDA 12.x |

- `electron-builder` config in `desktop/`.
- GitHub Actions matrix: `windows-latest`, `macos-14`, `ubuntu-22.04`.
  Existing `pytest` runs first; then a Playwright-for-Electron smoke test
  launches the packaged app on CPU, opens `ui/assets/cinematic_hdr_sunset.png`,
  masters EXR, and compares the output against the browser-mode master of the
  same frame (tolerance: bit-exact on CPU).
- Release channel `stable`, plus `beta` for Ahmed and FXTD staff.

## 7. Repository layout

```
desktop/
  package.json            electron, electron-builder, electron-updater
  src/main.ts             window, menus, lifecycle
  src/sidecar.ts          spawn, handshake, health, restart, shutdown
  src/preload.ts          window.rudraDesktop
  src/runtime.ts          python-standalone + uv, GPU detect, torch install
  src/ffmpeg.ts           probe + self-test
  src/checkpoints.ts      list, verify, download
  runtime/uv.lock
  build/                  icons (from ui/assets/rudra-mark.png), entitlements
  test/smoke.spec.ts
ui/                       renderer, unchanged except feature-detected hooks
rudra/                    unchanged except `mps` device + /api additions' helpers
```

## 8. Phases

| # | Phase | Work | Exit gate | Est. |
|---|---|---|---|---|
| 0 | Spike | Electron loads `ui/` against a hand-started sidecar. HDR output test page on Win HDR monitor + MacBook XDR: a 1000-nit patch must measure above SDR white | HDR patch verified on both, or HDR-out dropped from v1 | 3 d |
| 1 | Shell | Sidecar supervisor + handshake + token; native dialogs; drag-drop paths; menus; single instance; logs to `<appData>/RUDRA/logs` | Browser mode and desktop mode pass the same UI smoke test | 1 w |
| 2 | Runtime | First-run flow: licence, GPU detect, torch install, checkpoint verify/download, ffmpeg probe + self-test | Clean VM on each OS to first reconstruction with no terminal | 1 w |
| 3 | Deliver | Deliver dialog → `/api/deliver`; queue window on `rudra.batch`; SSE progress; resume; QC report and sidecar viewer; notifications, taskbar progress | A 3-clip queue killed mid-clip resumes and every output passes `rudra.video` QC | 1.5 w |
| 4 | HDR view | WebGPU extended output; live headroom; pipe bar reports HDR-out/SDR-out; parity test vs WebGL2 SDR path | Probe values identical in both paths; warning fires correctly on SDR display | 1 w |
| 5 | Package | Signing, notarization, updater, offline installer, CI matrix, smoke test on packaged builds | Signed builds install and update on all three | 1 w |
| 6 | Harden | MPS parity (if pursued), crash/restart of sidecar mid-render, low-disk, missing GPU, 8K frames, long queues | No open P0 | 1 w |

About 7.5 weeks for one engineer. Phase 0 is the only real unknown; everything
else is plumbing around code that already works.

## 9. Acceptance criteria for v1

- [ ] Installs and opens on Windows 11, macOS 14+, Ubuntu 22.04 without a terminal.
- [ ] Master EXR from desktop is bit-identical to browser mode on CPU for the same frame and settings.
- [ ] Drag a 4K EXR folder from Explorer/Finder: opens by path, no upload, first frame under 2 s after warm start.
- [ ] HDR10, HLG, ProRes 422 HQ, ProRes 4444 exported from the Deliver dialog pass the same QC the CLI applies; `.json` sidecar written; never overwrites.
- [ ] Queue survives app quit, sidecar crash and reboot; resumes at the interrupted clip.
- [ ] Viewer reports HDR-out or SDR-out truthfully; the clipped-on-screen warning uses the real display headroom.
- [ ] Weights licence accepted before any weights are downloaded; checkpoint sha shown before it is used.
- [ ] Cold start after first run under 5 s to an interactive window (model load may continue behind it, stated in the status bar).

## 10. Decisions for Ahmed

1. Electron (recommended) or Tauri 2.
2. macOS: CPU-only in v1, or spend Phase 6 on MPS parity.
3. FFmpeg: find/download (recommended) or bundle a GPL build with a source offer.
4. Torch: online first-run install (recommended) plus a Windows offline installer, or offline everywhere.
5. Distribution: public GitHub Releases, or FXTD-internal only while weights are non-commercial.

## 11. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Chromium HDR canvas behaves differently on Win vs mac | HDR-out wrong on one OS | Phase 0 gate; measured patch test; SDR fallback is today's behaviour |
| 2.5 GB torch download on first run | bad first impression, fails on studio networks | progress + resume; offline installer; CPU path usable meanwhile |
| MPS numerics differ from CUDA/CPU | macOS masters not comparable | CPU default on mac until parity test passes |
| FFmpeg version drift (the 7.x `bt2020` break) | silent bad encodes | per-binary self-test with full QC before first delivery |
| Antivirus flags embedded Python | install blocked | sign interpreter and app; submit to Defender |
| Sidecar crash mid-render | lost work | `rudra.batch` already writes durable per-job state; shell restarts sidecar and resumes |
