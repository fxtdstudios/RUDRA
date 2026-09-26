# RUDRA 0.9.0 beta 2

The desktop RUDRA: SDR footage in, scene-linear HDR out, with the places the
model reconstructed shown to you. This is the second public beta of the native
app (C++20, Qt 6.8, no Python at runtime). It is for evaluation and
non-commercial use (PolyForm Noncommercial 1.0.0, see LICENSE; the model
weights: LICENSE-weights).

## What is in it

- The Studio in a native window: Reconstruct, Grade and Deliver; residual
  strength, recovery modes, preserve, Region EV, undo; the scopes (waveform,
  histogram, vectorscope), the probe, false colour, wipe, guides.
- An HDR viewer: on an HDR display the image is shown above SDR white
  (Windows: Direct3D 12 scRGB and HDR10; macOS: Metal EDR on the XDR panel).
- Stills and sequences: PNG, JPEG, TIFF, WebP, BMP, a frame or a folder.
- Movies: open, scrub, and export HDR10 (HEVC 10-bit with its metadata), HLG
  or ProRes 422 HQ, with the audio, checked before it is published, through a
  queue you can stop and resume (ProRes 4444 with alpha from the command
  line: `rudra-native video --format prores4444`).
- Masters as ACES 2065-1 EXR sequences.
- The model: `sdr2hdr_shadow_v1`, the shipped RUDRA model, run by ONNX
  Runtime (DirectML on Windows GPUs, Core ML on Apple silicon, CPU
  everywhere). Every package is checked against its reference frames the
  first time it runs on a device, and refused if it drifts.
- `rudra-native`, the command line: `diff`, `video`, `deliver`, `batch`,
  `ffmpeg-check`.

## Changes since beta 1

- Includes the native highlight-grain correction from the current main branch.
- Packages identify themselves as 0.9.0-beta.2. The shipped model remains
  `sdr2hdr_shadow_v1`; this is an application beta, not newly trained weights.
- Packaging runs the native app tests on both platforms. macOS movie checks
  use FFmpeg 7; FFmpeg 9 is not qualified because metadata checks regressed.
- This native release is separate from the Python/browser Studio preview;
  it does not claim feature parity with that preview.

## Install

**macOS (Apple silicon, macOS 13 or later).** Open the DMG and drag RUDRA to
Applications. The beta is not notarised yet: the first time, right-click
RUDRA > Open (subject to your macOS security policy).

**Windows (x64, Windows 10 or 11).** Run `RUDRA-0.9.0-beta.2-windows-x64-setup.exe`:
it installs for your user by default (no administrator prompt), adds RUDRA to
the Start menu and can be removed from Settings > Apps. Or take the portable
ZIP: unzip anywhere and run `RUDRA.exe`. A GPU with DirectX 12 is used when
there is one; the CPU otherwise.

**Movies, both systems.** RUDRA runs `ffmpeg` and `ffprobe` from the PATH and
needs a build with `libx265`, `prores_ks` and `zscale`:
macOS `brew install ffmpeg@7`, with `$(brew --prefix ffmpeg@7)/bin` on PATH; Windows the "full" build from gyan.dev with
its `bin` folder on the PATH. `rudra-native ffmpeg-check` says whether yours
has everything. Stills need nothing extra.

## Known in this beta

- macOS: HDR output on Metal EDR, and inference on Core ML, are being
  measured on hardware; until then, check the first-run card (Help > Check the
  display and the model), which reports the display's real peak.
- The model does not yet beat the analytic baseline on every condition of
  the bench (see STATUS.md in the repository); the reconstruction is shown
  where it applied, so you can judge it frame by frame.
- Not signed with a Developer ID (macOS) or a code-signing certificate
  (Windows): SmartScreen may ask once.
- Linux builds are made from source (`scripts/native_app.sh`); there is no
  Linux package in this beta.

## Reporting

Issues and results: https://github.com/fxtdstudios/RUDRA/issues. Include the
version (RUDRA.app > Get Info, or RUDRA.exe > Properties > Details), what
Help > About shows, and for a crash the report macOS or Windows offers.
