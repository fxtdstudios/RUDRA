# RUDRA Studio 0.3.3rc1 — Windows preview

This is the Python/browser Studio workflow preview. It is separate from the
native RUDRA 0.9.0 beta; it does not replace or update that application's binaries.
The reconstruction model remains shipped Shadow v1, step 2400. No experimental
training candidate is included. Code is Apache-2.0; bundled weights and derivatives
are non-commercial. Read LICENSE-code and checkpoints/LICENSE before use.

## Install and start

1. Install Python 3.13 x64 from python.org with the Windows Python launcher.
2. Extract the ZIP to a writable local folder.
3. Double-click **Install Studio.cmd**. Internet access is needed; CUDA PyTorch
   downloads approximately 2 GB, plus runtime dependencies. Allow several GB of disk.
4. Double-click **Start Studio.cmd**. A terminal stays open and the browser opens
   http://localhost:8431. Stop an older Studio using that port first, or launch
   `"Start Studio.cmd" --port 8432` from a terminal.
5. Close the terminal or press Ctrl+C to stop Studio. To remove this preview,
   delete its extracted folder after stopping it; exported renders stay where saved.

This preview targets NVIDIA CUDA on Windows x64. It uses torch 2.13.0+cu130;
an NVIDIA driver compatible with that runtime is required. CPU fallback is
available but slower; use `"Start Studio.cmd" --device cpu` explicitly if needed.
The package contains no Python interpreter, NVIDIA driver or offline dependency
bundle. It is a preview ZIP with a setup script, not a native EXE installer.

## Use

Open an SDR image or ordered sequence. Auto color setup converts supported
8-bit embedded ICC input to sRGB; untagged images are explicitly assumed sRGB.
For tagged 16-bit PNG/TIFF, turn Auto off and select the correct OCIO input space:
automatic high-precision ICC conversion is not implemented. Unsigned 16-bit RGB
and grayscale inputs preserve precision; float/signed TIFF is unsupported.

Choose export format/color space, use **Browse folders** to select a destination,
and choose current image or all loaded frames. Set a render name/start frame and
click **Render export**. Existing outputs are protected from overwrite.

Exports: ACES2065-1 half-float EXR, linear Rec.2020 half-float EXR, sRGB 8-bit
PNG/TIFF with ICC, OCIO float32 EXR and OCIO display/view PNG. The default OCIO
config is bundled by OpenColorIO; custom configs need explicit bridge mappings.
PNG/TIFF export preview uses the export transform; the main HDR viewer remains
separate. Arbitrary display-view PNG is not certified native HDR delivery.

## Verification and limits

450 source tests passed. The wheel was installed in a fresh isolated Python 3.13
environment on the existing Windows host, with no shared environment packages.
GPU inference and 12 sequence outputs across all six formats were verified on an
RTX 4080 SUPER, including 16-bit transport, EXR chromaticities, sRGB ICC and
overwrite protection. Dependency consistency passed. This is not a clean OS,
all-GPU, macOS/Linux, calibrated HDR display or real-video temporal certification.
Sequence export support does not establish flicker-free reconstruction.

Model upgrades failed acceptance checks and are excluded. Clipped detail cannot
be faithfully recovered when it is absent from the input. This preview is for
evaluation/non-commercial use with the supplied weights.
