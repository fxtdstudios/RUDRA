#!/usr/bin/env bash
# Gate A on macOS (Apple silicon) or Linux: does the C++ build compute what
# Python computes, on every inference backend this machine has? The Windows
# twin is NATIVE_GATE_A.ps1.
#
#   scripts/native_gate_a.sh                          # export, build, diff, bench
#   SKIP_EXPORT=1 scripts/native_gate_a.sh            # use dist/models/<stem> as it is
#   BENCH_DIR=~/bench/sdr scripts/native_gate_a.sh    # add the bench frames to the parity check
#   NO_BENCH=1 scripts/native_gate_a.sh               # parity only, no timings
#
# Rows: LibTorch CPU and ONNX Runtime CPU everywhere; LibTorch MPS and ONNX
# Runtime Core ML on a Mac; LibTorch CUDA on Linux when this torch sees a GPU.
# Each is compared with the package's golden frames (eager PyTorch outputs) at
# the package's own tolerances.
#
# LibTorch is this Python's pip torch (headers and libraries, no TorchConfig).
# ONNX Runtime is the official release archive, fetched into tmp/native_deps
# (Core ML is built into the macOS one). If the Python given has no torch, a
# venv with torch, onnx, onnxruntime and numpy is made in tmp/native_deps.
# Nothing is installed system-wide. Needs CMake 3.24+, Ninja or Make, a C++20
# compiler (Xcode CLT on macOS), python3 and curl. OpenCV is not needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
ORT_VERSION=${ORT_VERSION:-1.22.0}
CHECKPOINT=${CHECKPOINT:-checkpoints/sdr2hdr_shadow_v1.pt}
BENCH_DIR=${BENCH_DIR:-}
BENCH_LIMIT=${BENCH_LIMIT:-0}
DEPS=tmp/native_deps
BUILD=build/native_gate_a
STEM=$(basename "$CHECKPOINT" .pt)
PACKAGE=dist/models/$STEM
STAMP=$(date +%Y-%m-%d_%H%M)
REPORT=reports/native_gate_a_${STAMP}.txt
mkdir -p "$DEPS" reports

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) OS=mac; ORT_ARCHIVE=onnxruntime-osx-arm64-$ORT_VERSION ;;
  Darwin-*) fail "an Intel Mac has no MPS or Core ML worth gating; use Apple silicon" ;;
  Linux-x86_64) OS=linux; ORT_ARCHIVE=onnxruntime-linux-x64-$ORT_VERSION ;;
  Linux-aarch64) OS=linux; ORT_ARCHIVE=onnxruntime-linux-aarch64-$ORT_VERSION ;;
  *) fail "use scripts/NATIVE_GATE_A.ps1 on Windows" ;;
esac

# ---------------------------------------------------------------------------
say "Python and torch"
if ! "$PY" -c "import torch, onnx, onnxruntime, numpy" 2>/dev/null; then
  VENV=$DEPS/venv_gate_a
  if [ ! -x "$VENV/bin/python" ]; then
    echo "no torch/onnx/onnxruntime in $PY: making $VENV"
    "$PY" -m venv "$VENV"
  fi
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet torch onnx onnxruntime numpy
  PY=$VENV/bin/python
fi
{ read -r TORCH_VERSION; read -r TORCH_ROOT; read -r CUDA_AVAIL; read -r MPS_AVAIL; read -r ABI; } < <("$PY" - <<'PY'
import os, torch
print(torch.__version__)
print(os.path.dirname(torch.__file__))
print(int(torch.cuda.is_available()))
print(int(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()))
print(int(torch.compiled_with_cxx11_abi()))
PY
)
TORCH_LIB=$TORCH_ROOT/lib
echo "torch $TORCH_VERSION  cuda $CUDA_AVAIL  mps $MPS_AVAIL"
echo "python $("$PY" -c 'import sys; print(sys.executable)')"
WITH_LIBTORCH=1
[ -f "$TORCH_ROOT/include/torch/script.h" ] || { echo "torch headers not under $TORCH_ROOT: LibTorch rows skipped"; WITH_LIBTORCH=0; }

# ---------------------------------------------------------------------------
if [ -z "${SKIP_EXPORT:-}" ]; then
  say "Export the model package"
  args=(tools/export_model.py "$CHECKPOINT" --out dist/models)
  if [ -n "$BENCH_DIR" ]; then
    args+=(--bench-dir "$BENCH_DIR")
    [ "$BENCH_LIMIT" != 0 ] && args+=(--bench-limit "$BENCH_LIMIT")
  fi
  "$PY" "${args[@]}" || fail "export_model.py (Python-side parity failed or export error)"
fi
[ -f "$PACKAGE/manifest.json" ] || fail "no package at $PACKAGE (export it, or copy dist/models/$STEM from the PC)"

# ---------------------------------------------------------------------------
say "ONNX Runtime $ORT_VERSION ($ORT_ARCHIVE)"
ORT_ROOT=$DEPS/$ORT_ARCHIVE
if [ ! -f "$ORT_ROOT/include/onnxruntime_cxx_api.h" ]; then
  url=https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_ARCHIVE.tgz
  echo "download $url"
  curl -fL --retry 5 --retry-all-errors -o "$DEPS/$ORT_ARCHIVE.tgz" "$url"
  tar -xzf "$DEPS/$ORT_ARCHIVE.tgz" -C "$DEPS"
  rm -f "$DEPS/$ORT_ARCHIVE.tgz"
fi
ORT_ROOT=$(cd "$ORT_ROOT" && pwd)
[ "$OS" = mac ] && { [ -f "$ORT_ROOT/include/coreml_provider_factory.h" ] || echo "no coreml_provider_factory.h: the Core ML row will report it"; }

# ---------------------------------------------------------------------------
if [ -z "${SKIP_BUILD:-}" ]; then
  say "Configure and build native/ (Release)"
  GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
  cfg=(-S native -B "$BUILD" ${GEN[@]+"${GEN[@]}"} -DCMAKE_BUILD_TYPE=Release
       -DRUDRA_BUILD_TESTS=OFF -DRUDRA_BUILD_APP=OFF
       -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=$ORT_ROOT"
       "-DCMAKE_BUILD_RPATH=$ORT_ROOT/lib;$TORCH_LIB")
  if [ "$WITH_LIBTORCH" = 1 ]; then
    cfg+=(-DRUDRA_WITH_LIBTORCH=ON "-DRUDRA_TORCH_ROOT=$TORCH_ROOT" "-DRUDRA_TORCH_CXX11_ABI=$ABI")
  else
    cfg+=(-DRUDRA_WITH_LIBTORCH=OFF)
  fi
  cmake "${cfg[@]}" >/dev/null || fail "cmake configure"
  cmake --build "$BUILD" --target rudra-native --parallel || fail "build"
fi
EXE=$BUILD/cli/rudra-native
[ -x "$EXE" ] || fail "rudra-native not built"
if [ "$OS" = mac ]; then export DYLD_LIBRARY_PATH="$ORT_ROOT/lib:$TORCH_LIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
else export LD_LIBRARY_PATH="$ORT_ROOT/lib:$TORCH_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi

# ---------------------------------------------------------------------------
say "Gate A"
# name|runtime|device|run
ROWS=("LibTorch CPU|libtorch|cpu|$WITH_LIBTORCH")
if [ "$OS" = mac ]; then
  ROWS+=("LibTorch MPS|libtorch|mps|$((WITH_LIBTORCH && MPS_AVAIL))")
else
  ROWS+=("LibTorch CUDA|libtorch|cuda|$((WITH_LIBTORCH && CUDA_AVAIL))")
fi
ROWS+=("ONNX Runtime CPU|onnxruntime|cpu|1")
[ "$OS" = mac ] && ROWS+=("ONNX Runtime Core ML|onnxruntime|coreml|1")

{
  echo "RUDRA native Gate A, $STAMP, $(uname -sm)"
  echo "package $PACKAGE"
  echo "torch $TORCH_VERSION, ONNX Runtime $ORT_VERSION"
  echo
  "$EXE" info "$PACKAGE" 2>&1 || true
  echo
} > "$REPORT"

status=0
SUMMARY=()
for row in "${ROWS[@]}"; do
  IFS='|' read -r name runtime device run <<<"$row"
  if [ "$run" != 1 ]; then SUMMARY+=("$name|skipped|"); continue; fi
  code=0
  out=$("$EXE" diff "$PACKAGE" --runtime "$runtime" --device "$device" 2>&1) || code=$?
  printf -- '---- %s\n%s\n\n' "$name" "$out" >> "$REPORT"
  worst=$(printf '%s\n' "$out" | grep -oE 'max \|d\| [0-9.eE+-]+' | awk '{print $3}' | sort -g | tail -1)
  case $code in
    0) result=PASS ;;
    1) result=FAIL; status=1 ;;
    *) result=ERROR; status=1; worst=$(printf '%s\n' "$out" | tail -2 | tr '\n' ' ') ;;
  esac
  SUMMARY+=("$name|$result|$worst")
done
table() {
  printf '%-22s %-8s %s\n' Backend Result "Worst |d|"
  for s in "${SUMMARY[@]}"; do IFS='|' read -r n r w <<<"$s"; printf '%-22s %-8s %s\n' "$n" "$r" "$w"; done
}
table | tee -a "$REPORT"

# ---------------------------------------------------------------------------
# Inference time at 1080p on every backend that passed, for the budget table
# (NATIVE_ARCHITECTURE.md 6.6). Wall time to fields in host memory.
if [ -z "${NO_BENCH:-}" ]; then
  say "Inference time, 1920x1080, fp32 (median of 5)"
  printf '%-22s %12s %16s\n' Backend "untiled ms" "tiled 512/64 ms" | tee -a "$REPORT"
  i=0
  for row in "${ROWS[@]}"; do
    IFS='|' read -r name runtime device run <<<"$row"
    IFS='|' read -r _ result _ <<<"${SUMMARY[$i]}"; i=$((i + 1))
    [ "$result" = PASS ] || continue
    out=$("$EXE" bench "$PACKAGE" --runtime "$runtime" --device "$device" --size 1920x1080 --iters 5 2>&1) || true
    printf -- '---- bench %s\n%s\n\n' "$name" "$out" >> "$REPORT"
    untiled=$(printf '%s\n' "$out" | awk '$1=="BENCH" && $5=="untiled" {print $6}')
    tiled=$(printf '%s\n' "$out" | awk '$1=="BENCH" && $5=="tiled" {print $6}')
    printf '%-22s %12s %16s\n' "$name" "${untiled:--}" "${tiled:--}" | tee -a "$REPORT"
  done

  say "Viewer measurements and scopes on the CPU"
  out=$("$EXE" bench-scopes 2>&1) || true
  printf '%s\n' "$out" | grep -v '^BENCH ' || true
  printf -- '---- bench-scopes\n%s\n\n' "$out" >> "$REPORT"
fi

say "Result"
table
echo "Full log: $REPORT"
exit $status
