#!/usr/bin/env bash
# Gate B on macOS (Metal, EDR) or Linux (Vulkan): does an HDR swapchain carry
# a 1 000-nit patch above SDR white? The Windows twin is NATIVE_GATE_B.ps1.
#
#   scripts/native_gate_b.sh               # build, run, report
#   FRAMES=0 scripts/native_gate_b.sh      # keep the window open and look (Esc quits)
#
# Qt 6.8 (with Qt Shader Tools) is fetched into tmp/native_deps/Qt with
# aqtinstall; nothing is installed system-wide. Needs CMake 3.24+, Ninja or
# Make, a C++20 compiler (Xcode CLT on macOS) and python3.
#
# PASS means the swapchain carried the 1 000-nit patch at least a stop above
# SDR white. On a Mac, run it on the XDR panel (or an HDR display with "High
# Dynamic Range" on) and check the patch by eye or meter too.
set -euo pipefail
cd "$(dirname "$0")/.."
QT_VERSION=${QT_VERSION:-6.8.3}
FRAMES=${FRAMES:-240}
PY=${PYTHON:-python3}
DEPS=tmp/native_deps
STAMP=$(date +%Y-%m-%d_%H%M)
mkdir -p "$DEPS" reports

case "$(uname -s)" in
  Darwin) HOST=mac; ARCH=clang_64; QT_DIR=macos; RUNS="metal:p3 metal:scrgb"; PARITY_APIS="metal gl" ;;
  Linux)  HOST=linux; ARCH=linux_gcc_64; QT_DIR=gcc_64; RUNS="vulkan:scrgb vulkan:hdr10 gl:scrgb"; PARITY_APIS="vulkan gl" ;;
  *) echo "use scripts/NATIVE_GATE_B.ps1 on Windows" >&2; exit 2 ;;
esac
QT_ROOT="$DEPS/Qt/$QT_VERSION/$QT_DIR"

if [ ! -x "$QT_ROOT/bin/qsb" ]; then
  echo "== Qt $QT_VERSION ($ARCH, qtshadertools) into $DEPS/Qt"
  "$PY" -m pip install --quiet --upgrade aqtinstall
  "$PY" -m aqt install-qt "$HOST" desktop "$QT_VERSION" "$ARCH" -m qtshadertools -O "$DEPS/Qt"
fi

echo "== build rudra-hdr-probe"
GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
cmake -S native -B build/native_gate_b "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
  -DRUDRA_BUILD_TESTS=OFF -DRUDRA_BUILD_CLI=OFF -DRUDRA_BUILD_APP=OFF -DRUDRA_BUILD_HDR_PROBE=ON \
  -DCMAKE_PREFIX_PATH="$PWD/$QT_ROOT" >/dev/null
cmake --build build/native_gate_b --target rudra-hdr-probe rudra-gpu-parity --parallel
EXE=build/native_gate_b/render/probe/rudra-hdr-probe
PARITY=build/native_gate_b/render/probe/rudra-gpu-parity

status=1
printf "\n%-8s %-6s %-6s %8s %8s %8s  %s\n" API ASKED GOT 203 1000 2000 VERDICT
for run in $RUNS; do
  api=${run%%:*}; fmt=${run##*:}
  json="reports/native_gate_b_${api}_${fmt}_${STAMP}.json"
  "$EXE" --api "$api" --format "$fmt" --frames "$FRAMES" --report "$json" >/dev/null 2>&1 || true
  if [ -f "$json" ]; then
    "$PY" - "$json" "$api" "$fmt" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = {int(x["target_nits"]): x["swapchain_nits"] for x in d["patches"]}
print(f"{sys.argv[2]:<8} {sys.argv[3]:<6} {d['output_path']:<6} {p[203]:>8} {p[1000]:>8} {p[2000]:>8}  {d['verdict']}")
PY
    grep -q '"verdict": "PASS"' "$json" && status=0
  else
    printf "%-8s %-6s %-6s %8s %8s %8s  %s\n" "$api" "$fmt" "" "" "" "" ERROR
  fi
done
echo; echo "Reports in reports/. PASS is the swapchain half of the gate; confirm on the glass."

echo; echo "== GPU composite parity (day 8)"
for api in $PARITY_APIS; do
  code=0
  outp=$("$PARITY" --api "$api" --report "reports/native_gpu_parity_${api}_${STAMP}.json" 2>&1) || code=$?
  printf '%s\n' "$outp" | grep -E "^GPU composite|=>" || true
  case $code in
    0) ;;
    2) echo "  $api: not available here" ;;
    *) echo "  $api: FAIL"; status=1 ;;
  esac
done
exit $status
