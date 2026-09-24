#!/usr/bin/env bash
# The RUDRA app on macOS (Apple silicon) or Linux: build it, run its Qt tests,
# open it. The Windows counterpart is NATIVE_PHASE3_EXIT.ps1.
#
#   scripts/native_app.sh                  # build, test, open the app
#   NO_LAUNCH=1 scripts/native_app.sh      # build and test only
#   SKIP_TESTS=1 scripts/native_app.sh     # build and open
#   SKIP_BUILD=1 scripts/native_app.sh     # open what was built last
#
# Everything it needs goes into tmp/native_deps, nothing system-wide:
#   Qt 6.8 with Shader Tools (aqtinstall, from a venv), as native_gate_b.sh;
#   ONNX Runtime, the release archive, as native_gate_a.sh (Core ML on a Mac);
#   OpenCV core + imgproc + imgcodecs, built from source with its own PNG,
#   JPEG, TIFF, WebP and OpenEXR (a few minutes, once): the still decoder the
#   Python uses, without Homebrew's opencv and everything it pulls in.
# The model packages are found in dist/models (native_gate_a.sh exports one).
# Movies need ffmpeg and ffprobe on the PATH (brew install ffmpeg).
# Needs CMake 3.24+, Ninja or Make, a C++20 compiler (Xcode CLT on macOS),
# python3 and curl.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
QT_VERSION=${QT_VERSION:-6.8.3}
ORT_VERSION=${ORT_VERSION:-1.22.0}
CV_VERSION=${CV_VERSION:-4.10.0}
DEPS=tmp/native_deps
BUILD=build/native_app
JOBS=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 4)
mkdir -p "$DEPS"

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) OS=mac; QT_HOST=mac; QT_ARCH=clang_64; QT_DIR=macos; ORT_ARCHIVE=onnxruntime-osx-arm64-$ORT_VERSION ;;
  Linux-x86_64) OS=linux; QT_HOST=linux; QT_ARCH=linux_gcc_64; QT_DIR=gcc_64; ORT_ARCHIVE=onnxruntime-linux-x64-$ORT_VERSION ;;
  *) fail "Apple silicon or x86-64 Linux; on Windows use scripts/NATIVE_PHASE3_EXIT.ps1" ;;
esac
QT_ROOT=$PWD/$DEPS/Qt/$QT_VERSION/$QT_DIR
ORT_ROOT=$PWD/$DEPS/$ORT_ARCHIVE
CV_ROOT=$PWD/$DEPS/opencv-$CV_VERSION-install

if [ -z "${SKIP_BUILD:-}" ]; then
  # -------------------------------------------------------------------------
  if [ ! -f "$QT_ROOT/lib/cmake/Qt6ShaderTools/Qt6ShaderToolsConfig.cmake" ]; then
    say "Qt $QT_VERSION ($QT_ARCH, qtshadertools) into $DEPS/Qt"
    VENV=$DEPS/venv_tools
    [ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --quiet --upgrade aqtinstall
    "$VENV/bin/python" -m aqt install-qt "$QT_HOST" desktop "$QT_VERSION" "$QT_ARCH" -m qtshadertools -O "$DEPS/Qt"
  fi
  # Qt 6.8 links -framework AGL, which the macOS 15+ SDKs no longer ship.
  if [ "$OS" = mac ]; then
    sed -i '' '/target_link_libraries.*__opengl_agl_fw_path/d' "$QT_ROOT/lib/cmake/Qt6/FindWrapOpenGL.cmake"
  fi

  # -------------------------------------------------------------------------
  if [ ! -f "$ORT_ROOT/include/onnxruntime_cxx_api.h" ]; then
    say "ONNX Runtime $ORT_VERSION"
    curl -fL --retry 5 --retry-all-errors -o "$DEPS/$ORT_ARCHIVE.tgz" \
      "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ORT_ARCHIVE.tgz"
    tar -xzf "$DEPS/$ORT_ARCHIVE.tgz" -C "$DEPS"
    rm -f "$DEPS/$ORT_ARCHIVE.tgz"
  fi

  # -------------------------------------------------------------------------
  if [ ! -f "$CV_ROOT/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
    say "OpenCV $CV_VERSION (core, imgproc, imgcodecs) from source"
    CV_SRC=$DEPS/opencv-$CV_VERSION
    if [ ! -f "$CV_SRC/CMakeLists.txt" ]; then
      # The release archive, else a shallow clone of the tag (some networks
      # refuse GitHub's archive host but not git).
      if curl -fL --retry 3 -o "$DEPS/opencv-$CV_VERSION.tar.gz" \
           "https://github.com/opencv/opencv/archive/refs/tags/$CV_VERSION.tar.gz"; then
        tar -xzf "$DEPS/opencv-$CV_VERSION.tar.gz" -C "$DEPS"
      else
        rm -rf "$CV_SRC"
        git clone --quiet --depth 1 --branch "$CV_VERSION" https://github.com/opencv/opencv.git "$CV_SRC"
      fi
      rm -f "$DEPS/opencv-$CV_VERSION.tar.gz"
    fi
    GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
    cmake -S "$CV_SRC" -B "$DEPS/opencv-$CV_VERSION-build" ${GEN[@]+"${GEN[@]}"} \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$CV_ROOT" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_LIST=core,imgproc,imgcodecs -DBUILD_SHARED_LIBS=OFF \
      -DBUILD_ZLIB=ON -DBUILD_PNG=ON -DBUILD_JPEG=ON -DBUILD_TIFF=ON -DBUILD_WEBP=ON \
      -DWITH_OPENEXR=ON -DBUILD_OPENEXR=ON \
      -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
      -DWITH_OPENCL=OFF -DWITH_IPP=OFF -DWITH_ITT=OFF -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF \
      -DWITH_PROTOBUF=OFF -DWITH_ADE=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_1394=OFF \
      -DWITH_AVFOUNDATION=OFF -DWITH_V4L=OFF -DWITH_IMGCODEC_HDR=ON \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
      -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF -DBUILD_DOCS=OFF >/dev/null
    cmake --build "$DEPS/opencv-$CV_VERSION-build" --parallel "$JOBS"
    cmake --install "$DEPS/opencv-$CV_VERSION-build" >/dev/null
  fi

  # -------------------------------------------------------------------------
  say "Configure and build RUDRA, rudra-native and the app tests (Release)"
  GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
  TESTS=ON; [ -n "${SKIP_TESTS:-}" ] && TESTS=OFF
  # Qt6_DIR pins the fetched Qt over any Homebrew or distro Qt 6; --fresh drops
  # whatever an earlier configure found.
  cmake --fresh -S native -B "$BUILD" ${GEN[@]+"${GEN[@]}"} -DCMAKE_BUILD_TYPE=Release \
    -DRUDRA_BUILD_APP=ON -DRUDRA_BUILD_RENDER=ON -DRUDRA_BUILD_CLI=ON -DRUDRA_BUILD_TESTS=$TESTS \
    -DRUDRA_WITH_ONNXRUNTIME=ON "-DONNXRUNTIME_ROOT=$ORT_ROOT" -DRUDRA_WITH_LIBTORCH=OFF \
    -DRUDRA_WITH_OPENCV=ON "-DOpenCV_DIR=$CV_ROOT/lib/cmake/opencv4" \
    "-DCMAKE_PREFIX_PATH=$QT_ROOT" "-DQt6_DIR=$QT_ROOT/lib/cmake/Qt6" \
    "-DCMAKE_BUILD_RPATH=$ORT_ROOT/lib" >/dev/null || fail "cmake configure"
  targets=(RUDRA rudra-native)
  [ "$TESTS" = ON ] && targets+=(rudra_app_tests)
  cmake --build "$BUILD" --target "${targets[@]}" --parallel "$JOBS" || fail "build"

  # -------------------------------------------------------------------------
  if [ "$TESTS" = ON ]; then
    say "The app's Qt tests (offscreen)"
    TEST_EXE=$(find "$BUILD/app" -name rudra_app_tests -type f -perm -u+x | head -1)
    [ -n "$TEST_EXE" ] || fail "rudra_app_tests not built"
    QT_QPA_PLATFORM=offscreen "$TEST_EXE" || fail "app tests"
  fi
fi

# ---------------------------------------------------------------------------
if [ "$OS" = mac ]; then APP=$BUILD/app/RUDRA.app/Contents/MacOS/RUDRA; else APP=$BUILD/app/RUDRA; fi
[ -x "$APP" ] || fail "no app at $APP (run without SKIP_BUILD)"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || echo "note: no $tool on the PATH: stills work, movies will not (brew install ffmpeg)"
done
ls dist/models/*/manifest.json >/dev/null 2>&1 ||
  echo "note: no model package in dist/models: run scripts/native_gate_a.sh (it exports one), or copy one from the PC"
if [ -n "${NO_LAUNCH:-}" ]; then echo; echo "Built: $APP"; exit 0; fi

say "Open RUDRA"
export RUDRA_PACKAGE_ROOTS="$PWD/dist/models${RUDRA_PACKAGE_ROOTS:+:$RUDRA_PACKAGE_ROOTS}"
exec "$APP" "$@"
