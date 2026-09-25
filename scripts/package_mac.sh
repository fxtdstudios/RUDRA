#!/usr/bin/env bash
# The RUDRA beta for macOS (Apple silicon): a self-contained RUDRA.app in a DMG.
#
#   scripts/package_mac.sh                 # build, test, package -> dist/beta/RUDRA-<version>-macos-arm64.dmg
#   SKIP_TESTS=1 scripts/package_mac.sh    # without the app's Qt tests
#   SKIP_BUILD=1 scripts/package_mac.sh    # package what native_app.sh built last
#
# The app is built by scripts/native_app.sh (Qt 6.8, ONNX Runtime with Core ML,
# OpenCV from source, all in tmp/native_deps). Into the bundle go:
#   Contents/Frameworks  Qt (macdeployqt) and ONNX Runtime
#   Contents/MacOS       RUDRA and rudra-native (the CLI: diff, video, deliver, batch)
#   Contents/Resources   the icon, the model package(s) from dist/models, LICENSE, NOTICE
# It is signed ad hoc (no Developer ID yet): the first open is right-click >
# Open, or `xattr -dr com.apple.quarantine RUDRA.app`. ffmpeg is not bundled
# (movies need an ffmpeg with libx265, prores_ks and zscale on the PATH, e.g.
# `brew install ffmpeg-full`); stills need nothing else.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(uname -s)-$(uname -m)" = "Darwin-arm64" ] || { echo "package_mac.sh runs on Apple silicon" >&2; exit 2; }
PY=${PYTHON:-python3}
QT_VERSION=${QT_VERSION:-6.8.3}
ORT_VERSION=${ORT_VERSION:-1.22.0}
BUILD=build/native_app
QT_ROOT=$PWD/tmp/native_deps/Qt/$QT_VERSION/macos
ORT_ROOT=$PWD/tmp/native_deps/onnxruntime-osx-arm64-$ORT_VERSION

say() { printf '\n== %s\n' "$*"; }
fail() { printf 'FAILED: %s\n' "$*" >&2; exit 1; }

ls dist/models/*/manifest.json >/dev/null 2>&1 ||
  fail "no model package in dist/models (scripts/native_gate_a.sh exports one)"

if [ -z "${SKIP_BUILD:-}" ]; then
  NO_LAUNCH=1 PYTHON="$PY" SKIP_TESTS="${SKIP_TESTS:-}" scripts/native_app.sh
fi
APP_SRC=$BUILD/app/RUDRA.app
CLI_SRC=$BUILD/cli/rudra-native
[ -d "$APP_SRC" ] && [ -x "$CLI_SRC" ] || fail "no build at $BUILD (run without SKIP_BUILD)"

VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP_SRC/Contents/Info.plist")
NAME=RUDRA-$VERSION-macos-arm64
OUT=dist/beta/$NAME
say "Package $NAME"
rm -rf "$OUT" "dist/beta/$NAME.dmg"
mkdir -p "$OUT"
APP=$OUT/RUDRA.app
cp -R "$APP_SRC" "$APP"
cp "$CLI_SRC" "$APP/Contents/MacOS/rudra-native"

# ONNX Runtime beside Qt, found through @rpath from both executables.
mkdir -p "$APP/Contents/Frameworks"
cp -a "$ORT_ROOT"/lib/libonnxruntime*.dylib "$APP/Contents/Frameworks/"
for exe in "$APP/Contents/MacOS/RUDRA" "$APP/Contents/MacOS/rudra-native"; do
  install_name_tool -add_rpath "@executable_path/../Frameworks" "$exe" 2>/dev/null || true
  # The build's absolute rpaths (tmp/native_deps) mean nothing on another Mac.
  for rp in $(otool -l "$exe" | awk '/LC_RPATH/ { getline; getline; print $2 }' | grep "^/" || true); do
    install_name_tool -delete_rpath "$rp" "$exe" 2>/dev/null || true
  done
done

say "Qt into the bundle (macdeployqt)"
"$QT_ROOT/bin/macdeployqt" "$APP" -executable="$APP/Contents/MacOS/rudra-native" -verbose=1

say "Models, licences"
mkdir -p "$APP/Contents/Resources/models"
cp -R dist/models/. "$APP/Contents/Resources/models/"
find "$APP/Contents/Resources/models" -name "*.safetensors" -delete -o -name "*.config.json" -delete
cp LICENSE NOTICE "$APP/Contents/Resources/"
[ -f checkpoints/LICENSE ] && cp checkpoints/LICENSE "$APP/Contents/Resources/LICENSE-weights"

say "Sign (ad hoc)"
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

say "Check the bundle runs from where it is"
# Nothing may still point into the build tree or Homebrew.
leaks=$(for f in "$APP/Contents/MacOS/"* "$APP/Contents/Frameworks/"*.dylib; do otool -L "$f" | tail -n +2; done |
        grep -E "$PWD|/opt/homebrew|/usr/local" || true)
[ -z "$leaks" ] || { printf '%s\n' "$leaks"; fail "the bundle links outside itself"; }
"$APP/Contents/MacOS/rudra-native" info "$APP/Contents/Resources/models/$(ls "$APP/Contents/Resources/models" | grep -v models.json | head -1)" >/dev/null ||
  fail "rudra-native in the bundle cannot read its model"
# The Cocoa platform: macdeployqt ships that one, not offscreen.
"$APP/Contents/MacOS/RUDRA" --theme-check "$OUT/theme-check.json" >/dev/null ||
  fail "RUDRA in the bundle does not start"
rm -f "$OUT/theme-check.json"

say "Disk image"
cp docs/BETA.md "$OUT/Read me first.md" 2>/dev/null || true
ln -s /Applications "$OUT/Applications"
hdiutil create -volname "RUDRA $VERSION" -srcfolder "$OUT" -ov -format UDZO "dist/beta/$NAME.dmg" >/dev/null
shasum -a 256 "dist/beta/$NAME.dmg" | tee "dist/beta/$NAME.dmg.sha256"
echo
echo "Built: dist/beta/$NAME.dmg"
