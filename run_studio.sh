#!/usr/bin/env bash
#
# Start RUDRA Studio.
#
# The first run builds a virtual environment beside this file and installs
# what the project needs. Every run after that checks the install is intact
# and goes straight to the server, so this is the only file you need.
#
#   ./run_studio.sh                        start the server
#   ./run_studio.sh --setup                reinstall, then start
#   ./run_studio.sh --port 9000            anything else goes to the server
#   ./run_studio.sh --device cpu --preload
#
# Set RUDRA_PYTHON to a Python you already have and this uses that instead of
# building a venv and downloading a second copy of torch.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
VENV="$PWD/.venv"
STAMP="$VENV/.rudra-setup"
force=0
args=()

for a in "$@"; do
    if [ "$a" = "--setup" ]; then force=1; else args+=("$a"); fi
done

die() { printf '\n  %s\n\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- interpreter
if [ -n "${RUDRA_PYTHON:-}" ]; then
    PY="$RUDRA_PYTHON"
    echo "Using RUDRA_PYTHON: $PY"
elif [ -x "$VENV/bin/python" ]; then
    PY="$VENV/bin/python"
else
    echo "No environment yet. Building one in .venv ..."
    base=""
    for c in python3.12 python3.11 python3.13 python3.10 python3 python; do
        command -v "$c" >/dev/null 2>&1 || continue
        if "$c" -c 'import sys; raise SystemExit(0 if (3,10)<=sys.version_info<(3,14) else 1)' 2>/dev/null; then
            base="$c"; break
        fi
    done
    [ -n "$base" ] || die "Could not find Python 3.10 to 3.13. Install one and run this again."
    echo "Using $base ($("$base" -V))"
    "$base" -m venv "$VENV" || die "Could not create the virtual environment. On Debian or Ubuntu you may need: sudo apt install python3-venv"
    PY="$VENV/bin/python"
    force=1
fi

"$PY" -c 'import sys' >/dev/null 2>&1 || die "The Python at $PY does not run. Delete .venv and try again."

# ------------------------------------------------------------------ up to date
# Reinstall when pyproject.toml has changed since the last successful setup,
# or when an import that should work does not.
want="$("$PY" -c 'import hashlib,sys; sys.stdout.write(hashlib.md5(open("pyproject.toml","rb").read()).hexdigest())')"
have="$(cat "$STAMP" 2>/dev/null || true)"

needs_install=0
[ "$force" = 1 ] && needs_install=1
[ "$want" != "$have" ] && needs_install=1
if [ "$needs_install" = 0 ]; then
    "$PY" -c 'import numpy, cv2, PIL, yaml, imageio, safetensors, tqdm, rudra' >/dev/null 2>&1 || needs_install=1
fi

# --------------------------------------------------------------------- install
if [ "$needs_install" = 1 ]; then
    cat <<'MSG'

Installing RUDRA and its dependencies. The first run downloads torch, which is
a couple of gigabytes, so give it a few minutes.

MSG
    "$PY" -m pip install --upgrade pip --quiet
    "$PY" -m pip install -e . || die "Install failed. The error is above."
    printf '%s' "$want" > "$STAMP"
fi

# ------------------------------------------------------------------------- run
"$PY" - <<'PYCHECK' 2>/dev/null || true
import torch
if torch.cuda.is_available():
    print(f"CUDA available: {torch.cuda.get_device_name(0)}")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    print("Metal (MPS) available. Start with --device mps.")
else:
    print("  No GPU in this environment. The server will run, slowly, on the CPU.")
    print("  For an NVIDIA card:  .venv/bin/pip install --force-reinstall torch \\")
    print("      --index-url https://download.pytorch.org/whl/cu124")
PYCHECK

echo
exec "$PY" ui/server.py ${args[@]+"${args[@]}"}
