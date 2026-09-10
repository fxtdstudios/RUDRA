#!/usr/bin/env bash
# Train RUDRA on your own footage.
#     ./train.sh /path/to/your_hdr_footage
#
# Everything else has a sensible default. See training/train_from_footage.py
# for the flags if you want something different.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
footage="${1:-}"

if [ -z "$footage" ]; then
  read -rp "Folder holding your HDR footage: " footage
fi
if [ -z "$footage" ]; then
  echo "No folder given. Nothing to do." >&2
  exit 1
fi
shift || true

# Prefer the repo's own venv, then whatever python3 is on PATH.
py="$repo/.venv/bin/python"
[ -x "$py" ] || py="$(command -v python3)"

if ! "$py" -c "import torch, cv2, numpy" 2>/dev/null; then
  echo
  echo "Installing requirements. This happens once."
  "$py" -m pip install -r "$repo/requirements.txt"
fi

exec "$py" "$repo/training/train_from_footage.py" "$footage" "$@"
