#!/usr/bin/env bash
# Build paper/main.pdf.
#
#   bash paper/build.sh
#   Windows:  cd paper; pdflatex -interaction=nonstopmode main.tex   (twice)
#
# The LaTeX is the source in this repository. _body.tex and _abstract.tex are
# committed and build on their own; nothing else is needed.
#
# They were originally generated from a markdown draft by mdtotex.py, which is
# kept for whoever still has that draft. It is skipped when the markdown is not
# here, which is the normal case for a clone.
set -e
cd "$(dirname "$0")"

DRAFT="../PAPER_DRAFT_2026-08-29.md"
if [ -f "$DRAFT" ]; then
  python mdtotex.py
else
  echo "no markdown draft; building the committed LaTeX"
fi

pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
if grep -q "^!" main.log; then
  echo "FAILED:"; grep -n "^!" main.log; exit 1
fi
echo "built: $(pwd)/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
