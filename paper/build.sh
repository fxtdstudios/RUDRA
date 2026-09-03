#!/usr/bin/env bash
# Rebuild the arXiv PDF from PAPER_DRAFT_2026-08-29.md.
#   bash paper/build.sh
# Windows:  python paper\mdtotex.py && pdflatex -output-directory=paper paper\main.tex  (x2)
set -e
cd "$(dirname "$0")"
python mdtotex.py
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "built: $(pwd)/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"
