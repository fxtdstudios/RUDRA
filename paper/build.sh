#!/usr/bin/env bash
# Build paper/main.pdf.
#
#   bash paper/build.sh
#   Windows:  cd paper; pdflatex -interaction=nonstopmode main.tex   (twice)
#
# The LaTeX is the source in this repository. _body.tex and _abstract.tex are
# committed and build on their own; nothing else is needed.
#
# They were originally generated from a markdown draft by a converter. The
# converter is gone (16 Sep 2026): while it existed, this script ran it
# whenever the draft was present, and on the author's machine it always was,
# so a build silently rewrote hand-edited .tex from a stale markdown. The
# test in tests/test_committed_checkpoint_2026_09_03.py asserts it stays gone.
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
if grep -q "^!" main.log; then
  echo "FAILED:"; grep -n "^!" main.log; exit 1
fi
echo "built: $(pwd)/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"

# research/ is the copy people browse to from the repo front page. It replaced
# RUDRA_V01.pdf, an early draft that stayed there long after it stopped being
# true. Written from the build so the two cannot drift; a test asserts they are
# byte-identical.
cp -f main.pdf ../research/RUDRA_HDR_2026.pdf
echo "copied: research/RUDRA_HDR_2026.pdf"
