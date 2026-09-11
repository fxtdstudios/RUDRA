#!/usr/bin/env bash
# Build paper/main.pdf.
#
#   bash paper/build.sh
#   Windows:  cd paper
#             pdflatex -interaction=nonstopmode main.tex
#             bibtex main
#             pdflatex -interaction=nonstopmode main.tex   (twice)
#
# The LaTeX is the source. There is no markdown master: the paper was once
# generated from a markdown draft through mdtotex.py, and that pipeline could
# not express equations, numbered floats or cross-references, all of which the
# paper now depends on. mdtotex.py and the generated _body.tex / _abstract.tex
# were removed when it was rewritten.
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex >/dev/null
bibtex main >/dev/null || true
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null

if grep -q "^!" main.log; then
  echo "FAILED:"; grep -n "^!" main.log; exit 1
fi
# An undefined reference is a silent defect in a built PDF: it renders as a
# bold ?? that a skim reads straight past. Treat it as a build failure.
if grep -q "Warning: Reference\|Warning: Citation\|undefined on input line" main.log; then
  echo "FAILED: undefined references or citations:"
  grep -n "Warning: Reference\|Warning: Citation" main.log
  exit 1
fi
echo "built: $(pwd)/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pages)"

# research/ is the copy people browse to from the repo front page. It replaced
# RUDRA_V01.pdf, an early draft that stayed there long after it stopped being
# true. Written from the build so the two cannot drift; a test asserts they are
# byte-identical.
cp -f main.pdf ../research/RUDRA_HDR_2026.pdf
echo "copied: research/RUDRA_HDR_2026.pdf"
