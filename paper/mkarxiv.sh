#!/usr/bin/env bash
# Build a flat, self-contained arXiv upload from the committed LaTeX.
#   bash paper/mkarxiv.sh   ->  paper/rudra-arxiv.tar.gz
#
# Staging happens in a temp dir, not in the repo: the mounted sandbox this is
# often run from cannot unlink files, so a build that littered the repo could
# never clean up after itself.
#
# arXiv runs latex but not bibtex, so main.bbl travels with the sources and the
# \bibliography line is replaced by it. refs.bib goes along for the reader.
set -e
cd "$(dirname "$0")"
HERE="$(pwd)"

bash build.sh

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp main.tex abstract.tex appendix.tex sec*.tex refs.bib main.bbl "$STAGE/"
cp ../docs/figures/*.pdf "$STAGE/"

# arXiv uploads are flat: figures sit beside the .tex, where LaTeX finds them
# without \graphicspath. Leaving the relative paths in would break their build.
sed -i 's|^\\graphicspath.*$|\\graphicspath{{./}}|' "$STAGE/main.tex"

cd "$STAGE"
pdflatex -interaction=nonstopmode main.tex >/dev/null
pdflatex -interaction=nonstopmode main.tex >/dev/null
if grep -q "^!" main.log; then
  echo "FAILED: errors building the flat package"; grep -n "^!" main.log; exit 1
fi
if grep -q "Warning: Reference\|Warning: Citation" main.log; then
  echo "FAILED: undefined references in the flat package"
  grep -n "Warning: Reference\|Warning: Citation" main.log; exit 1
fi
pages=$(pdfinfo main.pdf | awk '/Pages/{print $2}')
rm -f main.aux main.log main.out main.toc main.pdf

tar -czf "$HERE/rudra-arxiv.tar.gz" .
cd "$HERE"
echo "arxiv package OK: builds to $pages pages, $(du -h rudra-arxiv.tar.gz | cut -f1)"
echo "  upload:        paper/rudra-arxiv.tar.gz"
chars=$(python -c "import io;print(len(io.open('ABSTRACT_ARXIV.txt',encoding='utf-8').read().strip()))")
echo "  abstract field: paper/ABSTRACT_ARXIV.txt ($chars chars; arXiv caps at 1920)"
[ "$chars" -gt 1920 ] && echo "  WARNING: over the cap -- trim before submitting"
