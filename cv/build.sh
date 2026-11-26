#!/usr/bin/env bash
# Builds assets/cv.pdf from the LaTeX sources in this folder. GitHub runs this
# on every push, so edit the .tex files here and the site's CV follows.
# Locally it needs a TeX distribution with latexmk (MacTeX has one).
set -euo pipefail
cd "$(dirname "$0")"
out=$(mktemp -d)
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="$out" phd.tex
cp "$out/phd.pdf" ../assets/cv.pdf
echo "Wrote assets/cv.pdf"
