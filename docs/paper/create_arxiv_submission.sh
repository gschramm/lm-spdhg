#!/bin/bash

TMPDIR=`mktemp -d -p .`

cp ms.tex $TMPDIR
cp abstract.tex $TMPDIR
cp title.tex $TMPDIR
cp preamble.tex $TMPDIR
cp content.tex $TMPDIR
cp supplement.tex $TMPDIR
cp gs.sty $TMPDIR
cp ms.bib $TMPDIR
cp figure*.png $TMPDIR
cp figure*.pdf $TMPDIR

cd $TMPDIR

latexmk -pdf ms.tex
latexmk -C ms.tex
rm ms-blx.bib
rm ms.run.xml

latexmk -pdf supplement.tex
latexmk -C supplement.tex
rm supplement.bbl
rm supplement-blx.bib
rm supplement.run.xml

zip ../arxiv.zip *

cd ..

rm -r $TMPDIR

