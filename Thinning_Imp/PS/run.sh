#!/bin/bash
exe="thinexec"
imgdir="../img"
imgs=`ls $imgdir`
out="out.txt"
>$out
for img in $imgs; do
    ./$exe $imgdir/$img | tee -a $out
done
