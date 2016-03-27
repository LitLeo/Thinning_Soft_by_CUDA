#!/bin/bash
exe="okanoexec"
imgdir="../../img"
imgs=`ls $imgdir`

for img in $imgs; do
    ./$exe $imgdir/$img
done
