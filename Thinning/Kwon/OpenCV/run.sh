#!/bin/bash
exe="main"
imgdir="../../img"
imgs=`ls $imgdir`

for img in $imgs; do
    `./$exe $imgdir/$img`
done
