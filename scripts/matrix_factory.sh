#!/bin/bash

for d in $(seq 0 0.2 2)

do
  d2=`echo "$d + 0.2" | bc`
  echo "run from $d to $d2"
  python scripts/make_matrices_small.py $d $d2 &
done