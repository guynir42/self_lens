#!/bin/bash

for d in $(seq 0 0.25 2.25)

do
  d2=`echo "$d + 0.25" | bc`
  echo "run from $d to $d2"
  python scripts/make_matrices_small.py $d $d2 &
  sleep 1
done