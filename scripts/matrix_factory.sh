#!/bin/bash

#for d in $(seq 0 0.25 2.75)
#for d in $(seq 0 1 9 )
for d in $(seq 0 2 18)

do
#  d2=`echo "$d + 0.25" | bc`
#  d2=`echo "$d + 1.0" | bc`
  d2=`echo "$d + 2.0" | bc`
  echo "run from $d to $d2"
#  python scripts/make_matrices_small.py $d $d2 &
#  python scripts/make_matrices_medium.py $d $d2 &
  python scripts/make_matrices_large.py $d $d2 &
  sleep 1
done