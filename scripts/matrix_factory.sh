#!/bin/bash

if [ "$1" = "small" ]

then
  echo "Running small matrix factory..."
  for d in $(seq 0 0.25 2.75)
  do
    d2=`echo "$d + 0.25" | bc`
    echo "run from $d to $d2"
    python scripts/make_matrices_small.py $d $d2 &
    sleep 1
  done
fi

if [ "$1" = "medium" ]

then
  echo "Running medium matrix factory..."
  for d in $(seq 0 1 9 )
  do
    d2=`echo "$d + 1.0" | bc`
    echo "run from $d to $d2"
    python scripts/make_matrices_medium.py $d $d2 &
    sleep 1
  done
fi

if [ "$1" = "large" ]

then
  echo "Running large matrix factory..."
  for d in $(seq 0 2 18)
  do
    d2=`echo "$d + 2.0" | bc`
    echo "run from $d to $d2"
    python scripts/make_matrices_large.py $d $d2 &
    sleep 1
  done
fi
