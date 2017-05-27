#!/bin/sh

COMPILER="icc"
# Choose compiler taking into account the computers running macOS
if [ "$COMPILER" == "gcc" ]; then
    FLAGS="-O3 -march=native -mavx -mfma -lstdc++ -std=c++11"
    if [[ $(uname) == "Darwin" ]]; then
        CC=g++-4.9;
    else
        CC=g++
    fi
elif [ "$COMPILER" == "icc" ]; then
    CC=icc
    FLAGS="-O3 -march=core-avx2 -std=c++11"
fi
# Print compiler version
$CC --version

SRC="main.cpp"
BIN="run.o"
OUT_FILE=${1:-"benchmarking/test.csv"}


$CC $FLAGS $SRC -o ./$BIN
./$BIN > $OUT_FILE
