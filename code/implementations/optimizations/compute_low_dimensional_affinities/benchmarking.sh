#!/bin/sh

# Choose compiler taking into account the computers running macOS
if [ "$COMPILER" == "gcc" ]; then
    if [[ $(uname) == "Darwin" ]]; then
        CC=g++-4.9;
    else
        CC=g++
    fi
elif [ "$COMPILER" == "icc" ]; then
    CC=icc
fi
# Print compiler version
$CC --version

CFLAGS="-O3 -march=native -mavx -mfma -lstdc++ -std=c++11"
SRC="main.cpp"
BIN="run.o"
OUT_FILE=${1:-"benchmarking/test.csv"}


$CC $CFLAGS $SRC -o ./$BIN
./$BIN > $OUT_FILE
