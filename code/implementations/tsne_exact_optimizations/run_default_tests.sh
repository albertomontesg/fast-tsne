#!/bin/sh
mkdir -p bin

# gcc benchmarks
sh benchmarking.sh benchmark default gcc

# icc benchmarks
sh benchmarking.sh benchmark default icc
