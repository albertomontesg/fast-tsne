#!/bin/sh
mkdir -p bin

# gcc benchmarks
sh benchmarking.sh benchmark default gcc
#sh benchmarking.sh benchmark data_type gcc
sh benchmarking.sh benchmark compiler_flags gcc
sh benchmarking.sh benchmark input_dimensions gcc

# icc benchmarks
sh benchmarking.sh benchmark default icc
#sh benchmarking-icc.sh benchmark data_type icc
sh benchmarking.sh benchmark compiler_flags icc
sh benchmarking.sh benchmark input_dimensions icc
