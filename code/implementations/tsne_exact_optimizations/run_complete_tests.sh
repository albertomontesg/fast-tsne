#!/bin/sh
mkdir -p bin
sh benchmarking.sh benchmark default
sh benchmarking.sh benchmark data_type
sh benchmarking.sh benchmark compiler_flags
sh benchmarking.sh benchmark input_dimensions
