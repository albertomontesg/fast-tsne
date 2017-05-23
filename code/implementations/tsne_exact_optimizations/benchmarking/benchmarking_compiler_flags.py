# # Benchmarking Input Dimensions
# 
# This file is to test some benchmarking of the $\mathcal{O}(n^2)$ implementation of the t-SNE algorithm.
# 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().magic('matplotlib inline')
import sys

if(len(sys.argv)<3):
    print("Must be called with timestamp, gcc/icc")
    exit()
    
timestamp = sys.argv[1]
compiler = sys.argv[2]

FLAGS_GCC = (
    "-O3 -std=c++11",
    "-O3 -std=c++11 -march=native",
    "-O3 -std=c++11 -march=native -ffast-math"
)

FLAGS_ICC = (
    "-O3 -std=c++11",
    "-O3 -std=c++11 -xHost",
    "-O3 -std=c++11 -fast"
)

FLAGS=""
if compiler == "gcc":
    FLAGS=FLAGS_GCC
elif compiler == "icc":
    FLAGS=FLAGS_ICC
else:
    print("incorrect compiler specified (gcc/icc allowed)")
    exit()

file_prefix = f"compiler_flags@{timestamp}@{compiler}"

mpl.rcParams['figure.figsize'] = (8.1, 5)
mpl.rcParams['figure.dpi'] = 600
mpl.rcParams['font.family'] = 'Roboto'
mpl.rcParams['font.size'] = 15

from visualization import *
from util_tsne_exact import FUNCTION_NAMES, read_benchmark_exact


# Read the data
total_flops = []
flops_by_function = []
perf = []
perf_func = []
cycles = []

for flag in FLAGS:
    N, tf, ff, cy = read_benchmark_exact(
        "./benchmarking/" + file_prefix + f"@{flag}@float@", 
        stop=3000)
    total_flops.append(tf); flops_by_function.append(ff); cycles.append(cy)
    perf.append(tf/cy[:,-1])
    perf_func.append(ff/cy[:,:-1])


# Plot the performance plot.
title = "t-SNE Exact Computation Performance on Intel Xeon E3-1285Lv5 Skylake, 3.0-3.6GHz"
plot(N, perf, labels=FLAGS, title=title, store=True, store_name=file_prefix)


