# # Benchmarking Input Dimensions
# 
# This file is to test some benchmarking of the $\mathcal{O}(n^2)$ implementation of the t-SNE algorithm.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
#get_ipython().magic('matplotlib inline')
import sys

if(len(sys.argv)<4):
    print("Must be called with timestamp, gcc/icc, compiler flag:")
    exit()
    
timestamp = sys.argv[1]
compiler = sys.argv[2]
flags = sys.argv[3:]
flags = " ".join(flags)

file_prefix = f"default@{timestamp}@{compiler}@{flags}"

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

N, tf, ff, cy = read_benchmark_exact(
    "./benchmarking/" + file_prefix + "@@", 
    stop=3000)
total_flops.append(tf); flops_by_function.append(ff); cycles.append(cy)
perf.append(tf/cy[:,-1])
perf_func.append(ff/cy[:,:-1])


# Plot the performance plot.
title = "t-SNE Exact Computation Performance on Intel Xeon E3-1285Lv5 Skylake, 3.0-3.6GHz"
plot(N, perf, labels=[compiler + ", flags: " + flags], title=title, store=True, store_name=file_prefix)


