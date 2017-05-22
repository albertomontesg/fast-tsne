To install python: module load new gcc/4.8.2 python/3.6.0
To install icc: module load new intel/2017.0.098
To run all benchmarks: bsub -n 1 -R beta -R "rusage[mem=4096]" sh run_complete_tests.sh
