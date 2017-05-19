To install python: module load new gcc/4.8.2 python/3.6.0
To run all benchmarks: bsub -n 1 -R "rusage[mem=4096]" run_complete_tests.sh
