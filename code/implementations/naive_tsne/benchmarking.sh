#!/usr/bin/shell

MODE="benchmark"

# Algorithms parameters
MAX_ITER=1000       # Maximum number of iterations
DIMS=2              # Output dim
PERPLEXITY=50       # Perplexity best value
DATA_FILE="../../../data/mnist/train-images.idx3-ubyte"


CC=g++-6
COMPILER_FLAGS="-O3 -march=native"
BIN=tsne.o
BIN_COUNT=tsne_count.o
BIN_BENCH=tsne_bench.o

TODAY=$(date +%Y%m%d_%H%M%S)

COUNT_FILE="./benchmarking/$TODAY-iters.txt"
BENCH_FILE="./benchmarking/$TODAY-cycles.txt"

# Input size range
START=200
STOP=4000
INTERVAL=200

case $MODE in
    "timing")
        # Perform some timing to check until which size make the
        # benchmarking
        $CC $COMPILER_FLAGS naive_tsne.cpp ../utils/io.c -o $BIN;
        for N in $(seq $START $INTERVAL $STOP); do
            echo "/n/nN: $N";
            time ./$BIN $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER;
        done;
        ;;
    "benchmark")
        $CC -DCOUNTING $COMPILER_FLAGS naive_tsne.cpp ../utils/io.c -o $BIN_COUNT;
        $CC -DBENCHMARK $COMPILER_FLAGS naive_tsne.cpp ../utils/io.c -o $BIN_BENCH;
        # Create the files to store the
        touch $COUNT_FILE
        touch $BENCH_FILE
        for N in $(seq $START $INTERVAL $STOP); do
            printf "$N";
            ./$BIN_COUNT $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> $COUNT_FILE;
            ./$BIN_BENCH $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> $BENCH_FILE;
            printf " DONE\n"
        done;
        ;;
    *)
        ;;
esac


if [[ $TIMING == 1 ]]; then
    $CC $COMPILER_FLAGS naive_tsne.cpp ../utils/io.c -o $BIN;
    for N in $(seq $START $INTERVAL $STOP); do
        echo "/n/nN: $N";
        time ./$BIN ../../../data/mnist/train-images.idx3-ubyte result.dat $N 50 2 1000;
    done;
fi
