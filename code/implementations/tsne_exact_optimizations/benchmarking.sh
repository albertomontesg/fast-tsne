#!/usr/bin/shell

MODE="benchmark"
BENCHMARK="data_type"
NOTE=""

# Algorithms parameters
MAX_ITER=1000       # Maximum number of iterations
DIMS=2              # Output dim
PERPLEXITY=50       # Perplexity best value
DATA_FILE="../../../data/mnist/train-images.idx3-ubyte"

if [[ $(uname) == "Darwin" ]]; then
    CC=g++-4.9;
else
    CC=g++
fi

# See compiler version
$CC --version

COMPILER_FLAGS="-O3 -march=native -std=c++11"
SRC="tsne_exact.cpp ../utils/io.c ./computations/normalize.c ./computations/compute_squared_euclidean_distance.c ./computations/compute_pairwise_affinity_perplexity.c ./computations/symmetrize_affinities.c ./computations/early_exageration.c ./computations/compute_low_dimensional_affinities.c ./computations/gradient_computation.c ./computations/gradient_update.c"
BIN=tsne.o
BIN_COUNT=tsne_count.o
BIN_BENCH=tsne_bench.o

TODAY=$(date +%Y%m%d_%H%M%S)

COUNT_FILE="./benchmarking/$TODAY@$COMPILER_FLAGS@$NOTE@iters.txt"
BENCH_FILE="./benchmarking/$TODAY@$COMPILER_FLAGS@$NOTE@cycles.txt"
FILE_PREFIX="./benchmarking/$TODAY@$COMPILER_FLAGS"

# Input size range
START=200
STOP=3000
INTERVAL=200

case $MODE in
    "timing")
        # Perform some timing to check until which size make the
        # benchmarking
        $CC $COMPILER_FLAGS $SRC -o $BIN;
        for N in $(seq $START $INTERVAL $STOP); do
            echo "/n/nN: $N";
            time ./$BIN $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER;
        done;
        ;;
    "benchmark")
        case $BENCHMARK in
            "default")
                $CC -DCOUNTING $COMPILER_FLAGS $SRC -o $BIN_COUNT;
                $CC -DBENCHMARK $COMPILER_FLAGS $SRC -o $BIN_BENCH;
                # Create the files to store the
                # touch "$COUNT_FILE"
                # touch "$BENCH_FILE"
                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N";
                    ./$BIN_COUNT $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$COUNT_FILE";
                    ./$BIN_BENCH $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$BENCH_FILE";
                    printf " DONE\n"
                done;
                ;;
            "data_type")
                $CC -DCOUNTING $COMPILER_FLAGS $SRC -o tsne_count_d.o
                $CC -DCOUNTING -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o tsne_count_f.o
                $CC -DBENCHMARK $COMPILER_FLAGS $SRC -o tsne_bench_d.o;
                $CC -DBENCHMARK -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o tsne_bench_f.o;

                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N ";
                    ./tsne_count_d.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@double@iters.txt"
                    printf "."
                    ./tsne_count_f.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@float@iters.txt"
                    printf "."
                    ./tsne_bench_d.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@double@cycles.txt"
                    printf "."
                    ./tsne_bench_f.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@float@cycles.txt"
                    printf " DONE\n"
                done;
                ;;
        esac
        ;;
    *)
        ;;
esac
