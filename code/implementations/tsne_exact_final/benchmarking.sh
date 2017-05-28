#!/bin/sh

COMPILER="gcc"
# Choose compiler taking into account the computers running macOS
if [ "$COMPILER" == "gcc" ]; then
    FLAGS="-O3 -march=native -mavx -mfma -lstdc++ -std=c++11"
    if [[ $(uname) == "Darwin" ]]; then
        CC=g++-4.9;
    else
        CC=g++
    fi
elif [ "$COMPILER" == "icc" ]; then
    CC=icc
    FLAGS="-O3 -march=core-avx2 -std=c++11"
fi
# Print compiler version
$CC --version

SRC="tsne_exact_optimized.cpp"

HEADER='N,pairwise_squared_euclidean_distance,pairwise_affinity_perplexity,symmetrize_affinities,low_dimensional_affinities,gradient_computation_update_normalize'
TODAY=$(date +%Y%m%d_%H%M%S)
DATASET=../../../data/mnist/train-images.idx3-ubyte
MAX_ITER=1000       # Maximum number of iterations
DIMS=2              # Output dim
PERPLEXITY=50       # Perplexity best value


START=500
STOP=10000
INTERVAL=500

case $1 in
    baseline)
        CYCLES_FILE=benchmarking/baseline/$TODAY@cycles.csv
        COUNT_FILE=benchmarking/baseline/$TODAY@iters.csv
        CYCLES_BIN=./bin/tsne_baseline_bench.o
        COUNT_BIN=./bin/tsne_baseline_count.o

        touch $CYCLES_FILE
        touch $COUNT_FILE

        echo $HEADER > $CYCLES_FILE
        echo "N, cycles" > $COUNT_FILE

        $CC $FLAGS -DBASELINE -DCOUNTING $SRC -o $COUNT_BIN
        $CC $FLAGS -DBASELINE -DBENCHMARK $SRC -o $CYCLES_BIN

        for N in $(seq $START $INTERVAL $STOP); do
            printf "$N";
            $COUNT_BIN $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> $COUNT_FILE
            printf "."
            $CYCLES_BIN $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> $CYCLES_FILE
            printf " DONE\n"
        done;

        ;;
    scalar)
        CYCLES_FILE=benchmarking/scalar/$TODAY@cycles.csv
        COUNT_FILE=benchmarking/scalar/$TODAY@iters.csv
        CYCLES_BIN=./bin/tsne_scalar_bench.o
        COUNT_BIN=./bin/tsne_scalar_count.o

        touch $CYCLES_FILE
        touch $COUNT_FILE

        echo $HEADER > $CYCLES_FILE
        echo "N, cycles" > $COUNT_FILE

        $CC $FLAGS -DSCALAR -DCOUNTING $SRC -o $COUNT_BIN
        $CC $FLAGS -DSCALAR -DBENCHMARK $SRC -o $CYCLES_BIN

        for N in $(seq $START $INTERVAL $STOP); do
            printf "$N";
            $COUNT_BIN $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> $COUNT_FILE
            printf "."
            $CYCLES_BIN $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> $CYCLES_FILE
            printf " DONE\n"
        done;

        ;;
    avx)
        CYCLES_FILE=benchmarking/avx/$TODAY@cycles.csv
        COUNT_FILE=benchmarking/avx/$TODAY@iters.csv
        CYCLES_BIN=./bin/tsne_avx_bench.o
        COUNT_BIN=./bin/tsne_avx_count.o

        touch $CYCLES_FILE
        touch $COUNT_FILE

        echo $HEADER > $CYCLES_FILE
        echo "N, cycles" > $COUNT_FILE

        $CC $FLAGS -DAVX -DCOUNTING $SRC -o $COUNT_BIN
        $CC $FLAGS -DAVX -DBENCHMARK $SRC -o $CYCLES_BIN

        for N in $(seq $START $INTERVAL $STOP); do
            printf "$N";
            $COUNT_BIN $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> $COUNT_FILE
            printf "."
            $CYCLES_BIN $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> $CYCLES_FILE
            printf " DONE\n"
        done;

        ;;
esac
