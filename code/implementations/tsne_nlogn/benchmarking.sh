#!/bin/sh

# Choose between: "timing", "benchmark"
MODE=${1:-timing}
# Choose between: "default", "data_type", "compiler_flags"
BENCHMARK=${2:-default}

# Input size range
START=200
STOP=3000
INTERVAL=200

# Compiler flags for benchmarking
FLAGS1="-O3 -std=c++11"
FLAGS2="-O3 -std=c++11 -march=native"
FLAGS3="-O3 -std=c++11 -march=native -ffast-math"

# Algorithms parameters
MAX_ITER=1000       # Maximum number of iterations
DIMS=2              # Output dim
PERPLEXITY=50       # Perplexity best value
DATA_FILE="../../../data/mnist/train-images.idx3-ubyte"

# Choose compiler taking into account the computers running macOS
if [[ $(uname) == "Darwin" ]]; then
    CC=g++-4.9;
else
    CC=g++
fi
# Print compiler version
$CC --version

# Default compiler flags and source files
COMPILER_FLAGS="-O3 -std=c++11 -march=native"
SRC="tsne_nlogn.cpp trees/sptree.cpp"

TODAY=$(date +%Y%m%d_%H%M%S)

FILE_PREFIX="./benchmarking/$TODAY@$COMPILER_FLAGS"

case $MODE in
    "timing")
        # Perform some timing to check until which size make the
        # benchmarking
        if [ ! -d bin ] ; then mkdir -p bin ; fi;
        BIN="bin/tsne.o"
        $CC $COMPILER_FLAGS $SRC -o $BIN || exit
        for N in $(seq $START $INTERVAL $STOP); do
            printf "\nN: $N";
            time ./$BIN $DATA_FILE /dev/null $N $PERPLEXITY $DIMS $MAX_ITER;
        done;
        ;;
    "benchmark")
        case $BENCHMARK in
            "default")
                echo "** BENCHMARKING (default) **"
                COUNT_FILE="./benchmarking/$TODAY@$COMPILER_FLAGS@$NOTE@iters.txt"
                BENCH_FILE="./benchmarking/$TODAY@$COMPILER_FLAGS@$NOTE@cycles.txt"
                BIN_COUNT="bin/tsne_count.o"
                BIN_BENCH="bin/tsne_bench.o"

                echo "Compiling"
                $CC -DCOUNTING $COMPILER_FLAGS $SRC -o $BIN_COUNT || exit
                $CC -DBENCHMARK $COMPILER_FLAGS $SRC -o $BIN_BENCH || exit
                # Create the files to store the
                # touch "$COUNT_FILE"
                # touch "$BENCH_FILE"
                echo "Running..."
                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N";
                    ./$BIN_COUNT $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$COUNT_FILE";
                    ./$BIN_BENCH $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$BENCH_FILE";
                    printf " DONE\n"
                done;
                echo "Finish Successfully"
                ;;
            "data_type")
                echo "** BENCHMARKING DATA TYPE **"

                echo "Compiling"
                $CC -DCOUNTING $COMPILER_FLAGS $SRC -o tsne_count_d.o || exit
                $CC -DCOUNTING -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o tsne_count_f.o || exit
                $CC -DBENCHMARK $COMPILER_FLAGS $SRC -o tsne_bench_d.o || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o tsne_bench_f.o || exit

                echo "Running..."
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
                echo "Finish Successfully"
                ;;
            "compiler_flags")
                echo "** BENCHMARKING COMPILER FLAGS **"

                echo "Compiling"
                $CC -DCOUNTING $FLAGS1 $SRC -o tsne_cound_1.o || exit
                $CC -DCOUNTING $FLAGS2 $SRC -o tsne_cound_2.o || exit
                $CC -DCOUNTING $FLAGS3 $SRC -o tsne_cound_3.o || exit
                $CC -DBENCHMARK $FLAGS1 $SRC -o tsne_bench_1.o || exit
                $CC -DBENCHMARK $FLAGS2 $SRC -o tsne_bench_2.o || exit
                $CC -DBENCHMARK $FLAGS3 $SRC -o tsne_bench_3.o || exit

                FILE_PREFIX="./benchmarking/$TODAY"

                echo "Running..."
                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N ";
                    ./tsne_cound_1.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@$FLAGS1@double@iters.txt"
                    printf "."
                    ./tsne_cound_2.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@$FLAGS2@double@iters.txt"
                    printf "."
                    ./tsne_cound_3.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@$FLAGS3@double@iters.txt"
                    printf "."
                    ./tsne_bench_1.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@$FLAGS1@double@cycles.txt"
                    printf "."
                    ./tsne_bench_2.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@$FLAGS2@double@cycles.txt"
                    printf "."
                    ./tsne_bench_3.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER >> "$FILE_PREFIX@$FLAGS3@double@cycles.txt"
                    printf " DONE\n"
                done;
                echo "Finish Successfully"
                ;;
        esac
        ;;
    *)
        ;;
esac
