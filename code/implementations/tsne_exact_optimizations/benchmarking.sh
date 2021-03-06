#!/bin/sh

# Choose between: "timing", "benchmark"
MODE=${1:-benchmark}
# Choose between: "default", "data_type", "compiler_flags", "input_dimensions"
BENCHMARK=${2:-default}

COMPILER=${3:-gcc}

# Input size range
START=200
STOP=3000
INTERVAL=200

#START=20
#STOP=300
#INTERVAL=20

# Compiler flags for benchmarking
FLAGS1="-O3 -std=c++11"
FLAGS2="-O3 -std=c++11 -march=native"
FLAGS3="-O3 -std=c++11 -march=native -ffast-math"

# Algorithms parameters
MAX_ITER=1000       # Maximum number of iterations
DIMS=2              # Output dim
INPUT_DIMS=512      # Input dim for generated data
#INPUT_DIMS=8
PERPLEXITY=50       # Perplexity best value


# Choose compiler taking into account the computers running macOS
if [ "$COMPILER" == "gcc" ]; then
    if [[ $(uname) == "Darwin" ]]; then
        CC=g++-4.9;
    else
        CC=g++
    fi
elif [ "$COMPILER" == "icc" ]; then
    CC=icc
fi
# Print compiler version
$CC --version

# Default compiler flags and source files
COMPILER_FLAGS="-O3 -std=c++11 -march=native"
SRC="tsne_exact_op1_highdim.cpp"

TODAY=$(date +%Y%m%d_%H%M%S)

FILE_PREFIX="./benchmarking/$BENCHMARK@$TODAY@$COMPILER@$COMPILER_FLAGS"

# MNIST data set:
# DATA_FILE="../../../data/mnist/train-images.idx3-ubyte"
# Generated data set:
CLUSTER_SIZE=10
DATA_FILE=../../../data/${CLUSTER_SIZE}_${INPUT_DIMS}_$STOP
if [ ! -f $DATA_FILE ]; then
    echo "Creating data set with $CLUSTER_SIZE clusters, $INPUT_DIMS dimensions and $STOP entries"
    python3 ../../../data/generate_data.py $CLUSTER_SIZE $INPUT_DIMS $STOP ../../../data/
    echo "Writing file to $DATA_FILE"
else
    echo "Loaded data set with $CLUSTER_SIZE clusters, $INPUT_DIMS dimensions and $STOP entries"
fi

case $MODE in
    "timing")
        # Perform some timing to check until which size make the
        # benchmarking
        if [ ! -d bin ] ; then mkdir -p bin ; fi;
        BIN="bin/tsne.o"
        $CC $COMPILER_FLAGS $SRC -o $BIN || exit
        for N in $(seq $START $INTERVAL $STOP); do
            printf "\nN: $N ";
            time ./$BIN $DATA_FILE /dev/null $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS;
        done;
        ;;
    "benchmark")
        case $BENCHMARK in
            "default")
                echo "** BENCHMARKING (default:$COMPILER_FLAGS, single precision) **"
                COUNT_FILE="$FILE_PREFIX@$NOTE@iters.txt"
                BENCH_FILE="$FILE_PREFIX@$NOTE@cycles.txt"
                BIN_COUNT="bin/tsne_count.o"
                BIN_BENCH="bin/tsne_bench.o"

                echo "Compiling"
                $CC -DCOUNTING -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o $BIN_COUNT || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o $BIN_BENCH || exit

                echo "Running..."
                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N";
                    ./$BIN_COUNT $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$COUNT_FILE";
                    ./$BIN_BENCH $DATA_FILE result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$BENCH_FILE";
                    printf " DONE\n"
                done;
                echo "plotting"
                python3 ./benchmarking/benchmarking_default.py $TODAY $COMPILER $COMPILER_FLAGS
                echo "Finished Successfully"
                ;;
            "data_type")
                echo "** BENCHMARKING DATA TYPE **"

                echo "Compiling"
                $CC -DCOUNTING $COMPILER_FLAGS $SRC -o bin/tsne_count_d.o || exit
                $CC -DCOUNTING -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o bin/tsne_count_f.o || exit
                $CC -DBENCHMARK $COMPILER_FLAGS $SRC -o bin/tsne_bench_d.o || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o bin/tsne_bench_f.o || exit

                echo "Running..."
                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N ";
                    ./bin/tsne_count_d.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@double@iters.txt"
                    printf "."
                    ./bin/tsne_count_f.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@float@iters.txt"
                    printf "."
                    ./bin/tsne_bench_d.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@double@cycles.txt"
                    printf "."
                    ./bin/tsne_bench_f.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@float@cycles.txt"
                    printf " DONE\n"
                done;
                echo "Finished Successfully"
                ;;
            "compiler_flags")
                echo "** BENCHMARKING COMPILER FLAGS **"

                echo "Compiling"
                $CC -DCOUNTING -DSINGLE_PRECISION $FLAGS1 $SRC -o bin/tsne_count_1.o || exit
                $CC -DCOUNTING -DSINGLE_PRECISION $FLAGS2 $SRC -o bin/tsne_count_2.o || exit
                $CC -DCOUNTING -DSINGLE_PRECISION $FLAGS3 $SRC -o bin/tsne_count_3.o || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $FLAGS1 $SRC -o bin/tsne_bench_1.o || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $FLAGS2 $SRC -o bin/tsne_bench_2.o || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $FLAGS3 $SRC -o bin/tsne_bench_3.o || exit

                FILE_PREFIX="./benchmarking/$BENCHMARK@$TODAY"

                echo "Running..."
                for N in $(seq $START $INTERVAL $STOP); do
                    printf "$N ";
                    ./bin/tsne_count_1.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@$COMPILER@$FLAGS1@float@iters.txt"
                    printf "."
                    ./bin/tsne_count_2.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@$COMPILER@$FLAGS2@float@iters.txt"
                    printf "."
                    ./bin/tsne_count_3.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@$COMPILER@$FLAGS3@float@iters.txt"
                    printf "."
                    ./bin/tsne_bench_1.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@$COMPILER@$FLAGS1@float@cycles.txt"
                    printf "."
                    ./bin/tsne_bench_2.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@$COMPILER@$FLAGS2@float@cycles.txt"
                    printf "."
                    ./bin/tsne_bench_3.o $DATA_FILE /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $INPUT_DIMS >> "$FILE_PREFIX@$COMPILER@$FLAGS3@float@cycles.txt"
                    printf " DONE\n"
                done;
                echo "plotting"
                python3 ./benchmarking/benchmarking_input_dimension.py $TODAY $COMPILER
                echo "Finished Successfuly"
                ;;
            "input_dimensions")
                echo "** BENCHMARKING INPUT DIMENSIONS **"

                echo "Compiling(single precision, march=native)"
                $CC -DCOUNTING -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o bin/tsne_count_f.o || exit
                $CC -DBENCHMARK -DSINGLE_PRECISION $COMPILER_FLAGS $SRC -o bin/tsne_bench_f.o || exit

                DIM1=$((INPUT_DIMS/4))
                DATA_FILE_1=../../../data/${CLUSTER_SIZE}_${DIM1}_$STOP
                DIM2=$((INPUT_DIMS))
                DATA_FILE_2=$DATA_FILE # standard size
                DIM3=$((INPUT_DIMS*4))
                DATA_FILE_3=../../../data/${CLUSTER_SIZE}_${DIM3}_$STOP

                if [ ! -f $DATA_FILE_1 ]; then
                    echo "Creating data set with $CLUSTER_SIZE clusters, $DIM1 dimensions and $STOP entries"
                    python3 ../../../data/generate_data.py $CLUSTER_SIZE $DIM1 $STOP ../../../data/
                    echo "Writing file to $DATA_FILE_1"
                else
                    echo "Loaded data set with $CLUSTER_SIZE clusters, $DIM1 dimensions and $STOP entries"
                fi

                if [ ! -f $DATA_FILE_3 ]; then
                    echo "Creating data set with $CLUSTER_SIZE clusters, $DIM3 dimensions and $STOP entries"
                    python3 ../../../data/generate_data.py $CLUSTER_SIZE $DIM3 $STOP ../../../data/
                    echo "Writing file to $DATA_FILE_3"
                else
                    echo "Loaded data set with $CLUSTER_SIZE clusters, $DIM3 dimensions and $STOP entries"
                fi


                echo "Running..."
                for N in $(seq $START $INTERVAL $STOP); do
                    ./bin/tsne_count_f.o $DATA_FILE_1 /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $DIM1 >> "$FILE_PREFIX@dim_$DIM1@float@iters.txt"
                    printf "."
                    ./bin/tsne_count_f.o $DATA_FILE_2 /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $DIM2 >> "$FILE_PREFIX@dim_$DIM2@float@iters.txt"
                    printf "."
                    ./bin/tsne_count_f.o $DATA_FILE_3 /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $DIM3 >> "$FILE_PREFIX@dim_$DIM3@float@iters.txt"
                    printf "."
                    ./bin/tsne_bench_f.o $DATA_FILE_1 /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $DIM1 >> "$FILE_PREFIX@dim_$DIM1@float@cycles.txt"
                    printf "."
                    ./bin/tsne_bench_f.o $DATA_FILE_2 /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $DIM2 >> "$FILE_PREFIX@dim_$DIM2@float@cycles.txt"
                    printf "."
                    ./bin/tsne_bench_f.o $DATA_FILE_3 /tmp/result.dat $N $PERPLEXITY $DIMS $MAX_ITER $DIM3 >> "$FILE_PREFIX@dim_$DIM3@float@cycles.txt"
                    printf " DONE\n"
                done;
                echo "plotting"
                python3 ./benchmarking/benchmarking_input_dimension.py $TODAY $COMPILER $COMPILER_FLAGS
                echo "Finished Successfully"
                ;;

        esac
        ;;
    *)
        ;;
esac
