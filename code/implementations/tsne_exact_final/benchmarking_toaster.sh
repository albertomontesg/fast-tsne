#!/bin/sh

make

HEADER='N,pairwise_squared_euclidean_distance,pairwise_affinity_perplexity,symmetrize_affinities,low_dimensional_affinities,gradient_computation_update_normalize,total'
DATASET=../../../data/mnist/train-images.idx3-ubyte
MAX_ITER=1000       # Maximum number of iterations
DIMS=2              # Output dim
PERPLEXITY=50       # Perplexity best value


START=500
STOP=500
INTERVAL=500

mkdir toaster_bench

echo "N,iters" > toaster_bench/baseline_gcc_novec_count
echo "N,iters" > toaster_bench/baseline_gcc_vec_count
echo "N,iters" > toaster_bench/baseline_icc_novec_count
echo "N,iters" > toaster_bench/baseline_icc_vec_count
echo "N,iters" > toaster_bench/scalar_gcc_novec_count
echo "N,iters" > toaster_bench/scalar_gcc_vec_count
echo "N,iters" > toaster_bench/scalar_icc_novec_count
echo "N,iters" > toaster_bench/scalar_icc_vec_count
echo "N,iters" > toaster_bench/avx_gcc_vec_count
echo "N,iters" > toaster_bench/avx_icc_vec_count

echo $HEADER > toaster_bench/baseline_gcc_novec_cycles
echo $HEADER > toaster_bench/baseline_gcc_vec_cycles
echo $HEADER > toaster_bench/baseline_icc_novec_cycles
echo $HEADER > toaster_bench/baseline_icc_vec_cycles
echo $HEADER > toaster_bench/scalar_gcc_novec_cycles
echo $HEADER > toaster_bench/scalar_gcc_vec_cycles
echo $HEADER > toaster_bench/scalar_icc_novec_cycles
echo $HEADER > toaster_bench/scalar_icc_vec_cycles
echo $HEADER > toaster_bench/avx_gcc_vec_cycles
echo $HEADER > toaster_bench/avx_icc_vec_cycles

for N in $(seq $START $INTERVAL $STOP); do
    printf "$N baseline";
    ./baseline_gcc_novec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_gcc_novec_count
    ./baseline_gcc_vec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_gcc_vec_count
    ./baseline_icc_novec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_icc_novec_count
    ./baseline_icc_vec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_icc_vec_count
    printf "."
    ./baseline_gcc_novec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_gcc_novec_cycles
    ./baseline_gcc_vec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_gcc_vec_cycles
    ./baseline_icc_novec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_icc_novec_cycles
    ./baseline_icc_vec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/baseline_icc_vec_cycles
    printf " DONE\n"
    printf "$N scalar";
    ./scalar_gcc_novec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_gcc_novec_count
    ./scalar_gcc_vec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_gcc_vec_count
    ./scalar_icc_novec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_icc_novec_count
    ./scalar_icc_vec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_icc_vec_count
    printf "."
    ./scalar_gcc_novec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_gcc_novec_cycles
    ./scalar_gcc_vec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_gcc_vec_cycles
    ./scalar_icc_novec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_icc_novec_cycles
    ./scalar_icc_vec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/scalar_icc_vec_cycles
    printf " DONE\n"
    printf "$N avx";
    ./avx_gcc_vec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/avx_gcc_vec_count
    ./avx_icc_vec_counting $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/avx_icc_vec_count
    printf "."
    ./avx_gcc_vec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/avx_gcc_vec_cycles
    ./avx_icc_vec_bench $DATASET /dev/null $N $PERPLEXITY $DIMS $MAX_ITER >> toaster_bench/avx_icc_vec_cycles
    printf " DONE\n"
done;
