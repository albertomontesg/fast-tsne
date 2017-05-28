ifeq ($(OS),Darwin)
    CC=g++-4.9
else
    CC=g++
endif
CC=g++-4.9

CFLAGS=-O3 -march=native -DCOUNTING
SRC=tsne_exact_optimized.cpp
BIN_BASELINE=./bin/tsne_base.o
BIN_SCALAR=./bin/tsne_scalar.o
BIN_AVX=./bin/tsne_avx.o
DATASET=../../../data/mnist/train-images.idx3-ubyte

HEADER='N,pairwise_squared_euclidean_distance,pairwise_affinity_perplexity,symmetrize_affinities,low_dimensional_affinities,gradient_computation_update_normalize'

all: baseline scalar avx

baseline:
	$(CC) $(CFLAGS) -DBASELINE $(SRC) -o $(BIN_BASELINE)
	$(BIN_BASELINE) $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000

scalar:
	$(CC) $(CFLAGS) -DSCALAR $(SRC) -o $(BIN_SCALAR)
	$(BIN_SCALAR) $(DATASET) ./datum/result_scalar.dat 1000 50 2 1000

avx:
	$(CC) $(CFLAGS) -DAVX $(SRC) -o $(BIN_AVX)
	$(BIN_AVX) $(DATASET) ./datum/result_avx.dat 1000 50 2 1000

debug:
	$(CC) $(CFLAGS) -DBASELINE $(SRC) -o $(BIN_BASELINE)
	$(BIN_BASELINE) $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
