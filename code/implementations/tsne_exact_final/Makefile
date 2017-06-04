ifeq ($(OS),Darwin)
    GCC=g++-4.9
    ICC=icc
else
    GCC=g++
    ICC=/opt/intel/bin/icc
endif
GCCFLAGS=-O3 -march=native -mavx2 -mfma -ffast-math -std=c++11 -DSINGLE_PRECISION
ICCFLAGS=-O3 -march=core-avx2 -std=c++11 -fp-model fast -DSINGLE_PRECISION
SRC=tsne_exact_optimized.cpp
DATASET=../../../data/mnist/train-images.idx3-ubyte

HEADER='N,pairwise_squared_euclidean_distance,pairwise_affinity_perplexity,symmetrize_affinities,low_dimensional_affinities,gradient_computation_update_normalize,team'

all: baseline scalar avx

baseline:
	$(GCC) $(GCCFLAGS) -fno-tree-vectorize -mno-abm -DBASELINE $(SRC) -o baseline_gcc_novec
	$(GCC) $(GCCFLAGS) -DBASELINE $(SRC) -o baseline_gcc_vec
	$(ICC) $(ICCFLAGS) -no-vec -DBASELINE $(SRC) -o baseline_icc_novec
	$(ICC) $(ICCFLAGS) -DBASELINE $(SRC) -o baseline_icc_vec
	$(GCC) $(GCCFLAGS) -DCOUNTING -fno-tree-vectorize -mno-abm -DBASELINE $(SRC) -o baseline_gcc_novec_counting
	$(GCC) $(GCCFLAGS) -DCOUNTING -DBASELINE $(SRC) -o baseline_gcc_vec_counting
	$(ICC) $(ICCFLAGS) -DCOUNTING -no-vec -DBASELINE $(SRC) -o baseline_icc_novec_counting
	$(ICC) $(ICCFLAGS) -DCOUNTING -DBASELINE $(SRC) -o baseline_icc_vec_counting
	$(GCC) $(GCCFLAGS) -DBENCHMARK -fno-tree-vectorize -mno-abm -DBASELINE $(SRC) -o baseline_gcc_novec_bench
	$(GCC) $(GCCFLAGS) -DBENCHMARK -DBASELINE $(SRC) -o baseline_gcc_vec_bench
	$(ICC) $(ICCFLAGS) -DBENCHMARK -no-vec -DBASELINE $(SRC) -o baseline_icc_novec_bench
	$(ICC) $(ICCFLAGS) -DBENCHMARK -DBASELINE $(SRC) -o baseline_icc_vec_bench
	./baseline_gcc_novec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./baseline_gcc_vec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./baseline_icc_novec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./baseline_icc_vec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000

scalar:
	$(GCC) $(GCCFLAGS) -fno-tree-vectorize -mno-abm -DSCALAR $(SRC) -o scalar_gcc_novec
	$(GCC) $(GCCFLAGS) -DSCALAR $(SRC) -o scalar_gcc_vec
	$(ICC) $(ICCFLAGS) -no-vec -DSCALAR $(SRC) -o scalar_icc_novec
	$(ICC) $(ICCFLAGS) -DSCALAR $(SRC) -o scalar_icc_vec
	$(GCC) $(GCCFLAGS) -DCOUNTING -fno-tree-vectorize -mno-abm -DSCALAR $(SRC) -o scalar_gcc_novec_counting
	$(GCC) $(GCCFLAGS) -DCOUNTING -DSCALAR $(SRC) -o scalar_gcc_vec_counting
	$(ICC) $(ICCFLAGS) -DCOUNTING -no-vec -DSCALAR $(SRC) -o scalar_icc_novec_counting
	$(ICC) $(ICCFLAGS) -DCOUNTING -DSCALAR $(SRC) -o scalar_icc_vec_counting
	$(GCC) $(GCCFLAGS) -DBENCHMARK -fno-tree-vectorize -mno-abm -DSCALAR $(SRC) -o scalar_gcc_novec_bench
	$(GCC) $(GCCFLAGS) -DBENCHMARK -DSCALAR $(SRC) -o scalar_gcc_vec_bench
	$(ICC) $(ICCFLAGS) -DBENCHMARK -no-vec -DSCALAR $(SRC) -o scalar_icc_novec_bench
	$(ICC) $(ICCFLAGS) -DBENCHMARK -DSCALAR $(SRC) -o scalar_icc_vec_bench
	./scalar_gcc_novec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./scalar_gcc_vec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./scalar_icc_novec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./scalar_icc_vec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000


avx:
	$(GCC) $(GCCFLAGS) -DAVX $(SRC) -o avx_gcc_vec
	$(ICC) $(ICCFLAGS) -DAVX $(SRC) -o avx_icc_vec
	$(GCC) $(GCCFLAGS) -DCOUNTING -DAVX $(SRC) -o avx_gcc_vec_counting
	$(ICC) $(ICCFLAGS) -DCOUNTING -DAVX $(SRC) -o avx_icc_vec_counting
	$(GCC) $(GCCFLAGS) -DBENCHMARK -DAVX $(SRC) -o avx_gcc_vec_bench
	$(ICC) $(ICCFLAGS) -DBENCHMARK -DAVX $(SRC) -o avx_icc_vec_bench
	./avx_gcc_vec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
	./avx_icc_vec $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000

debug:
	$(CC) $(CFLAGS) -DBASELINE $(SRC) -o $(BIN_BASELINE)
	$(BIN_BASELINE) $(DATASET) ./datum/result_baseline.dat 1000 50 2 1000
