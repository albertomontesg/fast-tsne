CC=g++
CFLAGS=-O3 -march=native
SRC=tsne_nlogn.cpp ../utils/io.c ./trees/sptree.cpp ./trees/vptree.h ./computations/normalize.c ./computations/compute_squared_euclidean_distance.c ./computations/compute_pairwise_affinity_perplexity_nlogn.c ./computations/symmetrize_affinities_nlogn.c ./computations/early_exageration.c ./computations/compute_low_dimensional_affinities.c ./computations/gradient_computation.c ./computations/gradient_update.c
BIN=tsne.o

all: tsne

tsne:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN)

debug:
	$(CC) -DDEBUG $(CFLAGS) $(SRC) -o ./$(BIN)
	./$(BIN) ../../../data/mnist/train-images.idx3-ubyte result.dat 1000 50 2 1000

numeric_check:
	$(CC) -DNUMERIC_CHECK $(CFLAGS) $(SRC) -o ./$(BIN)
	./$(BIN) ../../../data/mnist/train-images.idx3-ubyte result.dat 1000 50 2 1000

benchmark:
	$(CC) -DBENCHMARK $(CFLAGS) $(SRC) -o ./$(BIN)
	./$(BIN) ../../../data/mnist/train-images.idx3-ubyte result.dat 1000 50 2 1000

counting:
	$(CC) -DCOUNTING $(CFLAGS) $(SRC) -o ./$(BIN)
	./$(BIN) ../../../data/mnist/train-images.idx3-ubyte result.dat 1000 50 2 1000

run:
	./$(BIN) ../../../data/mnist/train-images.idx3-ubyte result.dat 1000 50 2 1000

clean:
	rm ./*.o
	rm ./computations/*.o
	rm ../utils/*.o
	rm ./datum/*
