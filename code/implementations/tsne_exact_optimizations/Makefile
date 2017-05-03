CC=g++-6
CFLAGS=-O3 -march=native
SRC=tsne_exact.cpp ../utils/io.c ./computations/normalize.c ./computations/compute_squared_euclidean_distance.c ./computations/compute_pairwise_affinity_perplexity.c ./computations/symmetrize_affinities.c ./computations/early_exageration.c ./computations/compute_low_dimensional_affinities.c ./computations/gradient_computation.c ./computations/gradient_update.c
BIN_DOUBLE=tsned.o
BIN_FLOAT=tsnef.o

all: tsne

tsne:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)

debug:
	$(CC) -DDEBUG $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	$(CC) -DDEBUG -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000
	./$(BIN_FLOAT) ../../../data/mnist/train-images.idx3-ubyte result_f.dat 1000 50 2 1000

numeric_check_d:
	$(CC) -DDEBUG -DNUMERIC_CHECK $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

numeric_check_f:
	$(CC) -DDEBUG -DNUMERIC_CHECK -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_FLOAT) ../../../data/mnist/train-images.idx3-ubyte result_f.dat 1000 50 2 1000

benchmark:
	$(CC) -DBENCHMARK $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

counting:
	$(CC) -DCOUNTING $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

run:
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

clean:
	rm ./*.o
	rm ./computations/*.o
	rm ../utils/*.o
	rm ./datum/*
