CC=g++-6
CFLAGS=-O3 -march=native
SRC=tsne_exact.cpp ../utils/io.c
BIN=tsne.o

# # Running variables
# START=100
# END=1500
# STEP=100
# NUMBERS=$(shell seq $(START) $(STEP) $(END))


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
	rm ./$(BIN)
	rm ./datum/*
