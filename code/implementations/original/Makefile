CC=g++-6
CFLAGS=-O3 -march=native
BIN=bh_tsne.o

# Running variables
# START=100
# END=1500
# STEP=100
# NUMBERS=$(shell seq $(START) $(STEP) $(END))


all: tsne

tsne:
	$(CC) $(CFLAGS) sptree.cpp tsne.cpp ../utils/io.c -o ./$(BIN)

run:
	./$(BIN) ../../../data/mnist/train-images.idx3-ubyte result.dat 1000 0 50 2 1000

clean:
	rm ./$(BIN)
	rm ./datum/*
