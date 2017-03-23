CC=g++-6
CFLAGS=-O3
BIN=bh_tsne

# Running variables
START=100
END=1500
STEP=100
NUMBERS=$(shell seq $(START) $(STEP) $(END))


all: tsne

tsne:
	$(CC) $(CFLAGS) sptree.cpp tsne.cpp ../utils/io.c -o ./$(BIN)
