CC=g++-6
CFLAGS=-O3 -march=native
SRC=tsne_nlogn.cpp ./trees/sptree.cpp
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

asm:
	$(CC) $(CFLAGS) -S -fverbose-asm $(SRC) -o ./tsne_d.s
	$(CC) -DSINGLE_PRECISION $(CFLAGS) -S -fverbose-asm $(SRC) -o ./tsne_f.s

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
	rm *.o
	rm *.dat
	rm *.s
