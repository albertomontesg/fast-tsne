ifeq ($(OS),Darwin)
    CC=g++-4.9
else
    CC=g++
endif

CFLAGS=-O3 -march=native
SRC=single_run_normalize.cpp
BIN_DOUBLE=tsned.o
BIN_FLOAT=tsnef.o

all: tsne_f

tsne_f:
	$(CC) -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)

tsne_d:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)

debug:
	$(CC) -DDEBUG $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	$(CC) -DDEBUG -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000
	./$(BIN_FLOAT) ../../../data/mnist/train-images.idx3-ubyte result_f.dat 1000 50 2 1000

asm:
	$(CC) $(CFLAGS) -S -fverbose-asm $(SRC) -o ./tsne_d.s
	$(CC) -DSINGLE_PRECISION $(CFLAGS) -S -fverbose-asm $(SRC) -o ./tsne_f.s

store_calculation_f:
	$(CC) -DSTORE_CALCULATION -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_FLOAT) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

store_calculation_d:
	$(CC) -DSTORE_CALCULATION $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

validate_f:
	$(CC) -VALIDATION -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_FLOAT) ../../../data/mnist/train-images.idx3-ubyte result_f.dat 1000 50 2 1000

validate_d:
	$(CC) -VALIDATION $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_f.dat 1000 50 2 1000

run_f:
	./$(BIN_FLOAT) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

run_d:
	./$(BIN_DOUBLE) ../../../data/mnist/train-images.idx3-ubyte result_d.dat 1000 50 2 1000

clean:
	rm *.o
	rm *.dat
	rm *.s
