ifeq ($(OS),Darwin)
    CC=g++-4.9
else
    CC=g++
endif

CFLAGS=-O3 -march=native
SRC=tsne_exact.cpp
BIN_DOUBLE=tsned.o
BIN_FLOAT=tsnef.o
ARGS=../../../data/mnist/train-images.idx3-ubyte 1000 50 2 1000 784

all: tsne_f

tsne_f:
	$(CC) -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)

tsne_d:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)

debug:
	$(CC) -DDEBUG $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	$(CC) -DDEBUG -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_DOUBLE) $(ARGS)
	./$(BIN_FLOAT) $(ARGS)

asm:
	$(CC) $(CFLAGS) -S -fverbose-asm $(SRC) -o ./tsne_d.s
	$(CC) -DSINGLE_PRECISION $(CFLAGS) -S -fverbose-asm $(SRC) -o ./tsne_f.s

store_calculation_f:
	$(CC) -DSTORE_CALCULATION -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_FLOAT) $(ARGS)

store_calculation_d:
	$(CC) -DSTORE_CALCULATION $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) $(ARGS)

validate_f:
	$(CC) -DVALIDATION -DSINGLE_PRECISION $(CFLAGS) $(SRC) -o ./$(BIN_FLOAT)
	./$(BIN_FLOAT) $(ARGS)

validate_d:
	$(CC) -DVALIDATION $(CFLAGS) $(SRC) -o ./$(BIN_DOUBLE)
	./$(BIN_DOUBLE) $(ARGS)

run_f:
	./$(BIN_FLOAT) $(ARGS)

run_d:
	./$(BIN_DOUBLE) $(ARGS)

clean:
	rm *.o
	rm -f calculated/*
	rm *.s
