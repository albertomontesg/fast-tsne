CC=g++-4.9
CFLAGS=-O3 -march=native -mavx -std=c++11
SRC=main.cpp
BIN=run.o
OUT_FILE?=/dev/stdout;
BENCH_MEDIAN?=-DMEDIAN

all: benchmark

compile:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN)

benchmark:
	$(CC) $(CFLAGS) -S -fverbose-asm  $(SRC) -o ./run.s
	$(CC) $(CFLAGS) $(BENCH_MEDIAN) $(SRC) -o ./$(BIN)
	./$(BIN) > $(OUT_FILE)

asm:
	$(CC) $(CFLAGS) -S -fverbose-asm  $(SRC) -o ./run.s

clean:
	if [ -f $(BIN) ] ; then rm $(BIN) ; fi;
