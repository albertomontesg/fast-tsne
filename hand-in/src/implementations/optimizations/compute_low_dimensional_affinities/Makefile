CC=g++
CFLAGS=-O3 -march=native -mavx -mfma -std=c++11 -lstdc++
SRC=main.cpp
BIN=run.o
OUT_FILE?=/dev/stdout;
BENCH_MEDIAN?=-DMEDIAN

all: compile benchmark

compile:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN)

benchmark:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN)
	./$(BIN) > $(OUT_FILE)

asm:
	$(CC) $(CFLAGS) -S -fverbose-asm  $(SRC) -o ./run.s

clean:
	if [ -f $(BIN) ] ; then rm $(BIN) ; fi;
