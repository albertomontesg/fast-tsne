CC=g++-4.9
CFLAGS=-O3 -march=native -mavx -std=c++11
SRC=main.cpp
BIN=run.o
OUT_FILE?=/dev/stdout;

all: compile benchmark

compile:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN)

benchmark: compile
	./$(BIN) > $(OUT_FILE)

asm:
	$(CC) $(CFLAGS) -S -fverbose-asm  $(SRC) -o ./run.s

clean:
	if [ -f $(BIN) ] ; then rm $(BIN) ; fi;
