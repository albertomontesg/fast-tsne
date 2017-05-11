CC=g++-4.9
CFLAGS=-O3 -march=native -std=c++11 -DSINGLE_PRECISION
SRC=main.cpp
BIN=run.o
OUT_FILE?=/dev/stdout;

all: compile benchmark

compile:
	$(CC) $(CFLAGS) $(SRC) -o ./$(BIN)

benchmark: compile
	./$(BIN) >> $(OUT_FILE)

clean:
	if [ -f $(BIN) ] ; then rm $(BIN) ; fi;
