CC=g++-6
CFLAGS=-O3
BIN=tsne.o

# # Running variables
# START=100
# END=1500
# STEP=100
# NUMBERS=$(shell seq $(START) $(STEP) $(END))


all: tsne

tsne:
	$(CC) $(CFLAGS) naive_tsne.cpp ../utils/io.c -o ./$(BIN)

clean:
	rm ./$(BIN)
