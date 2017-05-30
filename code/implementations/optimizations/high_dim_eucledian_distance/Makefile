GCC=g++-4.9
ICC=icc
GCCFLAGS=-O3 -march=native -mavx2 -mfma -ffast-math -std=c++11 -DSINGLE_PRECISION
ICCFLAGS=-O3 -march=core-avx2 -std=c++11 -DSINGLE_PRECISION
SRC=main.cpp
BIN_NO_VEC_GCC = run_gcc
BIN_VEC_GCC = run_gcc_vec
BIN_NO_VEC_ICC = run_icc
BIN_VEC_ICC = run_icc_vec


all: compile benchmark

compile:
	$(GCC) $(GCCFLAGS)  $(SRC) -o ./$(BIN_VEC_GCC)
	$(ICC) $(ICCFLAGS) $(SRC) -o ./$(BIN_VEC_ICC)
	$(GCC) $(GCCFLAGS) -fno-tree-vectorize  $(SRC) -o ./$(BIN_NO_VEC_GCC)
	$(ICC) $(ICCFLAGS) -no-vec $(SRC) -o ./$(BIN_NO_VEC_ICC)

benchmark: compile
	./$(BIN) > $(OUT_FILE)

clean:
	if [ -f $(BIN) ] ; then rm $(BIN) ; fi;
