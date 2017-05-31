GCC=g++-4.9
ICC=icc
GCCFLAGS=-O3 -march=native -mavx2 -mfma -ffast-math -std=c++11 -DSINGLE_PRECISION
ICCFLAGS=-O3 -march=core-avx2 -std=c++11 -fp-model fast -DSINGLE_PRECISION
SRC=main.cpp
BIN_NO_VEC_GCC = run_gcc_novec
BIN_VEC_GCC = run_gcc_vec
BIN_NO_VEC_ICC = run_icc_novec
BIN_VEC_ICC = run_icc_vec
TRUE = 1
FALSE = 0
DO_SCALAR= $(TRUE)

all: compile benchmark

compile:
	{																							 \
	if [ "$(DO_SCALAR)" -eq "$(TRUE)" ]; then													 \
		$(GCC) $(GCCFLAGS) -DSCALAR -fno-tree-vectorize -mno-abm  $(SRC) -o ./$(BIN_NO_VEC_GCC)	;\
		$(ICC) $(ICCFLAGS) -DSCALAR -no-vec $(SRC) -o ./$(BIN_NO_VEC_ICC)						;\
	else																						 \
 		$(GCC) $(GCCFLAGS)  $(SRC) -o ./$(BIN_VEC_GCC)											;\
 		$(ICC) $(ICCFLAGS) $(SRC) -o ./$(BIN_VEC_ICC)											;\
	fi																							;\
	} 


benchmark: compile
	{										  \
	if [ "$(DO_SCALAR)" -eq "$(TRUE)" ]; then \
		./$(BIN_NO_VEC_GCC)					 ;\
		./$(BIN_NO_VEC_ICC)					 ;\
	else									  \
 		./$(BIN_VEC_GCC)					 ;\
 		./$(BIN_VEC_ICC)					 ;\
	fi										 ;\
	} 

clean:
	rm $(BIN_VEC_GCC)
	rm $(BIN_NO_VEC_GCC)
	rm $(BIN_VEC_ICC)
	rm $(BIN_NO_VEC_ICC)
