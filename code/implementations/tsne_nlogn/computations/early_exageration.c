#include "comp.h"
#include "stdio.h"

// Early exageration (Multiply all the values of P to the given value)
void early_exageration(double* P, int N, double scale) {
    for (int i = 0; i < N * N; i++) P[i] *= scale;
}

void early_exageration_sparse(double* val_P, int no_elements, double scale){
	for(int i=0; i<no_elements; i++) {
		val_P[i] *= scale;
	}
}
