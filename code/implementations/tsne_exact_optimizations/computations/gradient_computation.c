#include "comp.h"

// Gradient computation dC_dy
void gradient_computation(double* Y, double* P, double* Q, double sum_Q, int N,
						  int D, double* dC) {
	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		int mD = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for(int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
			}
			mD += D;
		}
		nN += N;
		nD += D;
	}
}
