#include "comp.h"

// Normalize X substracting mean and
void normalize(double* X, int N, int D, double* mean, int max_value) {
	int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}

	if (max_value > 0) {
		// Normalize to the maximum absolute value
		double max_X = .0;
		for(int i = 0; i < N * D; i++) {
	        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
	    }
	    for(int i = 0; i < N * D; i++) X[i] /= max_X;
	}
}
