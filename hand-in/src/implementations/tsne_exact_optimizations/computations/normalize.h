#ifndef NORMALIZE_H
#define NORMALIZE_H

#include "../../utils/data_type.h"

// Normalize X substracting mean and
inline void normalize(dt* X, int N, int D, dt* mean, int max_value) {
	int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (dt) N;
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
		dt max_X = .0;
		for(int i = 0; i < N * D; i++) {
	        if(fabs_c(X[i]) > max_X) max_X = fabs_c(X[i]);
	    }
	    for(int i = 0; i < N * D; i++) X[i] /= max_X;
	}
}

#endif
