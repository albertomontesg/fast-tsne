#ifndef COMPUTE_SQUARED_EUCLEDIAN_DISTANCE_HIGH_NORMALIZE_H
#define COMPUTE_SQUARED_EUCLEDIAN_DISTANCE_HIGH_NORMALIZE_H

#include <math.h>

// Normalize X substracting mean and
inline void normalize(float* X, int N, int D, float* mean) {
	int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (float) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}

	// Normalize to the maximum absolute value
	float max_X = .0;
	for(int i = 0; i < N * D; i++) {
        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
    }
    for(int i = 0; i < N * D; i++) X[i] /= max_X;
}

// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void compute_squared_euclidean_distance(float* X, int N, int D, float* DD) {
	const float* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const float* XmD = XnD + D;
        float* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
				float dist = (XnD[d] - XmD[d]);
                *curr_elem += dist * dist;
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

inline void compute_squared_euclidean_distance_high_normalize(float* X, int N, int D, float* DD, float* mean) {

    normalize(X, N, D, mean);
	compute_squared_euclidean_distance(X, N, D, DD);
}

#endif
