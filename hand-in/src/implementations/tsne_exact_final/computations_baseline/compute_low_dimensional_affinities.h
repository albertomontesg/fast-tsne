#ifndef COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H
#define COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H

#include <stdio.h>
#include <immintrin.h>

// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void compute_squared_euclidean_distance_ld(float* X, int N, int D, float* DD) {
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

inline float compute_low_dimensional_affinities(float* Y, int N, int no_dims, float* Q) {

	compute_squared_euclidean_distance_ld(Y, N, no_dims, Q);

	float sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
		for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + Q[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	return sum_Q;
}

#endif
