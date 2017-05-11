#ifndef COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H
#define COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H

#include "../../utils/data_type.h"

// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void compute_squared_euclidean_distance(dt* X, int N, int D, dt* DD) {
	const dt* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const dt* XmD = XnD + D;
        dt* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        dt* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
				dt dist = (XnD[d] - XmD[d]);
                *curr_elem += dist * dist;
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

inline void base_version(dt* X, int N, int D, dt* DD) {
	const dt* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const dt* XmD = XnD + D;
        dt* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        dt* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
				dt dist = (XnD[d] - XmD[d]);
                *curr_elem += dist * dist;
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

#endif
