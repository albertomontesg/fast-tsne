#ifndef COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H
#define COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H

#include "../../utils/data_type.h"
#include "compute_squared_euclidean_distance.h"

// Compute low dimensional affinities
inline dt compute_low_dimensional_affinities(dt* Y, int N, int no_dims,
										  dt* Q, dt* DD) {

	compute_squared_euclidean_distance(Y, N, no_dims, DD);

	dt sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	return sum_Q;
}

#endif
