#include "comp.h"

// Compute low dimensional affinities
double compute_low_dimensional_affinities(double* Y, int N, int no_dims,
										  double* Q, double* DD) {

	compute_squared_euclidean_distance(Y, N, no_dims, DD);

	double sum_Q = .0;
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
