#ifndef GRADIENT_COMPUTATION_NLOGN_H
#define GRADIENT_COMPUTATION_NLOGN_H

#include "../../utils/data_type.h"
#include "../trees/sptree.h"
#include "stdio.h"
#include "stdlib.h"
// Gradient computation dC_dy

inline void gradient_computation_nlogn(dt* Y, unsigned int* row_P, unsigned int* col_P, dt* val_P, int N,
						  int D, dt* dC, dt theta) {

	SPTree* tree = new SPTree(D, Y, N); // SPTree interface: (dimension, data, num_elements)

	// Perform the computation of the gradient
	dt sum_Q = 0.0;
	dt* pos_f = (dt*) calloc(N * D, sizeof(dt));
	dt* neg_f = (dt*) calloc(N * D, sizeof(dt));
	if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	tree->computeEdgeForces(row_P, col_P, val_P, N, pos_f);
	for(unsigned int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);
//    void computeNonEdgeForces(unsigned int point_index, dt theta, dt neg_f[], dt* sum_Q);

	// Compute final t-SNE gradient
	for(int i = 0; i < N * D; i++) {
	    dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);

	delete tree;
}

#endif
