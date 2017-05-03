#include "comp.h"
#include "../trees/sptree.h"
#include "stdio.h"
#include "stdlib.h"
// Gradient computation dC_dy

void gradient_computation(double* Y, unsigned int* row_P, unsigned int* col_P, double* val_P, int N,
						  int D, double* dC, double theta) {

	SPTree* tree = new SPTree(D, Y, N); // SPTree interface: (dimension, data, num_elements)

	// Perform the computation of the gradient
	double sum_Q = 0.0;
	double* pos_f = (double*) calloc(N * D, sizeof(double));
	double* neg_f = (double*) calloc(N * D, sizeof(double));
	if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	tree->computeEdgeForces(row_P, col_P, val_P, N, pos_f);
	for(unsigned int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);
//    void computeNonEdgeForces(unsigned int point_index, double theta, double neg_f[], double* sum_Q);

	// Compute final t-SNE gradient
	for(int i = 0; i < N * D; i++) {
	    dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}

