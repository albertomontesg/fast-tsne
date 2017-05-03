#include "comp.h"
#include "../trees/sptree.h"
// Gradient computation dC_dy
void gradient_computation(double* Y, double* row_P, double* col_P, double* val_p, int N,
						  int D, double* dC) {

	SPTree* tree = new SPTree(D, Y, N) // SPTree interface: (dimension, data, num_elements)

	// Perform the computation of the gradient
	double sum_Q = .0;
	double* pos_f = (double*) calloc(N * D, sizeof(double));
	double* neg_f = (double*) calloc(N * D, sizeof(double));
	if(pos_f == NULL || neg_f == NULL) { printf("Memory allocation failed!\n"); exit(1); }
	tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);
	for(int n = 0; n < N; n++) tree->computeNonEdgeForces(n, theta, neg_f + n * D, &sum_Q);

	// Compute final t-SNE gradient
	for(int i = 0; i < N * D; i++) {
	    dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
	}
	free(pos_f);
	free(neg_f);
	delete tree;
}

