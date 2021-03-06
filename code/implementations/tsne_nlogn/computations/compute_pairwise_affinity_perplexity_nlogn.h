#ifndef COMPUTE_PAIRWISE_AFFINITY_PERPLEXITY_NLOGN_H
#define COMPUTE_PAIRWISE_AFFINITY_PERPLEXITY_NLOGN_H

#include "../../utils/data_type.h"
#include <stdio.h>
#include <vector>
#include "../trees/vptree.h"

using namespace std;

// Compute pairwise affinity perplexity
inline void compute_pairwise_affinity_perplexity_nlogn(dt* X, int N, int D, dt* val_P,
										  unsigned int* row_P, unsigned int* col_P,
										  dt perplexity, unsigned int K)
{
    #ifdef COUNTING
    int ITERS = 0;
    #endif
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);
    dt* cur_P = (dt*) malloc((N - 1) * sizeof(dt));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    row_P[0] = 0;
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + K;


    vector<DataPoint> indices;
    vector<dt> distances;
	for (int n = 0; n < N; n++) {
        indices.clear();
        distances.clear();
        tree->search(obj_X[n], K + 1, &indices, &distances);
        bool found = false;
        dt beta = 1.0;
        dt min_beta = -DBL_MAX;
        dt max_beta =  DBL_MAX;
        dt tol = 1e-5;
		dt sum_P;

		int iter = 0;
		while (found == false && iter < 200) {

            // Compute Gaussian kernel row
            for(int m = 0; m < K; m++) cur_P[m] = exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for(int m = 0; m < K; m++) sum_P += cur_P[m];
            dt H = .0;
            for(int m = 0; m < K; m++) H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            dt Hdiff = H - log(perplexity);
            if(Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if(Hdiff > 0) {
                    min_beta = beta;
                    if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
		}
        #ifdef COUNTING
        ITERS += iter;
        #endif

        // Row-normalize current row of P and store in matrix
        for(unsigned int m = 0; m < K; m++) cur_P[m] /= sum_P;
        for(unsigned int m = 0; m < K; m++)
        {
            col_P[row_P[n] + m] = (unsigned int) indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
	}
    #ifdef COUNTING
    printf("it %d\nit_ec %d\nit_buildFromPoints %d\n", ITERS, ITERS_eucledianDistance, ITERS_buildfromPoints);
    #endif
}

#endif