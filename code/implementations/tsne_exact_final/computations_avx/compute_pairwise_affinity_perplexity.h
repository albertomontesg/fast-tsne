#ifndef COMPUTE_PAIRWISE_AFFINITY_PERPLEXITY
#define COMPUTE_PAIRWISE_AFFINITY_PERPLEXITY

#include <stdio.h>
#include <math.h>
#include <float.h>

float MAX_VAL = FLT_MAX;
float MIN_VAL = FLT_MIN;


// Compute pairwise affinity perplexity
inline void compute_pairwise_affinity_perplexity(float* X, int N, int D,
		float* P, float perplexity, float* DD){

	#ifdef COUNTING
	int ITERS = 0;
	#endif


	// Compute the Gaussian kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {
		int found = 0;
		float beta = 1.0;
        float min_beta = -MAX_VAL;
        float max_beta =  MAX_VAL;
		float tol = 1e-5;
		float sum_P;

		int iter = 0;
		while (found == 0 && iter < 200) {
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = MIN_VAL;

			// Compute entropy of current row
			sum_P = MIN_VAL;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			float H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			float Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = 1;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == MAX_VAL || max_beta == -MAX_VAL)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -MAX_VAL || min_beta == MAX_VAL)
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

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
		nN += N;
	}

	#ifdef COUNTING
    printf("%d,%d\n", N, ITERS);
	#endif

}

#endif
