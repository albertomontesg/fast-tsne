#ifndef COMPUTE_PAIRWISE_AFFINITY_PERPLEXITY
#define COMPUTE_PAIRWISE_AFFINITY_PERPLEXITY

#include <stdio.h>
#include "compute_squared_euclidean_distance.h"
#include <x86intrin.h>

float MAX_VAL = FLT_MAX;
float MIN_VAL = FLT_MIN;


// Compute pairwise affinity perplexity
inline void perplexity_blocking(float* X, int N, int D, float* P,
										  float perplexity, float* DD){

	#ifdef COUNTING
	int ITERS = 0;
	#endif

	compute_squared_euclidean_distance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {
		int found = 0;
		float beta = 1.0;
        float min_beta = -MAX_VAL;
        float max_beta =  MAX_VAL;
		float tol = 1e-5;
		float sum_P;

		__m256 beta_vec_neg = _mm256_set1_ps(-1*beta);
		__m256 beta_vec_pos = _mm256_set1_ps(beta);

		int iter = 0;
		while (found == 0 && iter < 20) { // iter < 200
			// Compute Gaussian kernel row
			// Compute entropy of current row
			float H = 0.0;
			sum_P = MIN_VAL; // = 0

			int m;
			for (m = 0; m + 8 < N; m+=8){
				__m256 DD_row = _mm256_load_ps(DD+nN+m);
				__m256 DD_row_beta = _mm256_mul_ps(beta_vec_neg,DD_row);
				__m256 DD_row_beta_exp = _mm256_exp_ps(DD_row_beta);
			//	__m256 DD_row_beta_pos = _mm256_mul_ps(beta_vec_pos,DD_row);
				_mm256_store_ps(P+nN+m,DD_row_beta_exp); // TODO: avoid storing back

				if((n >= m) || (n < m + 8)){ // diagonal case
					P[nN + n] = MIN_VAL; // = 0
				}

				__m256 P_row = _mm256_load_ps(P+nN+m);
				__m256 s = _mm256_hadd_ps(P_row,P_row);
				sum_P += s[0] + s[1] + s[4] + s[5];

				__m256 DD_row_beta_P = _mm256_mul_ps(beta_vec_pos,P_row);
				DD_row_beta_P = _mm256_mul_ps(DD_row_beta_P,P_row);

				__m256 s1 = _mm256_hadd_ps(DD_row_beta_P,DD_row_beta_P);
				H += s1[0] + s1[1] + s1[4] + s1[5];
			}


			//if N is not multiplicative factor of 8 do the rest sequentially
			for (int i = m; i < N; ++i){
				P[nN + i] = exp_c(-beta * DD[nN + i]);
				P[nN + n] = MIN_VAL;
				sum_P += P[nN + i];
				H += beta * (DD[nN + i] * P[nN + i]);
			}

//		    printf("blocking: \n");
//		    printf("H: %f \n", H);
//	        for(int i=0; i<N; i++){
//	            printf("P[%d][%d]= %f \n", n, i, P[n*N+i]);
//	        }
            float t1 = log_c(sum_P);
            float t2 = (H / sum_P);
			H = t1 + t2;
			// Evaluate whether the entropy is within the tolerance level
			float Hdiff = H - log_c(perplexity);
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
		__m256 sum_P_vec = _mm256_set1_ps(sum_P);
		int k;
		for(k = 0; k+8 < N; k+=8){
			__m256 P_row = _mm256_load_ps(P + nN+k);
			__m256 P_row_norm = _mm256_div_ps(P_row,sum_P_vec); // TODO rcp
			_mm256_store_ps(P+nN+k,P_row_norm);
		}

		//if N is not multiplicative factor of 8 do the rest sequentially
		float sum_P_inv = 1.0/sum_P; // TODO rcp
		for (int i = k; i < N; ++i)
		{
			P[nN + i] *= sum_P_inv;
		}

		nN += N;
	}

	#ifdef COUNTING
	printf("%d\n", ITERS);
	#endif

}

inline void base_version(float* X, int N, int D, float* P,
										  float perplexity, float* DD){

	#ifdef COUNTING
	int ITERS = 0;
	#endif

	compute_squared_euclidean_distance(X, N, D, DD);

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
		while (found == 0 && iter < 20) {
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp_c(-beta * DD[nN + m]);
			P[nN + n] = MIN_VAL;

			// Compute entropy of current row
			sum_P = MIN_VAL;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			float H = 0.0;
			for(int m = 0; m < N; m++) H += beta * DD[nN + m] * P[nN + m];

//		    	    printf("reference: \n");
//		    	    printf("H: %f \n", H);
//		            for(int i=0; i<N; i++){
//		                printf("P[%d][%d]= %f \n", n, i, P[n*N+i]);
//		            }
			H = (H / sum_P) + log_c(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			float Hdiff = H - log_c(perplexity);
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
	printf("%d\n", ITERS);
	#endif

}

#endif
