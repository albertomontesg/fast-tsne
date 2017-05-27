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
		float tol = 1e-5;
		float sum_P;

		__m256 beta_vec = _mm256_set1_ps(1.0); // beta (init)
        __m256 beta_min_vec = _mm256_set1_ps(-MAX_VAL);
        __m256 beta_max_vec = _mm256_set1_ps(MAX_VAL);
        __m256 n_vec = _mm256_set1_ps(n);
        __m256 half = _mm256_set1_ps(0.5);
        __m256 two = _mm256_set1_ps(2);
        __m256 minus = _mm256_set1_ps(-1);
        __m256 eight = _mm256_set1_ps(8);

		int iter = 0;
        int maxIter = 3; // 200
		while (found == 0 && iter < maxIter) { // iter < 200 
			// Compute Gaussian kernel row
			// Compute entropy of current row
			float H = 0.0;
			sum_P = MIN_VAL; // = 0
            
            __m256 sum_P_accum = _mm256_setzero_ps();
            __m256 H_accum = _mm256_setzero_ps();
            __m256 m_vec = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
            int m;
            
            // first part until diagonal 
			for (m = 0; m + 8 <= N; m+=8){
                // DD[m:m+7]
				__m256 DD_row = _mm256_load_ps(DD+nN+m);
                // -beta 
                __m256 DD_row_beta = _mm256_mul_ps(beta_vec, minus);
                // -beta * DD[m:m+7]
                DD_row_beta = _mm256_mul_ps(DD_row_beta, DD_row);
				// P_vec = P[m:m+7] = exp(-beta * DD[m:m+7]
                __m256 P_vec = _mm256_exp_ps(DD_row_beta); 
                // n==m, set P to zero
                // create mask. 0xFFFFFF if inequal, otherwise 0x000000
                __m256 mask = _mm256_cmp_ps(m_vec, n_vec, 4);
                __m256 min_val_vec = _mm256_set1_ps(MIN_VAL);
                mask = _mm256_add_ps(mask, min_val_vec);// set to min_val not to 0

                P_vec = _mm256_and_ps(P_vec, mask);
                // add P to sum_P accumulator
                sum_P_accum = _mm256_add_ps(sum_P_accum, P_vec);
                // H_now = beta * DD * P
                __m256 H_now = _mm256_mul_ps(beta_vec,DD_row);
				H_now = _mm256_mul_ps(H_now,P_vec);
                 // H += H_now
                H_accum = _mm256_add_ps(H_accum, H_now);

                // m+=8
                m_vec = _mm256_add_ps(m_vec, eight);
			}
            
			//if N is not multiplicative factor of 8 do the rest sequentially
            float beta = beta_vec[0];
			for (int i = m; i < N; ++i){
				float Pni = exp_c(-beta * DD[nN + i]);
				if(i==n) Pni = MIN_VAL;
				sum_P += Pni;
				H += beta * (DD[nN + i] * Pni);
			}
            
            // H += Hs[m:m+7]
            H_accum = _mm256_hadd_ps(H_accum,H_accum);
            H += H_accum[0] + H_accum[1] + H_accum[4] + H_accum[5];

            // sum_P += P[m:m+7]
			sum_P_accum = _mm256_hadd_ps(sum_P_accum,sum_P_accum);
			sum_P += sum_P_accum[0] + sum_P_accum[1] + sum_P_accum[4] + sum_P_accum[5];
	
//            printf("H before %.10e \n", H);
//            printf("sum_P %.10e \n", sum_P);
             
            // revert diagonal term added
            //printf("diagonal values\n P[nN+n]= %.30f \t H_now = %.30f \n", P[nN+n], (beta*DD[nN+n]*P[nN+n]));
//            float Pdiag = exp_c(-beta * DD[nN + n]);
//            printf("sum_P before substracting: %.30f \n", sum_P);
//            sum_P -= Pdiag;
//            H -= beta * DD[nN + n] * Pdiag;
            
			H = (H / sum_P) + log_c(sum_P);
            
//            printf("H after %.10e \n", H);
            // Evaluate whether the entropy is within the tolerance level
			float Hdiff = H - log_c(perplexity);
			if((iter+1 == maxIter) || (Hdiff < tol && -Hdiff < tol)) {
				found = 1;
			}
			else {
				if(Hdiff > 0) {
                    beta_min_vec = beta_vec;
					if(beta_max_vec[0] == MAX_VAL || beta_max_vec[0] == -MAX_VAL){
                        beta_vec = _mm256_mul_ps(two, beta_vec);
                    } else{
                        beta_vec = _mm256_add_ps(beta_vec, beta_max_vec);
                        beta_vec = _mm256_mul_ps(half, beta_vec);
                    }

				}
				else {
                    beta_max_vec = beta_vec;
					if(beta_min_vec[0] == -MAX_VAL || beta_min_vec[0] == MAX_VAL){
                        beta_vec = _mm256_mul_ps(half, beta_vec);
                    } else{
                        beta_vec = _mm256_add_ps(beta_vec, beta_min_vec);
                        beta_vec = _mm256_mul_ps(half, beta_vec);
                    }
				}
			}

			// Update iteration counter
			iter++;
//            printf("beta: %f \n", beta_vec[0]);
//            printf("iter %d, row %d \t H = %.30f \n", iter,n, H);
//            printf("sum_P: %.30f \n", sum_P);
//            printf("===\n");

		}
		#ifdef COUNTING
		ITERS += iter;
		#endif
        
        // based on beta write values
        float beta = beta_vec[0];
        for(int i=0; i<N; i++){
            P[nN+i] = exp_c(-1 * beta * DD[nN + i]);
            if(n==i) P[nN+i] = MIN_VAL;
        }
//        printf("sum_P before normalize: %.5e \n", sum_P);
//        for(int i=0; i<N; i++){
//            printf("P[%d][%d] before normalize: %.5e \n", n, i, P[nN+i]);
//        }
        
        
		// Row normalize P
		__m256 sum_P_inv_vec = _mm256_set1_ps(1.0/sum_P);
		int k;
		for(k = 0; k+8 <= N; k+=8){
			__m256 P_row = _mm256_load_ps(P + nN+k);
//            printf("1st and 2nd vals P: %.5e %.5e \n", P_row[0], P_row[1]);
//            printf("1st and 2nd vals 1/sum_p: %.5e %.5e \n", sum_P_inv_vec[0], sum_P_inv_vec[1]);
			__m256 P_row_norm = _mm256_mul_ps(P_row,sum_P_inv_vec); 
//            printf("1st and 2nd vals: %.5e %.5e \n", P_row_norm[0], P_row_norm[1]);
//            exit(1);
			_mm256_store_ps(P+nN+k,P_row_norm);
		}

		//if N is not multiplicative factor of 8 do the rest sequentially
		float sum_P_inv = 1.0/sum_P; 
		for (int i = k; i < N; ++i) 
		{
			P[nN + i] *= sum_P_inv;
		}
        
//        for(int i=0; i<N; i++){
//            printf("P[%d][%d] after normalize: %.5e \n", n, i, P[nN+i]);
//        }

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
        
        int maxIter = 3; // 200
		while (found == 0 && iter < maxIter) {
//            printf("iter %d \n", iter);
//            printf("-beta: %.30f \n", -beta);
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp_c(-beta * DD[nN + m]);
			float Pdiag = P[nN+n];
            P[nN + n] = MIN_VAL;
            
//            for(int i=0; i<N; i++){
//                printf("(%d,%d) DD: %.30f \n", n, i, DD[nN+i]);
//                printf("(%d,%d) exp arg: %.30f \n", n,i,(-beta * DD[nN+i]));
//            }
			// Compute entropy of current row
			sum_P = MIN_VAL;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			float H = 0.0;
			for(int m = 0; m < N; m++) H += beta * DD[nN + m] * P[nN + m];

            
//            printf("iter %d, row %d \t H = %.30f \n", iter,n, H);
//            printf("P[nN+n]: %.30f \t sum_P: %.30f \n", Pdiag, sum_P);
//            printf("===\n");
//            for(int i=0; i<N; i++){
//                    printf("row[%d] P[%d]: %.30f \n", n, i, P[nN + i]);
//            }
//            for(int i=0; i<N; i++){
//                    printf("row[%d] DD[%d]: %.30f \n", n, i, DD[nN + i]);
//            }
//
//            for(int i=0; i<N; i++){
//                    printf("row[%d] H[%d]: %.30f \n", n, i, beta * DD[nN+i] * P[nN+i]);
//            }
//
//            printf("H before: %.30f \n", H);
//            printf("sum_P: %.30f \n", sum_P);
//            printf("H part 1: %.30f \t part 2: %.30f \n", (H/sum_P), log_c(sum_P));

//            printf("H before %.10e \n", H);
//            printf("sum_P %.10e \n", sum_P);
			H = (H / sum_P) + log_c(sum_P);

//            printf("H after %.10e \n", H);
//            printf("H after: %.30f \n", H);

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
//        printf("beta: %.30f \n", beta);
//        for(int i=0; i<N; i++){
//            printf("P[%d][%d]= %.30f \n", n, i, P[n*N+i]);
//        }
//        printf("=======\n");
			// Update iteration counter
			iter++;
//            printf("beta: %f \n", beta);
//            printf("iter %d, row %d \t H = %.30f \n", iter,n, H);
//            printf("sum_P: %.30f \n", sum_P);
//            printf("===\n");

		}
		#ifdef COUNTING
		ITERS += iter;
		#endif
         
//        printf("sum_P before normalize: %.5e \n", sum_P);
//        for(int i=0; i<N; i++){
//            printf("P[%d][%d] before normalize: %.5e \n", n, i, P[nN+i]);
//        }

		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;

//        for(int i=0; i<N; i++){
//            printf("P[%d][%d] after normalize: %.5e \n", n, i, P[nN+i]);
//        }

        
		nN += N;
	}

	#ifdef COUNTING
	printf("%d\n", ITERS);
	#endif

}

#endif
