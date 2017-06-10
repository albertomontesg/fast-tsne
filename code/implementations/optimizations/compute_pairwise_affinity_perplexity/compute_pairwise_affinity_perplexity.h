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

    __m256 half = _mm256_set1_ps(0.5);
    __m256 one = _mm256_set1_ps(1);
    __m256 two = _mm256_set1_ps(2);
    __m256 minus = _mm256_set1_ps(-1);
    __m256 eight = _mm256_set1_ps(8);

    float log_perplexity = log_c(perplexity);
    int nN_n1 = 0;
    int nN_n2 = N;
	for (int n = 0; n+1 < N; n+=2) { // TODO: N not divisible by 2
		int found_n1 = 0;
		int found_n2 = 0;
		float tol = 1e-5;
		float sum_P_n1;
        float sum_P_n2;

		__m256 beta_vec_n1 = one; // beta (init)
        __m256 beta_min_vec_n1 = _mm256_set1_ps(-MAX_VAL);
        __m256 beta_max_vec_n1 = _mm256_set1_ps(MAX_VAL);
        __m256 n_vec_n1 = _mm256_set1_ps(n);

		__m256 beta_vec_n2 = beta_vec_n1; // beta (init)
        __m256 beta_min_vec_n2 = beta_min_vec_n1;
        __m256 beta_max_vec_n2 = beta_max_vec_n2;
        __m256 n_vec_n2 = n_vec_n1;
        n_vec_n2 = _mm256_add_ps(n_vec_n2, one); // n+1

		int iter = 0;
        int maxIter = 200; // 200
		while ((found_n1 == 0 || found_n2 == 0) && iter < maxIter) { // iter < 200 
			// Compute Gaussian kernel row
			// Compute entropy of current row
			float H_n1 = 0.0;
			float H_n2 = 0.0;
            
			sum_P_n1 = MIN_VAL; // = 0
            __m256 sum_P_accum_n1 = _mm256_setzero_ps();
            __m256 H_accum_n1 = _mm256_setzero_ps();
            
			sum_P_n2 = MIN_VAL; // = 0
            __m256 sum_P_accum_n2 = sum_P_accum_n1;
            __m256 H_accum_n2 = H_accum_n1;
                
            int m;
            __m256 m_vec = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
            // inner loop unrolling
			for (m = 0; m + 8 <= N; m+=8){
                // DD[m:m+7]
				__m256 DD_row_n1 = _mm256_load_ps(DD+nN_n1+m);
				__m256 DD_row_n2 = _mm256_load_ps(DD+nN_n2+m);
                // -beta 
                __m256 DD_row_beta_n1 = _mm256_mul_ps(beta_vec_n1, minus);
                __m256 DD_row_beta_n2 = _mm256_mul_ps(beta_vec_n2, minus);
                // -beta * DD[m:m+7]
                DD_row_beta_n1 = _mm256_mul_ps(DD_row_beta_n1, DD_row_n1);
                DD_row_beta_n2 = _mm256_mul_ps(DD_row_beta_n2, DD_row_n2);
				// P_vec = P[m:m+7] = exp(-beta * DD[m:m+7]
                __m256 P_vec_n1 = _mm256_exp_ps(DD_row_beta_n1); 
                __m256 P_vec_n2 = _mm256_exp_ps(DD_row_beta_n2); 
                // n==m, set P to zero
                // create mask. 0xFFFFFF if inequal, otherwise 0x000000
                __m256 mask_n1 = _mm256_cmp_ps(m_vec, n_vec_n1, 4);
                __m256 mask_n2 = _mm256_cmp_ps(m_vec, n_vec_n2, 4);

                __m256 min_val_vec = _mm256_set1_ps(MIN_VAL);
                mask_n1 = _mm256_add_ps(mask_n1, min_val_vec);// set to min_val not to 0
                mask_n2 = _mm256_add_ps(mask_n2, min_val_vec);// set to min_val not to 0

                P_vec_n1 = _mm256_and_ps(P_vec_n1, mask_n1);
                P_vec_n2 = _mm256_and_ps(P_vec_n2, mask_n2);
                // add P to sum_P accumulator
                sum_P_accum_n1 = _mm256_add_ps(sum_P_accum_n1, P_vec_n1);
                sum_P_accum_n2 = _mm256_add_ps(sum_P_accum_n2, P_vec_n2);
                // H_now = beta * DD * P
                // beta * DD
                __m256 H_now_n1 = _mm256_mul_ps(beta_vec_n1,DD_row_n1);
                __m256 H_now_n2 = _mm256_mul_ps(beta_vec_n2,DD_row_n2);
                // fma: H += (beta * DD) * P
                H_accum_n1 = _mm256_fmadd_ps(H_now_n1, P_vec_n1, H_accum_n1);
                H_accum_n2 = _mm256_fmadd_ps(H_now_n2, P_vec_n2, H_accum_n2);

                // m+=8
                m_vec = _mm256_add_ps(m_vec, eight);
			}
            
			//if N is not multiplicative factor of 8 do the rest sequentially
            float beta_n1 = beta_vec_n1[0];
            float beta_n2 = beta_vec_n2[0];
			for (int i = m; i < N; ++i){
				float Pni_n1 = exp_c(-beta_n1 * DD[nN_n1 + i]);
				float Pni_n2 = exp_c(-beta_n2 * DD[nN_n2 + i]);
				if(i==n) Pni_n1 = MIN_VAL;
				if(i==n+1) Pni_n2 = MIN_VAL;
				sum_P_n1 += Pni_n1;
				sum_P_n2 += Pni_n2;
				H_n1 += beta_n1 * (DD[nN_n1 + i] * Pni_n1);
				H_n2 += beta_n2 * (DD[nN_n2 + i] * Pni_n2);
			}
            
            // H += Hs[m:m+7]
            H_accum_n1 = _mm256_hadd_ps(H_accum_n1,H_accum_n1);
            H_accum_n2 = _mm256_hadd_ps(H_accum_n2,H_accum_n2);
            H_n1 += H_accum_n1[0] + H_accum_n1[1] + H_accum_n1[4] + H_accum_n1[5];
            H_n2 += H_accum_n2[0] + H_accum_n2[1] + H_accum_n2[4] + H_accum_n2[5];

            // sum_P += P[m:m+7]
			sum_P_accum_n1 = _mm256_hadd_ps(sum_P_accum_n1,sum_P_accum_n1);
			sum_P_accum_n2 = _mm256_hadd_ps(sum_P_accum_n2,sum_P_accum_n2);
			sum_P_n1 += sum_P_accum_n1[0] + sum_P_accum_n1[1] + sum_P_accum_n1[4] + sum_P_accum_n1[5];
			sum_P_n2 += sum_P_accum_n2[0] + sum_P_accum_n2[1] + sum_P_accum_n2[4] + sum_P_accum_n2[5];
	
			H_n1 = (H_n1 / sum_P_n1) + log_c(sum_P_n1);
			H_n2 = (H_n2 / sum_P_n2) + log_c(sum_P_n2);
            
            // Evaluate whether the entropy is within the tolerance level
			float Hdiff_n1 = H_n1 - log_perplexity;
			float Hdiff_n2 = H_n2 - log_perplexity;

            bool finished_n1 = (Hdiff_n1 < tol && -Hdiff_n1 < tol);
            bool finished_n2 = (Hdiff_n2 < tol && -Hdiff_n2 < tol);
			if((iter+1 == maxIter) || 
                (finished_n1 && finished_n2)){     
				found_n1 = 1;
                found_n2 = 1;
			}
            else{
                if(!finished_n1){
                    if(Hdiff_n1 > 0) {
                        beta_min_vec_n1 = beta_vec_n1;
    					if(beta_max_vec_n1[0] == MAX_VAL || beta_max_vec_n1[0] == -MAX_VAL){
                            beta_vec_n1 = _mm256_add_ps(beta_vec_n1, beta_vec_n1);
                        } else{
                            beta_vec_n1 = _mm256_add_ps(beta_vec_n1, beta_max_vec_n1);
                            beta_vec_n1 = _mm256_mul_ps(half, beta_vec_n1);
                        }
    				}
    				else {
                        beta_max_vec_n1 = beta_vec_n1;
    					if(beta_min_vec_n1[0] == -MAX_VAL || beta_min_vec_n1[0] == MAX_VAL){
                            beta_vec_n1 = _mm256_mul_ps(half, beta_vec_n1);
                        } else{
                            beta_vec_n1 = _mm256_add_ps(beta_vec_n1, beta_min_vec_n1);
                            beta_vec_n1 = _mm256_mul_ps(half, beta_vec_n1);
                        }
    				}
                }
                if(!finished_n2){
                    if(Hdiff_n2 > 0) {
                        beta_min_vec_n2 = beta_vec_n2;
    					if(beta_max_vec_n2[0] == MAX_VAL || beta_max_vec_n2[0] == -MAX_VAL){
                            beta_vec_n2 = _mm256_add_ps(beta_vec_n2, beta_vec_n2);
                        } else{
                            beta_vec_n2 = _mm256_add_ps(beta_vec_n2, beta_max_vec_n2);
                            beta_vec_n2 = _mm256_mul_ps(half, beta_vec_n2);
                        }
    				}
    				else {
                        beta_max_vec_n2 = beta_vec_n2;
    					if(beta_min_vec_n2[0] == -MAX_VAL || beta_min_vec_n2[0] == MAX_VAL){
                            beta_vec_n2 = _mm256_mul_ps(half, beta_vec_n2);
                        } else{
                            beta_vec_n2 = _mm256_add_ps(beta_vec_n2, beta_min_vec_n2);
                            beta_vec_n2 = _mm256_mul_ps(half, beta_vec_n2);
                        }
    				}
                }
            }
			
			// Update iteration counter
			iter++;
		}
		#ifdef COUNTING
		ITERS += iter;
		#endif

        // based on beta write values
        float beta_n1 = beta_vec_n1[0];
        float beta_n2 = beta_vec_n2[0];
        for(int i=0; i<N; i++){
            P[nN_n1+i] = exp_c(-1 * beta_n1 * DD[nN_n1 + i]);
            P[nN_n2+i] = exp_c(-1 * beta_n2 * DD[nN_n2 + i]);
            if(n==i) P[nN_n1+i] = MIN_VAL;
            if((n+1)==i) P[nN_n2+i] = MIN_VAL;
        }
        

               
		// Row normalize P
        __m256 sum_P_inv_vec_n1 = _mm256_set1_ps(sum_P_n1);
        __m256 sum_P_inv_vec_n2 = _mm256_set1_ps(sum_P_n2);
        sum_P_inv_vec_n1 = _mm256_rcp_ps(sum_P_inv_vec_n1);
        sum_P_inv_vec_n2 = _mm256_rcp_ps(sum_P_inv_vec_n2);
		int k;
		for(k = 0; k+8 <= N; k+=8){
			__m256 P_row_n1 = _mm256_load_ps(P + nN_n1+k);
			__m256 P_row_n2 = _mm256_load_ps(P + nN_n2+k);

			__m256 P_row_norm_n1 = _mm256_mul_ps(P_row_n1,sum_P_inv_vec_n1); 
			__m256 P_row_norm_n2 = _mm256_mul_ps(P_row_n2,sum_P_inv_vec_n2); 

			_mm256_store_ps(P+nN_n1+k,P_row_norm_n1);
			_mm256_store_ps(P+nN_n2+k,P_row_norm_n2);
		}

		//if N is not multiplicative factor of 8 do the rest sequentially
		float sum_P_inv_n1 = sum_P_inv_vec_n1[0];   // TODO: load from previous rcp calc?
		float sum_P_inv_n2 = sum_P_inv_vec_n2[0]; 
		for (int i = k; i < N; ++i) 
		{
			P[nN_n1 + i] *= sum_P_inv_n1;
			P[nN_n2 + i] *= sum_P_inv_n2;
		}
        
		nN_n1 += N+N;
		nN_n2 += N+N;
	}

	#ifdef COUNTING
	printf("%d\n", ITERS);
	#endif
}

// Compute pairwise affinity perplexity
inline void perplexity_unrolling(float* X, int N, int D, float* P,
										  float perplexity, float* DD){

	#ifdef COUNTING
	int ITERS = 0;
	#endif

	compute_squared_euclidean_distance(X, N, D, DD);

    __m256 half = _mm256_set1_ps(0.5);
    __m256 two = _mm256_set1_ps(2);
    __m256 minus = _mm256_set1_ps(-1);
    __m256 eight = _mm256_set1_ps(8);

    float log_c_perplexity = log_c(perplexity);
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


		int iter = 0;
        int maxIter = 200; // 200
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
	
			H = (H / sum_P) + log_c(sum_P);
            
            // Evaluate whether the entropy is within the tolerance level
			float Hdiff = H - log_c_perplexity;
			if((iter+1 == maxIter) || (Hdiff < tol && -Hdiff < tol)) {
				found = 1;
			}
			else {
				if(Hdiff > 0) {
                    beta_min_vec = beta_vec;
					if(beta_max_vec[0] == MAX_VAL || beta_max_vec[0] == -MAX_VAL){
                        beta_vec = _mm256_add_ps(beta_vec, beta_vec);
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
        
		// Row normalize P
        __m256 sum_P_inv_vec = _mm256_set1_ps(sum_P);
        sum_P_inv_vec = _mm256_rcp_ps(sum_P_inv_vec);
		int k;
		for(k = 0; k+8 <= N; k+=8){
			__m256 P_row = _mm256_load_ps(P + nN+k);
			__m256 P_row_norm = _mm256_mul_ps(P_row,sum_P_inv_vec); 
			_mm256_store_ps(P+nN+k,P_row_norm);
		}

		//if N is not multiplicative factor of 8 do the rest sequentially
		float sum_P_inv = 1.0/sum_P; 
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


inline void scalar_optimization(float* X, int N, int D, float* P,
										  float perplexity, float* DD){

	#ifdef COUNTING
	int ITERS = 0;
	#endif

	compute_squared_euclidean_distance(X, N, D, DD);
    float log_c_perplexity = log_c(perplexity); // common subexpression

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
        
        int maxIter = 200; // 200
		while (found == 0 && iter < maxIter) {
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp_c(-beta * DD[nN + m]);
			float Pdiag = P[nN+n];
            P[nN + n] = MIN_VAL;
            
			// Compute entropy of current row
			sum_P = MIN_VAL;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			float H = 0.0;
			for(int m = 0; m < N; m++) H += beta * DD[nN + m] * P[nN + m];

			H = (H / sum_P) + log_c(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			float Hdiff = H - log_c_perplexity;
                           
			if(Hdiff < tol && -Hdiff < tol) {
				found = 1;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == MAX_VAL || max_beta == -MAX_VAL)
						beta = beta + beta;
					else
						beta = (beta + max_beta) * 0.5;
				}
				else {
					max_beta = beta;
					if(min_beta == -MAX_VAL || min_beta == MAX_VAL)
						beta *= 0.5;
					else
						beta = (beta + min_beta) * 0.5;
				}
			}
			// Update iteration counter
			iter++;

		}
		#ifdef COUNTING
		ITERS += iter;
		#endif
		// Row normalize P
        float inv_sum_P = 1.0/sum_P;
		for(int m = 0; m < N; m++) P[nN + m] *= inv_sum_P;
        
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
        
        int maxIter = 200; // 200
		while (found == 0 && iter < maxIter) {
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp_c(-beta * DD[nN + m]);
			float Pdiag = P[nN+n];
            P[nN + n] = MIN_VAL;
            
			// Compute entropy of current row
			sum_P = MIN_VAL;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			float H = 0.0;
			for(int m = 0; m < N; m++) H += beta * DD[nN + m] * P[nN + m];

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
//        for(int i=0; i<N; i++){
//            if(n<3) printf("P[%d][%d]: %.5e \n", n, i, P[n*N+i]);
//        }
        
		nN += N;
	}

	#ifdef COUNTING
	printf("%d\n", ITERS);
	#endif

}

#endif
