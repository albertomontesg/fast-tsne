#ifndef COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H
#define COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H

#include <stdio.h>
#include <immintrin.h>

// Compute low dimensional affinities
inline float blocking_32_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;

	__m128 ones = _mm_set1_ps(1.);
	__m128 zeros = _mm_set1_ps(0.);

	// Loops for block size 32
	for (int iB = 0; iB < N; iB += B) {
		for (int jB = iB; jB < N; jB += B) {

			// Loops for microblocks of size 4
			for (int iK = iB; iK < iB + B; iK += K) {
				for (int jK = jB; jK < jB + B; jK += K) {
					// In case the result DD and symetric are the same
					if (iK == jK) {
						// Case where the resulting block of BxB needs only to
						// be computed the upper triangle part
                        const float* XnD = X + iK * D;
						const float* XmD = X + jK * D;
						float* DDij = DD + iK * N + jK;

						__m256 Xn_0 = _mm256_set_ps(XnD[1], XnD[0], XnD[1], XnD[0], XnD[1], XnD[0], XnD[1], XnD[0]);
						__m256 Xn_1 = _mm256_set_ps(XnD[3], XnD[2], XnD[3], XnD[2], XnD[3], XnD[2], XnD[3], XnD[2]);
						__m256 Xn_2 = _mm256_set_ps(XnD[5], XnD[4], XnD[5], XnD[4], XnD[5], XnD[4], XnD[5], XnD[4]);
						__m256 Xn_3 = _mm256_set_ps(XnD[7], XnD[6], XnD[7], XnD[6], XnD[7], XnD[6], XnD[7], XnD[6]);

						__m256 Xm = _mm256_loadu_ps(XmD);

						__m256 diff_0 = _mm256_sub_ps(Xm, Xn_0);
						__m256 diff_1 = _mm256_sub_ps(Xm, Xn_1);
						__m256 diff_2 = _mm256_sub_ps(Xm, Xn_2);
						__m256 diff_3 = _mm256_sub_ps(Xm, Xn_3);

						__m256 diff_sq_0 = _mm256_mul_ps(diff_0, diff_0);
						__m256 diff_sq_1 = _mm256_mul_ps(diff_1, diff_1);
						__m256 diff_sq_2 = _mm256_mul_ps(diff_2, diff_2);
						__m256 diff_sq_3 = _mm256_mul_ps(diff_3, diff_3);

						__m256 diff_sq_shuf_0 = _mm256_shuffle_ps(diff_sq_0, diff_sq_0, 177);
						__m256 diff_sq_shuf_1 = _mm256_shuffle_ps(diff_sq_1, diff_sq_1, 177);
						__m256 diff_sq_shuf_2 = _mm256_shuffle_ps(diff_sq_2, diff_sq_2, 177);
						__m256 diff_sq_shuf_3 = _mm256_shuffle_ps(diff_sq_3, diff_sq_3, 177);

						__m256 norm_0x = _mm256_add_ps(diff_sq_0, diff_sq_shuf_0);
						__m256 norm_1x = _mm256_add_ps(diff_sq_1, diff_sq_shuf_1);
						__m256 norm_2x = _mm256_add_ps(diff_sq_2, diff_sq_shuf_2);
						__m256 norm_3x = _mm256_add_ps(diff_sq_3, diff_sq_shuf_3);

						__m128 norm_0_low = _mm256_castps256_ps128(norm_0x);
						__m128 norm_1_low = _mm256_castps256_ps128(norm_1x);
						__m128 norm_2_low = _mm256_castps256_ps128(norm_2x);
						__m128 norm_3_low = _mm256_castps256_ps128(norm_3x);
						__m128 norm_0_high = _mm256_extractf128_ps(norm_0x, 1);
						__m128 norm_1_high = _mm256_extractf128_ps(norm_1x, 1);
						__m128 norm_2_high = _mm256_extractf128_ps(norm_2x, 1);
						__m128 norm_3_high = _mm256_extractf128_ps(norm_3x, 1);

						__m128 norm_0 = _mm_shuffle_ps(norm_0_low, norm_0_high, 136);
						__m128 norm_1 = _mm_shuffle_ps(norm_1_low, norm_1_high, 136);
						__m128 norm_2 = _mm_shuffle_ps(norm_2_low, norm_2_high, 136);
						__m128 norm_3 = _mm_shuffle_ps(norm_3_low, norm_3_high, 136);

						__m128 n_0 = _mm_add_ps(norm_0, ones);
						__m128 n_1 = _mm_add_ps(norm_1, ones);
						__m128 n_2 = _mm_add_ps(norm_2, ones);
						__m128 n_3 = _mm_add_ps(norm_3, ones);

						__m128 qinv_0 = _mm_rcp_ps(n_0);
						__m128 qinv_1 = _mm_rcp_ps(n_1);
						__m128 qinv_2 = _mm_rcp_ps(n_2);
						__m128 qinv_3 = _mm_rcp_ps(n_3);

						__m128 q_0 = _mm_blend_ps(qinv_0, zeros, 1);
						__m128 q_1 = _mm_blend_ps(qinv_1, zeros, 2);
						__m128 q_2 = _mm_blend_ps(qinv_2, zeros, 4);
						__m128 q_3 = _mm_blend_ps(qinv_3, zeros, 8);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

					}
					else if (jK > iK) {
						// In this case, the block has to be computed all and the symmetric position is not inside the block.
						const float* XnD = X + iK * D;
						const float* XmD = X + jK * D;
						float* DDij = DD + iK * N + jK;
						float* DDji = DD + jK * N + iK;

						__m256 Xn_0 = _mm256_set_ps(XnD[1], XnD[0], XnD[1], XnD[0], XnD[1], XnD[0], XnD[1], XnD[0]);
						__m256 Xn_1 = _mm256_set_ps(XnD[3], XnD[2], XnD[3], XnD[2], XnD[3], XnD[2], XnD[3], XnD[2]);
						__m256 Xn_2 = _mm256_set_ps(XnD[5], XnD[4], XnD[5], XnD[4], XnD[5], XnD[4], XnD[5], XnD[4]);
						__m256 Xn_3 = _mm256_set_ps(XnD[7], XnD[6], XnD[7], XnD[6], XnD[7], XnD[6], XnD[7], XnD[6]);

						__m256 Xm = _mm256_loadu_ps(XmD);

						__m256 diff_0 = _mm256_sub_ps(Xm, Xn_0);
						__m256 diff_1 = _mm256_sub_ps(Xm, Xn_1);
						__m256 diff_2 = _mm256_sub_ps(Xm, Xn_2);
						__m256 diff_3 = _mm256_sub_ps(Xm, Xn_3);

						__m256 diff_sq_0 = _mm256_mul_ps(diff_0, diff_0);
						__m256 diff_sq_1 = _mm256_mul_ps(diff_1, diff_1);
						__m256 diff_sq_2 = _mm256_mul_ps(diff_2, diff_2);
						__m256 diff_sq_3 = _mm256_mul_ps(diff_3, diff_3);

						__m256 diff_sq_shuf_0 = _mm256_shuffle_ps(diff_sq_0, diff_sq_0, 177);
						__m256 diff_sq_shuf_1 = _mm256_shuffle_ps(diff_sq_1, diff_sq_1, 177);
						__m256 diff_sq_shuf_2 = _mm256_shuffle_ps(diff_sq_2, diff_sq_2, 177);
						__m256 diff_sq_shuf_3 = _mm256_shuffle_ps(diff_sq_3, diff_sq_3, 177);

						__m256 norm_0x = _mm256_add_ps(diff_sq_0, diff_sq_shuf_0);
						__m256 norm_1x = _mm256_add_ps(diff_sq_1, diff_sq_shuf_1);
						__m256 norm_2x = _mm256_add_ps(diff_sq_2, diff_sq_shuf_2);
						__m256 norm_3x = _mm256_add_ps(diff_sq_3, diff_sq_shuf_3);

						__m128 norm_0_low = _mm256_castps256_ps128(norm_0x);
						__m128 norm_1_low = _mm256_castps256_ps128(norm_1x);
						__m128 norm_2_low = _mm256_castps256_ps128(norm_2x);
						__m128 norm_3_low = _mm256_castps256_ps128(norm_3x);
						__m128 norm_0_high = _mm256_extractf128_ps(norm_0x, 1);
						__m128 norm_1_high = _mm256_extractf128_ps(norm_1x, 1);
						__m128 norm_2_high = _mm256_extractf128_ps(norm_2x, 1);
						__m128 norm_3_high = _mm256_extractf128_ps(norm_3x, 1);

						__m128 norm_0 = _mm_shuffle_ps(norm_0_low, norm_0_high, 136);
						__m128 norm_1 = _mm_shuffle_ps(norm_1_low, norm_1_high, 136);
						__m128 norm_2 = _mm_shuffle_ps(norm_2_low, norm_2_high, 136);
						__m128 norm_3 = _mm_shuffle_ps(norm_3_low, norm_3_high, 136);

						__m128 n_0 = _mm_add_ps(norm_0, ones);
						__m128 n_1 = _mm_add_ps(norm_1, ones);
						__m128 n_2 = _mm_add_ps(norm_2, ones);
						__m128 n_3 = _mm_add_ps(norm_3, ones);

						__m128 q_0 = _mm_div_ps(ones, n_0);
						__m128 q_1 = _mm_div_ps(ones, n_1);
						__m128 q_2 = _mm_div_ps(ones, n_2);
						__m128 q_3 = _mm_div_ps(ones, n_3);
						// __m128 q_0 = _mm_rcp_ps(n_0);
						// __m128 q_1 = _mm_rcp_ps(n_1);
						// __m128 q_2 = _mm_rcp_ps(n_2);
						// __m128 q_3 = _mm_rcp_ps(n_3);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

						_MM_TRANSPOSE4_PS(q_0, q_1, q_2, q_3);

						_mm_store_ps(DDji, q_0);
						_mm_store_ps(DDji+N, q_1);
						_mm_store_ps(DDji+2*N, q_2);
						_mm_store_ps(DDji+3*N, q_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop

	return 0.;
}


// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void compute_squared_euclidean_distance(float* X, int N, int D, float* DD) {
	const float* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const float* XmD = XnD + D;
        float* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
				float dist = (XnD[d] - XmD[d]);
                *curr_elem += dist * dist;
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}

inline float base_version(float* Y, int N, int no_dims, float* Q) {

	compute_squared_euclidean_distance(Y, N, no_dims, Q);

	float sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + Q[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	return sum_Q;
}

#endif
