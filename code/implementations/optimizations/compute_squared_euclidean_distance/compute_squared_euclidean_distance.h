#ifndef COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H
#define COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H

#include "../../utils/data_type.h"
#include <stdio.h>
#include <immintrin.h>

inline void blocking_32_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;


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

						_mm_store_ps(DDij, norm_0);
						_mm_store_ps(DDij+N, norm_1);
						_mm_store_ps(DDij+2*N, norm_2);
						_mm_store_ps(DDij+3*N, norm_3);

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

						_mm_store_ps(DDij, norm_0);
						_mm_store_ps(DDij+N, norm_1);
						_mm_store_ps(DDij+2*N, norm_2);
						_mm_store_ps(DDij+3*N, norm_3);

						_MM_TRANSPOSE4_PS(norm_0, norm_1, norm_2, norm_3);

						_mm_store_ps(DDji, norm_0);
						_mm_store_ps(DDji+N, norm_1);
						_mm_store_ps(DDji+2*N, norm_2);
						_mm_store_ps(DDji+3*N, norm_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop
}

inline void blocking_4_unfoold_sr(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 4; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;

	for (int i = 0; i < N; i += B) {
		for (int j = i; j < N; j += B) {

			if (i == j) {
				// Case where the resulting block of BxB needs only to be
				// computed the upper triangle part
				const float* XnD = X + i*D;
				for (int n = i; n < i+B; n++, XnD += D) {
					const float* XmD = XnD + D;
					float* curr_elem = &DD[n*N + n];
					*curr_elem = 0.0;
					float* curr_elem_sym = curr_elem + N;

					for (int m = n + 1; m < j + B; m++, XmD += D, curr_elem_sym += N) {
						float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*(++curr_elem) = sum;
						*curr_elem_sym = sum;
					}
				}
			}
			else {
				// In this case, the block has to be computed all and the symmetric position is not inside the block.
				const float* XnD = X + i*D;
				for (int n = i; n < i + B; n++, XnD += D) {
					const float* XmD = X + j*D;
					float* curr_elem = &DD[n*N + j];
                    float* curr_elem_sym = &DD[j*N + n];

					// Unfold the `m` and `d` loops
					float sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0;

					float Xn_1 = XnD[0], Xn_2 = XnD[1];
					float Xm_01 = XmD[0], Xm_02 = XmD[1];
					float Xm_11 = XmD[0+D], Xm_12 = XmD[1+D];
					float Xm_21 = XmD[0+2*D], Xm_22 = XmD[1+2*D];
					float Xm_31 = XmD[0+3*D], Xm_32 = XmD[1+3*D];

					float diff_01 = Xm_01 - Xn_1, diff_02 = Xm_02 - Xn_2;
					float diff_11 = Xm_11 - Xn_1, diff_12 = Xm_12 - Xn_2;
					float diff_21 = Xm_21 - Xn_1, diff_22 = Xm_22 - Xn_2;
					float diff_31 = Xm_31 - Xn_1, diff_32 = Xm_32 - Xn_2;
					sum_0 += diff_01 * diff_01;
					sum_1 += diff_11 * diff_11;
					sum_2 += diff_21 * diff_21;
					sum_3 += diff_31 * diff_31;
					sum_0 += diff_02 * diff_02;
					sum_1 += diff_12 * diff_12;
					sum_2 += diff_22 * diff_22;
					sum_3 += diff_32 * diff_32;

					*(curr_elem++) = sum_0;
					*(curr_elem++) = sum_1;
					*(curr_elem++) = sum_2;
					*(curr_elem++) = sum_3;

					*(curr_elem_sym) = sum_0;
					*(curr_elem_sym + 1 * N) = sum_1;
					*(curr_elem_sym + 2 * N) = sum_2;
					*(curr_elem_sym + 3 * N) = sum_3;
				}
			}

		}
	}
}

inline void blocking_32_block_4(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;
	const int K = 4;


	// Loops for block size 32
	for (int iB = 0; iB < N; iB += B) {
		for (int jB = iB; jB < N; jB += B) {

			// Loops for microblocks of size 4
			for (int iK = iB; iK < iB + B; iK += K) {
				for (int jK = jB; jK < jB + B; jK += K) {
					// In case the result DD and symetric are the same
					if (iK == jK) {
						// Case where the resulting block of BxB needs only to be
						// computed the upper triangle part
						const float* XnD = X + iK*D;
						for (int n = iK; n < iK + K; n++, XnD += D){
							const float* XmD = XnD + D;
							float* curr_elem = &DD[n*N + n];
							*curr_elem = 0.0;
							float* curr_elem_sym = curr_elem + N;
							for (int m = n + 1; m < jK + K; m++, XmD += D, curr_elem_sym += N) {
								float sum = 0.;
								for (int d = 0; d < D; d++) {
									float dif = XnD[d] - XmD[d];
									sum += dif * dif;
								}
								*(++curr_elem) = sum;
								*curr_elem_sym = sum;
							}
						}
					}
					else if (jK > iK) {
						// In this case, the block has to be computed all and the symmetric position is not inside the block.
						const float* XnD = X + iK*D;
						for (int n = iK; n < iK + K; n++, XnD += D) {
							const float* XmD = X + jK * D;
							float* curr_elem = &DD[n*N + jK];
		                    float* curr_elem_sym = &DD[jK*N + n];
							for (int m = jK; m < jK + K; m++, XmD += D, curr_elem_sym += N, curr_elem++) {
		                        float sum = 0.;
								for (int d = 0; d < D; d++) {
									float dif = XnD[d] - XmD[d];
									sum += dif * dif;
								}
								*curr_elem = sum;
								*curr_elem_sym = sum;
							}
						}
					}
				}
			} // End of K loop

		}
	} // End of B loop
}

inline void blocking_32_block_4_unfold_sr(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;
	const int K = 4;


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
						float* DDij = DD + iK * N + jK;

						// Unfold the `m` and `d` loops
						float sum_00 = 0, sum_01 = 0, sum_02 = 0, sum_03 = 0;
						float sum_11 = 0, sum_12 = 0, sum_13 = 0;
						float sum_22 = 0, sum_23 = 0;
						float sum_33 = 0;

						float Xn_01 = XnD[0], Xn_02 = XnD[1];
						float Xn_11 = XnD[D], Xn_12 = XnD[1+D];
						float Xn_21 = XnD[2*D], Xn_22 = XnD[1+2*D];
						float Xn_31 = XnD[3*D], Xn_32 = XnD[1+3*D];

						float diff_011 = Xn_01 - Xn_11, diff_012 = Xn_02 - Xn_12;
						float diff_021 = Xn_01 - Xn_21, diff_022 = Xn_02 - Xn_22;
						float diff_031 = Xn_01 - Xn_31, diff_032 = Xn_02 - Xn_32;

						float diff_121 = Xn_11 - Xn_21, diff_122 = Xn_12 - Xn_22;
						float diff_131 = Xn_11 - Xn_31, diff_132 = Xn_12 - Xn_32;

						float diff_231 = Xn_21 - Xn_31, diff_232 = Xn_22 - Xn_32;

						sum_01 += diff_011 * diff_011;
						sum_02 += diff_021 * diff_021;
						sum_03 += diff_031 * diff_031;
						sum_12 += diff_121 * diff_121;
						sum_13 += diff_131 * diff_131;
						sum_23 += diff_231 * diff_231;

						sum_01 += diff_012 * diff_012;
						sum_02 += diff_022 * diff_022;
						sum_03 += diff_032 * diff_032;
						sum_12 += diff_122 * diff_122;
						sum_13 += diff_132 * diff_132;
						sum_23 += diff_232 * diff_232;

						// Store the results on the matrix
						DDij[0] = sum_00;
						DDij[1] = sum_01;
						DDij[2] = sum_02;
						DDij[3] = sum_03;
						DDij[N] = sum_01;
						DDij[N+1] = sum_11;
						DDij[N+2] = sum_12;
						DDij[N+3] = sum_13;
						DDij[2*N] = sum_02;
						DDij[2*N+1] = sum_12;
						DDij[2*N+2] = sum_22;
						DDij[2*N+3] = sum_23;
						DDij[3*N] = sum_03;
						DDij[3*N+1] = sum_13;
						DDij[3*N+2] = sum_23;
						DDij[3*N+3] = sum_33;

					}
					else if (jK > iK) {
						// In this case, the block has to be computed all and the symmetric position is not inside the block.
						const float* XnD = X + iK * D;
						const float* XmD = X + jK * D;
						float* DDij = DD + iK * N + jK;
						float* DDji = DD + jK * N + iK;

						// Unfold the `m` and `d` loops
						float sum_00 = 0, sum_01 = 0, sum_02 = 0, sum_03 = 0;
						float sum_10 = 0, sum_11 = 0, sum_12 = 0, sum_13 = 0;
						float sum_20 = 0, sum_21 = 0, sum_22 = 0, sum_23 = 0;
						float sum_30 = 0, sum_31 = 0, sum_32 = 0, sum_33 = 0;

						float Xn_01 = XnD[0], Xn_02 = XnD[1];
						float Xn_11 = XnD[D], Xn_12 = XnD[1+D];
						float Xn_21 = XnD[2*D], Xn_22 = XnD[1+2*D];
						float Xn_31 = XnD[3*D], Xn_32 = XnD[1+3*D];

						float Xm_01 = XmD[0], Xm_02 = XmD[1];
						float Xm_11 = XmD[D], Xm_12 = XmD[1+D];
						float Xm_21 = XmD[2*D], Xm_22 = XmD[1+2*D];
						float Xm_31 = XmD[3*D], Xm_32 = XmD[1+3*D];

						float diff_001 = Xm_01 - Xn_01, diff_002 = Xm_02 - Xn_02;
						float diff_011 = Xm_11 - Xn_01, diff_012 = Xm_12 - Xn_02;
						float diff_021 = Xm_21 - Xn_01, diff_022 = Xm_22 - Xn_02;
						float diff_031 = Xm_31 - Xn_01, diff_032 = Xm_32 - Xn_02;

						float diff_101 = Xm_01 - Xn_11, diff_102 = Xm_02 - Xn_12;
						float diff_111 = Xm_11 - Xn_11, diff_112 = Xm_12 - Xn_12;
						float diff_121 = Xm_21 - Xn_11, diff_122 = Xm_22 - Xn_12;
						float diff_131 = Xm_31 - Xn_11, diff_132 = Xm_32 - Xn_12;


						float diff_201 = Xm_01 - Xn_21, diff_202 = Xm_02 - Xn_22;
						float diff_211 = Xm_11 - Xn_21, diff_212 = Xm_12 - Xn_22;
						float diff_221 = Xm_21 - Xn_21, diff_222 = Xm_22 - Xn_22;
						float diff_231 = Xm_31 - Xn_21, diff_232 = Xm_32 - Xn_22;


						float diff_301 = Xm_01 - Xn_31, diff_302 = Xm_02 - Xn_32;
						float diff_311 = Xm_11 - Xn_31, diff_312 = Xm_12 - Xn_32;
						float diff_321 = Xm_21 - Xn_31, diff_322 = Xm_22 - Xn_32;
						float diff_331 = Xm_31 - Xn_31, diff_332 = Xm_32 - Xn_32;


						sum_00 += diff_001 * diff_001;
						sum_01 += diff_011 * diff_011;
						sum_02 += diff_021 * diff_021;
						sum_03 += diff_031 * diff_031;
						sum_10 += diff_101 * diff_101;
						sum_11 += diff_111 * diff_111;
						sum_12 += diff_121 * diff_121;
						sum_13 += diff_131 * diff_131;
						sum_20 += diff_201 * diff_201;
						sum_21 += diff_211 * diff_211;
						sum_22 += diff_221 * diff_221;
						sum_23 += diff_231 * diff_231;
						sum_30 += diff_301 * diff_301;
						sum_31 += diff_311 * diff_311;
						sum_32 += diff_321 * diff_321;
						sum_33 += diff_331 * diff_331;

						sum_00 += diff_002 * diff_002;
						sum_01 += diff_012 * diff_012;
						sum_02 += diff_022 * diff_022;
						sum_03 += diff_032 * diff_032;
						sum_10 += diff_102 * diff_102;
						sum_11 += diff_112 * diff_112;
						sum_12 += diff_122 * diff_122;
						sum_13 += diff_132 * diff_132;
						sum_20 += diff_202 * diff_202;
						sum_21 += diff_212 * diff_212;
						sum_22 += diff_222 * diff_222;
						sum_23 += diff_232 * diff_232;
						sum_30 += diff_302 * diff_302;
						sum_31 += diff_312 * diff_312;
						sum_32 += diff_322 * diff_322;
						sum_33 += diff_332 * diff_332;

						// Store the results on the matrix
						DDij[0] = sum_00;
						DDij[1] = sum_01;
						DDij[2] = sum_02;
						DDij[3] = sum_03;
						DDij[N] = sum_10;
						DDij[N+1] = sum_11;
						DDij[N+2] = sum_12;
						DDij[N+3] = sum_13;
						DDij[2*N] = sum_20;
						DDij[2*N+1] = sum_21;
						DDij[2*N+2] = sum_22;
						DDij[2*N+3] = sum_23;
						DDij[3*N] = sum_30;
						DDij[3*N+1] = sum_31;
						DDij[3*N+2] = sum_32;
						DDij[3*N+3] = sum_33;

						// Store the results on the symmetric matrix
						DDji[0] = sum_00;
						DDji[1] = sum_10;
						DDji[2] = sum_20;
						DDji[3] = sum_30;
						DDji[N] = sum_01;
						DDji[N+1] = sum_11;
						DDji[N+2] = sum_21;
						DDji[N+3] = sum_31;
						DDji[2*N] = sum_02;
						DDji[2*N+1] = sum_12;
						DDji[2*N+2] = sum_22;
						DDji[2*N+3] = sum_32;
						DDji[3*N] = sum_03;
						DDji[3*N+1] = sum_13;
						DDji[3*N+2] = sum_23;
						DDji[3*N+3] = sum_33;

					}
				}
			} // End of K loop

		}
	} // End of B loop
}

inline void blocking_16_block_8_unfold_sr(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 16; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;
	const int K = 8;


	// Loops for block size 16
	for (int iB = 0; iB < N; iB += B) {
		for (int jB = iB; jB < N; jB += B) {

			// Loops for microblocks of size 8
			for (int iK = iB; iK < iB + B; iK += K) {
				for (int jK = jB; jK < jB + B; jK += K) {
					// In case the result DD and symetric are the same
					if (iK == jK) {
						const float* XnD = X + iK*D;
						for (int n = iK; n < iK+K; n++, XnD += D) {
							const float* XmD = XnD + D;
							float* curr_elem = &DD[n*N + n];
							*curr_elem = 0.0;
							float* curr_elem_sym = curr_elem + N;

							for (int m = n + 1; m < jK + K; m++, XmD += D, curr_elem_sym += N) {
								float sum = 0.;
								for (int d = 0; d < D; d++) {
									float dif = XnD[d] - XmD[d];
									sum += dif * dif;
								}
								*(++curr_elem) = sum;
								*curr_elem_sym = sum;
							}
						}

					}
					else if (jK > iK) {
						// In this case, the block has to be computed all and the symmetric position is not inside the block.
						const float* XnD = X + iK*D;
						for (int n = iK; n < iK + K; n++, XnD += D) {
							const float* XmD = X + jK*D;
							float* curr_elem = &DD[n*N + jK];
		                    float* curr_elem_sym = &DD[jK*N + n];

							// Unfold the `m` and `d` loops
							float sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0, sum_4 = 0, sum_5 = 0, sum_6 = 0, sum_7 = 0;

							float Xn_1 = XnD[0], Xn_2 = XnD[1];
							float Xm_01 = XmD[0], Xm_02 = XmD[1];
							float Xm_11 = XmD[D], Xm_12 = XmD[1+D];
							float Xm_21 = XmD[2*D], Xm_22 = XmD[1+2*D];
							float Xm_31 = XmD[3*D], Xm_32 = XmD[1+3*D];
							float Xm_41 = XmD[4*D], Xm_42 = XmD[1+4*D];
							float Xm_51 = XmD[5*D], Xm_52 = XmD[1+5*D];
							float Xm_61 = XmD[6*D], Xm_62 = XmD[1+6*D];
							float Xm_71 = XmD[7*D], Xm_72 = XmD[1+7*D];

							float diff_01 = Xm_01 - Xn_1, diff_02 = Xm_02 - Xn_2;
							float diff_11 = Xm_11 - Xn_1, diff_12 = Xm_12 - Xn_2;
							float diff_21 = Xm_21 - Xn_1, diff_22 = Xm_22 - Xn_2;
							float diff_31 = Xm_31 - Xn_1, diff_32 = Xm_32 - Xn_2;
							float diff_41 = Xm_41 - Xn_1, diff_42 = Xm_42 - Xn_2;
							float diff_51 = Xm_51 - Xn_1, diff_52 = Xm_52 - Xn_2;
							float diff_61 = Xm_61 - Xn_1, diff_62 = Xm_62 - Xn_2;
							float diff_71 = Xm_71 - Xn_1, diff_72 = Xm_72 - Xn_2;
							sum_0 += diff_01 * diff_01;
							sum_1 += diff_11 * diff_11;
							sum_2 += diff_21 * diff_21;
							sum_3 += diff_31 * diff_31;
							sum_4 += diff_41 * diff_41;
							sum_5 += diff_51 * diff_51;
							sum_6 += diff_61 * diff_61;
							sum_7 += diff_71 * diff_71;

							sum_0 += diff_02 * diff_02;
							sum_1 += diff_12 * diff_12;
							sum_2 += diff_22 * diff_22;
							sum_3 += diff_32 * diff_32;
							sum_4 += diff_42 * diff_42;
							sum_5 += diff_52 * diff_52;
							sum_6 += diff_62 * diff_62;
							sum_7 += diff_72 * diff_72;

							*(curr_elem++) = sum_0;
							*(curr_elem++) = sum_1;
							*(curr_elem++) = sum_2;
							*(curr_elem++) = sum_3;
							*(curr_elem++) = sum_4;
							*(curr_elem++) = sum_5;
							*(curr_elem++) = sum_6;
							*(curr_elem++) = sum_7;

							*(curr_elem_sym) = sum_0;
							*(curr_elem_sym + 1 * N) = sum_1;
							*(curr_elem_sym + 2 * N) = sum_2;
							*(curr_elem_sym + 3 * N) = sum_3;
							*(curr_elem_sym + 4 * N) = sum_4;
							*(curr_elem_sym + 5 * N) = sum_5;
							*(curr_elem_sym + 6 * N) = sum_6;
							*(curr_elem_sym + 7 * N) = sum_7;
						}
					}
				}
			} // End of K loop

		}
	} // End of B loop
}

inline void blocking_64_block_8_unfold_sr(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 64; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;
	const int K = 8;


	// Loops for block size 16
	for (int iB = 0; iB < N; iB += B) {
		for (int jB = iB; jB < N; jB += B) {

			// Loops for microblocks of size 8
			for (int iK = iB; iK < iB + B; iK += K) {
				for (int jK = jB; jK < jB + B; jK += K) {
					// In case the result DD and symetric are the same
					if (iK == jK) {
						const float* XnD = X + iK*D;
						for (int n = iK; n < iK+K; n++, XnD += D) {
							const float* XmD = XnD + D;
							float* curr_elem = &DD[n*N + n];
							*curr_elem = 0.0;
							float* curr_elem_sym = curr_elem + N;

							for (int m = n + 1; m < jK + K; m++, XmD += D, curr_elem_sym += N) {
								float sum = 0.;
								for (int d = 0; d < D; d++) {
									float dif = XnD[d] - XmD[d];
									sum += dif * dif;
								}
								*(++curr_elem) = sum;
								*curr_elem_sym = sum;
							}
						}

					}
					else if (jK > iK) {
						// In this case, the block has to be computed all and the symmetric position is not inside the block.
						const float* XnD = X + iK*D;
						for (int n = iK; n < iK + K; n++, XnD += D) {
							const float* XmD = X + jK*D;
							float* curr_elem = &DD[n*N + jK];
		                    float* curr_elem_sym = &DD[jK*N + n];

							// Unfold the `m` and `d` loops
							float sum_0 = 0, sum_1 = 0, sum_2 = 0, sum_3 = 0, sum_4 = 0, sum_5 = 0, sum_6 = 0, sum_7 = 0;

							float Xn_1 = XnD[0], Xn_2 = XnD[1];
							float Xm_01 = XmD[0], Xm_02 = XmD[1];
							float Xm_11 = XmD[D], Xm_12 = XmD[1+D];
							float Xm_21 = XmD[2*D], Xm_22 = XmD[1+2*D];
							float Xm_31 = XmD[3*D], Xm_32 = XmD[1+3*D];
							float Xm_41 = XmD[4*D], Xm_42 = XmD[1+4*D];
							float Xm_51 = XmD[5*D], Xm_52 = XmD[1+5*D];
							float Xm_61 = XmD[6*D], Xm_62 = XmD[1+6*D];
							float Xm_71 = XmD[7*D], Xm_72 = XmD[1+7*D];

							float diff_01 = Xm_01 - Xn_1, diff_02 = Xm_02 - Xn_2;
							float diff_11 = Xm_11 - Xn_1, diff_12 = Xm_12 - Xn_2;
							float diff_21 = Xm_21 - Xn_1, diff_22 = Xm_22 - Xn_2;
							float diff_31 = Xm_31 - Xn_1, diff_32 = Xm_32 - Xn_2;
							float diff_41 = Xm_41 - Xn_1, diff_42 = Xm_42 - Xn_2;
							float diff_51 = Xm_51 - Xn_1, diff_52 = Xm_52 - Xn_2;
							float diff_61 = Xm_61 - Xn_1, diff_62 = Xm_62 - Xn_2;
							float diff_71 = Xm_71 - Xn_1, diff_72 = Xm_72 - Xn_2;
							sum_0 += diff_01 * diff_01;
							sum_1 += diff_11 * diff_11;
							sum_2 += diff_21 * diff_21;
							sum_3 += diff_31 * diff_31;
							sum_4 += diff_41 * diff_41;
							sum_5 += diff_51 * diff_51;
							sum_6 += diff_61 * diff_61;
							sum_7 += diff_71 * diff_71;

							sum_0 += diff_02 * diff_02;
							sum_1 += diff_12 * diff_12;
							sum_2 += diff_22 * diff_22;
							sum_3 += diff_32 * diff_32;
							sum_4 += diff_42 * diff_42;
							sum_5 += diff_52 * diff_52;
							sum_6 += diff_62 * diff_62;
							sum_7 += diff_72 * diff_72;

							*(curr_elem++) = sum_0;
							*(curr_elem++) = sum_1;
							*(curr_elem++) = sum_2;
							*(curr_elem++) = sum_3;
							*(curr_elem++) = sum_4;
							*(curr_elem++) = sum_5;
							*(curr_elem++) = sum_6;
							*(curr_elem++) = sum_7;

							*(curr_elem_sym) = sum_0;
							*(curr_elem_sym + 1 * N) = sum_1;
							*(curr_elem_sym + 2 * N) = sum_2;
							*(curr_elem_sym + 3 * N) = sum_3;
							*(curr_elem_sym + 4 * N) = sum_4;
							*(curr_elem_sym + 5 * N) = sum_5;
							*(curr_elem_sym + 6 * N) = sum_6;
							*(curr_elem_sym + 7 * N) = sum_7;
						}
					}
				}
			} // End of K loop

		}
	} // End of B loop
}


inline void blocking_4(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 4; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;


	for (int i = 0; i < N; i += B) {
		for (int j = i; j < N; j += B) {
			// printf("%d\t%d\n", i, j);

			if (i == j) {
				// Case where the resulting block of BxB needs only to be
				// computed the upper triangle part
				const float* XnD = X + i*D;
				for (int n = i; n < i+B; n++, XnD += D){
					const float* XmD = XnD + D;
					float* curr_elem = &DD[n*N + n];
					*curr_elem = 0.0;

					float* curr_elem_sym = curr_elem + N;
					for (int m = n + 1; m < j + B; m++, XmD += D, curr_elem_sym += N) {
						float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*(++curr_elem) = sum;
						*curr_elem_sym = sum;
					}
				}
			} else {
				// In this case, the block has to be computed all and the symmetric position is not inside the block.
				const float* XnD = X + i*D;
				for (int n = i; n < i + B; n++, XnD += D) {
					const float* XmD = X + j*D;
					float* curr_elem = &DD[n*N + j];
                    float* curr_elem_sym = &DD[j*N + n];
					for (int m = j; m < j + B; m++, XmD += D, curr_elem_sym += N, curr_elem++) {
                        float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*curr_elem = sum;
						*curr_elem_sym = sum;
					}
				}
			}

		}
	}
}

inline void blocking_8(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 8; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;

	for (int i = 0; i < N; i += B) {
		for (int j = i; j < N; j += B) {

			if (i == j) {
				// Case where the resulting block of BxB needs only to be
				// computed the upper triangle part
				const float* XnD = X + i*D;
				for (int n = i; n < i+B; n++, XnD += D){
					const float* XmD = XnD + D;
					float* curr_elem = &DD[n*N + n];
					*curr_elem = 0.0;
					float* curr_elem_sym = curr_elem + N;
					for (int m = n + 1; m < j + B; m++, XmD += D, curr_elem_sym += N) {
						float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*(++curr_elem) = sum;
						*curr_elem_sym = sum;
					}
				}
			} else {
				// In this case, the block has to be computed all and the symmetric position is not inside the block.
				const float* XnD = X + i*D;
				for (int n = i; n < i + B; n++, XnD += D) {
					const float* XmD = X + j*D;
					float* curr_elem = &DD[n*N + j];
                    float* curr_elem_sym = &DD[j*N + n];
					for (int m = j; m < j + B; m++, XmD += D, curr_elem_sym += N, curr_elem++) {
                        float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*curr_elem = sum;
						*curr_elem_sym = sum;
					}
				}
			}

		}
	}
}

inline void blocking_16(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 16; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;

	for (int i = 0; i < N; i += B) {
		for (int j = i; j < N; j += B) {

			if (i == j) {
				// Case where the resulting block of BxB needs only to be
				// computed the upper triangle part
				const float* XnD = X + i*D;
				for (int n = i; n < i+B; n++, XnD += D){
					const float* XmD = XnD + D;
					float* curr_elem = &DD[n*N + n];
					*curr_elem = 0.0;
					float* curr_elem_sym = curr_elem + N;
					for (int m = n + 1; m < j + B; m++, XmD += D, curr_elem_sym += N) {
						float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*(++curr_elem) = sum;
						*curr_elem_sym = sum;
					}
				}
			} else {
				// In this case, the block has to be computed all and the symmetric position is not inside the block.
				const float* XnD = X + i*D;
				for (int n = i; n < i + B; n++, XnD += D) {
					const float* XmD = X + j*D;
					float* curr_elem = &DD[n*N + j];
                    float* curr_elem_sym = &DD[j*N + n];
					for (int m = j; m < j + B; m++, XmD += D, curr_elem_sym += N, curr_elem++) {
                        float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*curr_elem = sum;
						*curr_elem_sym = sum;
					}
				}
			}

		}
	}
}

inline void blocking_32(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;

	for (int i = 0; i < N; i += B) {
		for (int j = i; j < N; j += B) {

			if (i == j) {
				// Case where the resulting block of BxB needs only to be
				// computed the upper triangle part
				const float* XnD = X + i*D;
				for (int n = i; n < i+B; n++, XnD += D){
					const float* XmD = XnD + D;
					float* curr_elem = &DD[n*N + n];
					*curr_elem = 0.0;
					float* curr_elem_sym = curr_elem + N;
					for (int m = n + 1; m < j + B; m++, XmD += D, curr_elem_sym += N) {
						float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*(++curr_elem) = sum;
						*curr_elem_sym = sum;
					}
				}
			} else {
				// In this case, the block has to be computed all and the symmetric position is not inside the block.
				const float* XnD = X + i*D;
				for (int n = i; n < i + B; n++, XnD += D) {
					const float* XmD = X + j*D;
					float* curr_elem = &DD[n*N + j];
                    float* curr_elem_sym = &DD[j*N + n];
					for (int m = j; m < j + B; m++, XmD += D, curr_elem_sym += N, curr_elem++) {
                        float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*curr_elem = sum;
						*curr_elem_sym = sum;
					}
				}
			}

		}
	}
}

inline void blocking_64(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 64; // Desired Block size
	const int B =  (N > Bd) ? Bd : N;

	for (int i = 0; i < N; i += B) {
		for (int j = i; j < N; j += B) {

			if (i == j) {
				// Case where the resulting block of BxB needs only to be
				// computed the upper triangle part
				const float* XnD = X + i*D;
				for (int n = i; n < i+B; n++, XnD += D){
					const float* XmD = XnD + D;
					float* curr_elem = &DD[n*N + n];
					*curr_elem = 0.0;
					float* curr_elem_sym = curr_elem + N;
					for (int m = n + 1; m < j + B; m++, XmD += D, curr_elem_sym += N) {
						float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*(++curr_elem) = sum;
						*curr_elem_sym = sum;
					}
				}
			} else {
				// In this case, the block has to be computed all and the symmetric position is not inside the block.
				const float* XnD = X + i*D;
				for (int n = i; n < i + B; n++, XnD += D) {
					const float* XmD = X + j*D;
					float* curr_elem = &DD[n*N + j];
                    float* curr_elem_sym = &DD[j*N + n];
					for (int m = j; m < j + B; m++, XmD += D, curr_elem_sym += N, curr_elem++) {
                        float sum = 0.;
						for (int d = 0; d < D; d++) {
							float dif = XnD[d] - XmD[d];
							sum += dif * dif;
						}
						*curr_elem = sum;
						*curr_elem_sym = sum;
					}
				}
			}

		}
	}
}

// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void base_version(float* X, int N, int D, float* DD) {
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

#endif
