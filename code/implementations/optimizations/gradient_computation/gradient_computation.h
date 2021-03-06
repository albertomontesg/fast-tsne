#ifndef GRADIENT_COMPUTATION_H
#define GRADIENT_COMPUTATION_H

#include <stdio.h>
#include <immintrin.h>

//
// inline void unfold_d_unfold_mx4_vec_t(float* Y, float* P, float* Q, float sum_Q,
// 						 			  int N, int D, float* dC) {
// 	const int M = 4;
// 	const int B = 4;
//
// 	// Perform the computation of the gradient
// 	int nN = 0;
// 	int nD = 0;
//
// 	float inv_sum_Q = 1 / sum_Q;
// 	__m256 inv_q = _mm256_set1_ps(inv_sum_Q);
//
// 	for(int n = 0; n < N; n += B) {
// 		int mD = 0;
//
// 		__m256 dC_n = _mm256_setzero_ps();
// 		__m256 Yn   = _mm256_loadu_ps(Y + nD);
//
// 		for (int m = 0; m < N; m += M) {
//             __m256 Ym = _mm256_loadu_ps(Y + mD);
//
//             __m256 Yn_0 = _mm256_set_ps(Y[nD+1], Y[nD],
//                                         Y[nD+1], Y[nD],
//                                         Y[nD+1], Y[nD],
//                                         Y[nD+1], Y[nD]);
//             __m256 Yn_1 = _mm256_set_ps(Y[nD+D+1], Y[nD+D],
//                                         Y[nD+D+1], Y[nD+D],
//                                         Y[nD+D+1], Y[nD+D],
//                                         Y[nD+D+1], Y[nD+D]);
//             __m256 Yn_2 = _mm256_set_ps(Y[nD+2*D+1], Y[nD+2*D],
//                                         Y[nD+2*D+1], Y[nD+2*D],
//                                         Y[nD+2*D+1], Y[nD+2*D],
//                                         Y[nD+2*D+1], Y[nD+2*D]);
//             __m256 Yn_3 = _mm256_set_ps(Y[nD+3*D+1], Y[nD+3*D],
//                                         Y[nD+3*D+1], Y[nD+3*D],
//                                         Y[nD+3*D+1], Y[nD+3*D],
//                                         Y[nD+3*D+1], Y[nD+3*D]);
//
//
//
// 			__m256 p_0 = _mm256_set_ps(P[nN+3*N+m],   P[nN+3*N+m],
// 									   P[nN+2*N+m],   P[nN+2*N+m],
// 									   P[nN+N+m],     P[nN+N+m],
// 									   P[nN+m],       P[nN+m]);
// 			__m256 p_1 = _mm256_set_ps(P[nN+3*N+m+1], P[nN+3*N+m+1],
// 									   P[nN+2*N+m+1], P[nN+2*N+m+1],
// 									   P[nN+N+m+1],   P[nN+N+m+1],
// 									   P[nN+m+1],     P[nN+m+1]);
// 			__m256 p_2 = _mm256_set_ps(P[nN+3*N+m+2], P[nN+3*N+m+2],
// 									   P[nN+2*N+m+2], P[nN+2*N+m+2],
// 									   P[nN+N+m+2],   P[nN+N+m+2],
// 									   P[nN+m+2],     P[nN+m+2]);
// 			__m256 p_3 = _mm256_set_ps(P[nN+3*N+m+3], P[nN+3*N+m+3],
// 									   P[nN+2*N+m+3], P[nN+2*N+m+3],
// 									   P[nN+N+m+3],   P[nN+N+m+3],
// 									   P[nN+m+3],     P[nN+m+3]);
//
// 			__m256 q_0 = _mm256_set_ps(Q[nN+3*N+m],   Q[nN+3*N+m],
// 									   Q[nN+2*N+m],   Q[nN+2*N+m],
// 									   Q[nN+N+m],     Q[nN+N+m],
// 									   Q[nN+m],       Q[nN+m]);
// 			__m256 q_1 = _mm256_set_ps(Q[nN+3*N+m+1], Q[nN+3*N+m+1],
// 									   Q[nN+2*N+m+1], Q[nN+2*N+m+1],
// 									   Q[nN+N+m+1],   Q[nN+N+m+1],
// 									   Q[nN+m+1],     Q[nN+m+1]);
// 			__m256 q_2 = _mm256_set_ps(Q[nN+3*N+m+2], Q[nN+3*N+m+2],
// 									   Q[nN+2*N+m+2], Q[nN+2*N+m+2],
// 									   Q[nN+N+m+2],   Q[nN+N+m+2],
// 									   Q[nN+m+2],     Q[nN+m+2]);
// 			__m256 q_3 = _mm256_set_ps(Q[nN+3*N+m+3], Q[nN+3*N+m+3],
// 									   Q[nN+2*N+m+3], Q[nN+2*N+m+3],
// 									   Q[nN+N+m+3],   Q[nN+N+m+3],
// 									   Q[nN+m+3],     Q[nN+m+3]);
//
// 			__m256 Ynm_0 = _mm256_sub_ps(Yn, Ym_0);
// 			__m256 Ynm_1 = _mm256_sub_ps(Yn, Ym_1);
// 			__m256 Ynm_2 = _mm256_sub_ps(Yn, Ym_2);
// 			__m256 Ynm_3 = _mm256_sub_ps(Yn, Ym_3);
//
// 			__m256 qq_0 = _mm256_mul_ps(q_0, inv_q);
// 			__m256 qq_1 = _mm256_mul_ps(q_1, inv_q);
// 			__m256 qq_2 = _mm256_mul_ps(q_2, inv_q);
// 			__m256 qq_3 = _mm256_mul_ps(q_3, inv_q);
//
// 			__m256 pqq_0 = _mm256_sub_ps(p_0, qq_0);
// 			__m256 pqq_1 = _mm256_sub_ps(p_1, qq_1);
// 			__m256 pqq_2 = _mm256_sub_ps(p_2, qq_2);
// 			__m256 pqq_3 = _mm256_sub_ps(p_3, qq_3);
//
// 			__m256 yq_0 = _mm256_mul_ps(Ynm_0, q_0);
// 			__m256 yq_1 = _mm256_mul_ps(Ynm_1, q_1);
// 			__m256 yq_2 = _mm256_mul_ps(Ynm_2, q_2);
// 			__m256 yq_3 = _mm256_mul_ps(Ynm_3, q_3);
//
// 			__m256 dC_0 = _mm256_mul_ps(pqq_0, yq_0);
// 			__m256 dC_1 = _mm256_mul_ps(pqq_1, yq_1);
// 			__m256 dC_2 = _mm256_mul_ps(pqq_2, yq_2);
// 			__m256 dC_3 = _mm256_mul_ps(pqq_3, yq_3);
//
//
// 			__m256 dC_4 = _mm256_add_ps(dC_0, dC_1);
// 			__m256 dC_5 = _mm256_add_ps(dC_2, dC_3);
//
// 			__m256 dC_6 = _mm256_add_ps(dC_4, dC_5);
//
// 			dC_n = _mm256_add_ps(dC_n, dC_6);
//
// 			mD += M * D;
// 		}
//
// 		_mm256_storeu_ps(dC + nD, dC_n);
//
// 		nN += B * N;
// 		nD += B * D;
// 	}
// }
//

inline void unfold_d_unfold_mx8_vec(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC) {
	const int M = 8;
	const int B = 4;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;

	float inv_sum_Q = 1 / sum_Q;
	__m256 inv_q = _mm256_set1_ps(inv_sum_Q);

	for(int n = 0; n < N; n += B) {
		int mD = 0;

		__m256 dC_n = _mm256_setzero_ps();
		__m256 Yn   = _mm256_loadu_ps(Y + nD);

		for (int m = 0; m < N; m += M) {

			__m256 Ym_0 = _mm256_set_ps(Y[mD+1],     Y[mD],
										Y[mD+1],     Y[mD],
										Y[mD+1],     Y[mD],
										Y[mD+1],     Y[mD]);
			__m256 Ym_1 = _mm256_set_ps(Y[mD+D+1],   Y[mD+D],
										Y[mD+D+1],   Y[mD+D],
										Y[mD+D+1],   Y[mD+D],
										Y[mD+D+1],   Y[mD+D]);
			__m256 Ym_2 = _mm256_set_ps(Y[mD+2*D+1], Y[mD+2*D],
										Y[mD+2*D+1], Y[mD+2*D],
										Y[mD+2*D+1], Y[mD+2*D],
										Y[mD+2*D+1], Y[mD+2*D]);
			__m256 Ym_3 = _mm256_set_ps(Y[mD+3*D+1], Y[mD+3*D],
										Y[mD+3*D+1], Y[mD+3*D],
										Y[mD+3*D+1], Y[mD+3*D],
										Y[mD+3*D+1], Y[mD+3*D]);
			__m256 Ym_4 = _mm256_set_ps(Y[mD+4*D+1], Y[mD+4*D],
										Y[mD+4*D+1], Y[mD+4*D],
										Y[mD+4*D+1], Y[mD+4*D],
										Y[mD+4*D+1], Y[mD+4*D]);
			__m256 Ym_5 = _mm256_set_ps(Y[mD+5*D+1], Y[mD+5*D],
										Y[mD+5*D+1], Y[mD+5*D],
										Y[mD+5*D+1], Y[mD+5*D],
										Y[mD+5*D+1], Y[mD+5*D]);
			__m256 Ym_6 = _mm256_set_ps(Y[mD+6*D+1], Y[mD+6*D],
										Y[mD+6*D+1], Y[mD+6*D],
										Y[mD+6*D+1], Y[mD+6*D],
										Y[mD+6*D+1], Y[mD+6*D]);
			__m256 Ym_7 = _mm256_set_ps(Y[mD+7*D+1], Y[mD+7*D],
										Y[mD+7*D+1], Y[mD+7*D],
										Y[mD+7*D+1], Y[mD+7*D],
										Y[mD+7*D+1], Y[mD+7*D]);

			__m256 p_0 = _mm256_set_ps(P[nN+3*N+m],   P[nN+3*N+m],
									   P[nN+2*N+m],   P[nN+2*N+m],
									   P[nN+N+m],     P[nN+N+m],
									   P[nN+m],       P[nN+m]);
			__m256 p_1 = _mm256_set_ps(P[nN+3*N+m+1], P[nN+3*N+m+1],
									   P[nN+2*N+m+1], P[nN+2*N+m+1],
									   P[nN+N+m+1],   P[nN+N+m+1],
									   P[nN+m+1],     P[nN+m+1]);
			__m256 p_2 = _mm256_set_ps(P[nN+3*N+m+2], P[nN+3*N+m+2],
									   P[nN+2*N+m+2], P[nN+2*N+m+2],
									   P[nN+N+m+2],   P[nN+N+m+2],
									   P[nN+m+2],     P[nN+m+2]);
			__m256 p_3 = _mm256_set_ps(P[nN+3*N+m+3], P[nN+3*N+m+3],
									   P[nN+2*N+m+3], P[nN+2*N+m+3],
									   P[nN+N+m+3],   P[nN+N+m+3],
									   P[nN+m+3],     P[nN+m+3]);
			__m256 p_4 = _mm256_set_ps(P[nN+3*N+m+4], P[nN+3*N+m+4],
									   P[nN+2*N+m+4], P[nN+2*N+m+4],
									   P[nN+N+m+4],   P[nN+N+m+4],
									   P[nN+m+4],     P[nN+m+4]);
			__m256 p_5 = _mm256_set_ps(P[nN+3*N+m+5], P[nN+3*N+m+5],
									   P[nN+2*N+m+5], P[nN+2*N+m+5],
									   P[nN+N+m+5],   P[nN+N+m+5],
									   P[nN+m+5],     P[nN+m+5]);
			__m256 p_6 = _mm256_set_ps(P[nN+3*N+m+6], P[nN+3*N+m+6],
									   P[nN+2*N+m+6], P[nN+2*N+m+6],
									   P[nN+N+m+6],   P[nN+N+m+6],
									   P[nN+m+6],     P[nN+m+6]);
			__m256 p_7 = _mm256_set_ps(P[nN+3*N+m+7], P[nN+3*N+m+7],
									   P[nN+2*N+m+7], P[nN+2*N+m+7],
									   P[nN+N+m+7],   P[nN+N+m+7],
									   P[nN+m+7],     P[nN+m+7]);

			__m256 q_0 = _mm256_set_ps(Q[nN+3*N+m],   Q[nN+3*N+m],
									   Q[nN+2*N+m],   Q[nN+2*N+m],
									   Q[nN+N+m],     Q[nN+N+m],
									   Q[nN+m],       Q[nN+m]);
			__m256 q_1 = _mm256_set_ps(Q[nN+3*N+m+1], Q[nN+3*N+m+1],
									   Q[nN+2*N+m+1], Q[nN+2*N+m+1],
									   Q[nN+N+m+1],   Q[nN+N+m+1],
									   Q[nN+m+1],     Q[nN+m+1]);
			__m256 q_2 = _mm256_set_ps(Q[nN+3*N+m+2], Q[nN+3*N+m+2],
									   Q[nN+2*N+m+2], Q[nN+2*N+m+2],
									   Q[nN+N+m+2],   Q[nN+N+m+2],
									   Q[nN+m+2],     Q[nN+m+2]);
			__m256 q_3 = _mm256_set_ps(Q[nN+3*N+m+3], Q[nN+3*N+m+3],
									   Q[nN+2*N+m+3], Q[nN+2*N+m+3],
									   Q[nN+N+m+3],   Q[nN+N+m+3],
									   Q[nN+m+3],     Q[nN+m+3]);
			__m256 q_4 = _mm256_set_ps(Q[nN+3*N+m+4], Q[nN+3*N+m+4],
									   Q[nN+2*N+m+4], Q[nN+2*N+m+4],
									   Q[nN+N+m+4],   Q[nN+N+m+4],
									   Q[nN+m+4],     Q[nN+m+4]);
			__m256 q_5 = _mm256_set_ps(Q[nN+3*N+m+5], Q[nN+3*N+m+5],
									   Q[nN+2*N+m+5], Q[nN+2*N+m+5],
									   Q[nN+N+m+5],   Q[nN+N+m+5],
									   Q[nN+m+5],     Q[nN+m+5]);
			__m256 q_6 = _mm256_set_ps(Q[nN+3*N+m+6], Q[nN+3*N+m+6],
									   Q[nN+2*N+m+6], Q[nN+2*N+m+6],
									   Q[nN+N+m+6],   Q[nN+N+m+6],
									   Q[nN+m+6],     Q[nN+m+6]);
			__m256 q_7 = _mm256_set_ps(Q[nN+3*N+m+7], Q[nN+3*N+m+7],
									   Q[nN+2*N+m+7], Q[nN+2*N+m+7],
									   Q[nN+N+m+7],   Q[nN+N+m+7],
									   Q[nN+m+7],     Q[nN+m+7]);

			__m256 Ynm_0 = _mm256_sub_ps(Yn, Ym_0);
			__m256 Ynm_1 = _mm256_sub_ps(Yn, Ym_1);
			__m256 Ynm_2 = _mm256_sub_ps(Yn, Ym_2);
			__m256 Ynm_3 = _mm256_sub_ps(Yn, Ym_3);
			__m256 Ynm_4 = _mm256_sub_ps(Yn, Ym_4);
			__m256 Ynm_5 = _mm256_sub_ps(Yn, Ym_5);
			__m256 Ynm_6 = _mm256_sub_ps(Yn, Ym_6);
			__m256 Ynm_7 = _mm256_sub_ps(Yn, Ym_7);

			__m256 qq_0 = _mm256_mul_ps(q_0, inv_q);
			__m256 qq_1 = _mm256_mul_ps(q_1, inv_q);
			__m256 qq_2 = _mm256_mul_ps(q_2, inv_q);
			__m256 qq_3 = _mm256_mul_ps(q_3, inv_q);
			__m256 qq_4 = _mm256_mul_ps(q_4, inv_q);
			__m256 qq_5 = _mm256_mul_ps(q_5, inv_q);
			__m256 qq_6 = _mm256_mul_ps(q_6, inv_q);
			__m256 qq_7 = _mm256_mul_ps(q_7, inv_q);

			__m256 pqq_0 = _mm256_sub_ps(p_0, qq_0);
			__m256 pqq_1 = _mm256_sub_ps(p_1, qq_1);
			__m256 pqq_2 = _mm256_sub_ps(p_2, qq_2);
			__m256 pqq_3 = _mm256_sub_ps(p_3, qq_3);
			__m256 pqq_4 = _mm256_sub_ps(p_4, qq_4);
			__m256 pqq_5 = _mm256_sub_ps(p_5, qq_5);
			__m256 pqq_6 = _mm256_sub_ps(p_6, qq_6);
			__m256 pqq_7 = _mm256_sub_ps(p_7, qq_7);

			__m256 yq_0 = _mm256_mul_ps(Ynm_0, q_0);
			__m256 yq_1 = _mm256_mul_ps(Ynm_1, q_1);
			__m256 yq_2 = _mm256_mul_ps(Ynm_2, q_2);
			__m256 yq_3 = _mm256_mul_ps(Ynm_3, q_3);
			__m256 yq_4 = _mm256_mul_ps(Ynm_4, q_4);
			__m256 yq_5 = _mm256_mul_ps(Ynm_5, q_5);
			__m256 yq_6 = _mm256_mul_ps(Ynm_6, q_6);
			__m256 yq_7 = _mm256_mul_ps(Ynm_7, q_7);

			__m256 dC_0 = _mm256_mul_ps(pqq_0, yq_0);
			__m256 dC_1 = _mm256_mul_ps(pqq_1, yq_1);
			__m256 dC_2 = _mm256_mul_ps(pqq_2, yq_2);
			__m256 dC_3 = _mm256_mul_ps(pqq_3, yq_3);
			__m256 dC_4 = _mm256_mul_ps(pqq_4, yq_4);
			__m256 dC_5 = _mm256_mul_ps(pqq_5, yq_5);
			__m256 dC_6 = _mm256_mul_ps(pqq_6, yq_6);
			__m256 dC_7 = _mm256_mul_ps(pqq_7, yq_7);


			__m256 dC_8 = _mm256_add_ps(dC_0, dC_1);
			__m256 dC_9 = _mm256_add_ps(dC_2, dC_3);
			__m256 dC_10 = _mm256_add_ps(dC_4, dC_5);
			__m256 dC_11 = _mm256_add_ps(dC_6, dC_7);

			__m256 dC_12 = _mm256_add_ps(dC_8, dC_9);
			__m256 dC_13 = _mm256_add_ps(dC_10, dC_11);

			__m256 dC_14 = _mm256_add_ps(dC_12, dC_13);

			dC_n = _mm256_add_ps(dC_n, dC_14);

			mD += M * D;
		}

		_mm256_storeu_ps(dC + nD, dC_n);

		nN += B * N;
		nD += B * D;
	}
}


inline void unfold_d_unfold_mx4_vec(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC) {
	const int M = 4;
	const int B = 4;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;

	float inv_sum_Q = 1 / sum_Q;
	__m256 inv_q = _mm256_set1_ps(inv_sum_Q);

	for(int n = 0; n < N; n += B) {
		int mD = 0;

		__m256 dC_n = _mm256_setzero_ps();
		__m256 Yn   = _mm256_loadu_ps(Y + nD);

		for (int m = 0; m < N; m += M) {
			__m256 Ym_0 = _mm256_set_ps(Y[mD+1],     Y[mD],
										Y[mD+1],     Y[mD],
										Y[mD+1],     Y[mD],
										Y[mD+1],     Y[mD]);
			__m256 Ym_1 = _mm256_set_ps(Y[mD+D+1],   Y[mD+D],
										Y[mD+D+1],   Y[mD+D],
										Y[mD+D+1],   Y[mD+D],
										Y[mD+D+1],   Y[mD+D]);
			__m256 Ym_2 = _mm256_set_ps(Y[mD+2*D+1], Y[mD+2*D],
										Y[mD+2*D+1], Y[mD+2*D],
										Y[mD+2*D+1], Y[mD+2*D],
										Y[mD+2*D+1], Y[mD+2*D]);
			__m256 Ym_3 = _mm256_set_ps(Y[mD+3*D+1], Y[mD+3*D],
										Y[mD+3*D+1], Y[mD+3*D],
										Y[mD+3*D+1], Y[mD+3*D],
										Y[mD+3*D+1], Y[mD+3*D]);

			__m256 p_0 = _mm256_set_ps(P[nN+3*N+m],   P[nN+3*N+m],
									   P[nN+2*N+m],   P[nN+2*N+m],
									   P[nN+N+m],     P[nN+N+m],
									   P[nN+m],       P[nN+m]);
			__m256 p_1 = _mm256_set_ps(P[nN+3*N+m+1], P[nN+3*N+m+1],
									   P[nN+2*N+m+1], P[nN+2*N+m+1],
									   P[nN+N+m+1],   P[nN+N+m+1],
									   P[nN+m+1],     P[nN+m+1]);
			__m256 p_2 = _mm256_set_ps(P[nN+3*N+m+2], P[nN+3*N+m+2],
									   P[nN+2*N+m+2], P[nN+2*N+m+2],
									   P[nN+N+m+2],   P[nN+N+m+2],
									   P[nN+m+2],     P[nN+m+2]);
			__m256 p_3 = _mm256_set_ps(P[nN+3*N+m+3], P[nN+3*N+m+3],
									   P[nN+2*N+m+3], P[nN+2*N+m+3],
									   P[nN+N+m+3],   P[nN+N+m+3],
									   P[nN+m+3],     P[nN+m+3]);

			__m256 q_0 = _mm256_set_ps(Q[nN+3*N+m],   Q[nN+3*N+m],
									   Q[nN+2*N+m],   Q[nN+2*N+m],
									   Q[nN+N+m],     Q[nN+N+m],
									   Q[nN+m],       Q[nN+m]);
			__m256 q_1 = _mm256_set_ps(Q[nN+3*N+m+1], Q[nN+3*N+m+1],
									   Q[nN+2*N+m+1], Q[nN+2*N+m+1],
									   Q[nN+N+m+1],   Q[nN+N+m+1],
									   Q[nN+m+1],     Q[nN+m+1]);
			__m256 q_2 = _mm256_set_ps(Q[nN+3*N+m+2], Q[nN+3*N+m+2],
									   Q[nN+2*N+m+2], Q[nN+2*N+m+2],
									   Q[nN+N+m+2],   Q[nN+N+m+2],
									   Q[nN+m+2],     Q[nN+m+2]);
			__m256 q_3 = _mm256_set_ps(Q[nN+3*N+m+3], Q[nN+3*N+m+3],
									   Q[nN+2*N+m+3], Q[nN+2*N+m+3],
									   Q[nN+N+m+3],   Q[nN+N+m+3],
									   Q[nN+m+3],     Q[nN+m+3]);

			__m256 Ynm_0 = _mm256_sub_ps(Yn, Ym_0);
			__m256 Ynm_1 = _mm256_sub_ps(Yn, Ym_1);
			__m256 Ynm_2 = _mm256_sub_ps(Yn, Ym_2);
			__m256 Ynm_3 = _mm256_sub_ps(Yn, Ym_3);

			__m256 qq_0 = _mm256_mul_ps(q_0, inv_q);
			__m256 qq_1 = _mm256_mul_ps(q_1, inv_q);
			__m256 qq_2 = _mm256_mul_ps(q_2, inv_q);
			__m256 qq_3 = _mm256_mul_ps(q_3, inv_q);

			__m256 pqq_0 = _mm256_sub_ps(p_0, qq_0);
			__m256 pqq_1 = _mm256_sub_ps(p_1, qq_1);
			__m256 pqq_2 = _mm256_sub_ps(p_2, qq_2);
			__m256 pqq_3 = _mm256_sub_ps(p_3, qq_3);

			__m256 yq_0 = _mm256_mul_ps(Ynm_0, q_0);
			__m256 yq_1 = _mm256_mul_ps(Ynm_1, q_1);
			__m256 yq_2 = _mm256_mul_ps(Ynm_2, q_2);
			__m256 yq_3 = _mm256_mul_ps(Ynm_3, q_3);

			__m256 dC_0 = _mm256_mul_ps(pqq_0, yq_0);
			__m256 dC_1 = _mm256_mul_ps(pqq_1, yq_1);
			__m256 dC_2 = _mm256_mul_ps(pqq_2, yq_2);
			__m256 dC_3 = _mm256_mul_ps(pqq_3, yq_3);


			__m256 dC_4 = _mm256_add_ps(dC_0, dC_1);
			__m256 dC_5 = _mm256_add_ps(dC_2, dC_3);

			__m256 dC_6 = _mm256_add_ps(dC_4, dC_5);

			dC_n = _mm256_add_ps(dC_n, dC_6);

			mD += M * D;
		}

		_mm256_storeu_ps(dC + nD, dC_n);

		nN += B * N;
		nD += B * D;
	}
}

inline void unfold_d_unfold_mx8(float* Y, float* P, float* Q, float sum_Q,
								int N, int D, float* dC) {
	int M = 8;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;

	float inv_sum_Q = 1 / sum_Q;

	for(int n = 0; n < N; n++) {
		int mD = 0;
		float dC_n_0 = 0, dC_n_1 = 0;

		float Yn_0 = Y[nD];
		float Yn_1 = Y[nD + 1];

		for (int m = 0; m < N; m += M) {
			float p_0 = P[nN + m];
			float p_1 = P[nN + m + 1];
			float p_2 = P[nN + m + 2];
			float p_3 = P[nN + m + 3];
			float p_4 = P[nN + m + 4];
			float p_5 = P[nN + m + 5];
			float p_6 = P[nN + m + 6];
			float p_7 = P[nN + m + 7];

			float q_0 = Q[nN + m];
			float q_1 = Q[nN + m + 1];
			float q_2 = Q[nN + m + 2];
			float q_3 = Q[nN + m + 3];
			float q_4 = Q[nN + m + 4];
			float q_5 = Q[nN + m + 5];
			float q_6 = Q[nN + m + 6];
			float q_7 = Q[nN + m + 7];

			float Ym_0_0 = Y[mD + 0*D + 0];
			float Ym_0_1 = Y[mD + 0*D + 1];
			float Ym_1_0 = Y[mD + 1*D + 0];
			float Ym_1_1 = Y[mD + 1*D + 1];
			float Ym_2_0 = Y[mD + 2*D + 0];
			float Ym_2_1 = Y[mD + 2*D + 1];
			float Ym_3_0 = Y[mD + 3*D + 0];
			float Ym_3_1 = Y[mD + 3*D + 1];
			float Ym_4_0 = Y[mD + 4*D + 0];
			float Ym_4_1 = Y[mD + 4*D + 1];
			float Ym_5_0 = Y[mD + 5*D + 0];
			float Ym_5_1 = Y[mD + 5*D + 1];
			float Ym_6_0 = Y[mD + 6*D + 0];
			float Ym_6_1 = Y[mD + 6*D + 1];
			float Ym_7_0 = Y[mD + 7*D + 0];
			float Ym_7_1 = Y[mD + 7*D + 1];

			float qq_0 = q_0 * inv_sum_Q;
			float qq_1 = q_1 * inv_sum_Q;
			float qq_2 = q_2 * inv_sum_Q;
			float qq_3 = q_3 * inv_sum_Q;
			float qq_4 = q_4 * inv_sum_Q;
			float qq_5 = q_5 * inv_sum_Q;
			float qq_6 = q_6 * inv_sum_Q;
			float qq_7 = q_7 * inv_sum_Q;

			float pq_0 = p_0 - qq_0;
			float pq_1 = p_1 - qq_1;
			float pq_2 = p_2 - qq_2;
			float pq_3 = p_3 - qq_3;
			float pq_4 = p_4 - qq_4;
			float pq_5 = p_5 - qq_5;
			float pq_6 = p_6 - qq_6;
			float pq_7 = p_7 - qq_7;

			float mult_0 = pq_0 * q_0;
			float mult_1 = pq_1 * q_1;
			float mult_2 = pq_2 * q_2;
			float mult_3 = pq_3 * q_3;
			float mult_4 = pq_4 * q_4;
			float mult_5 = pq_5 * q_5;
			float mult_6 = pq_6 * q_6;
			float mult_7 = pq_7 * q_7;

			float y_y0_0 = Yn_0 - Ym_0_0;
			float y_y0_1 = Yn_1 - Ym_0_1;
			float y_y1_0 = Yn_0 - Ym_1_0;
			float y_y1_1 = Yn_1 - Ym_1_1;
			float y_y2_0 = Yn_0 - Ym_2_0;
			float y_y2_1 = Yn_1 - Ym_2_1;
			float y_y3_0 = Yn_0 - Ym_3_0;
			float y_y3_1 = Yn_1 - Ym_3_1;
			float y_y4_0 = Yn_0 - Ym_4_0;
			float y_y4_1 = Yn_1 - Ym_4_1;
			float y_y5_0 = Yn_0 - Ym_5_0;
			float y_y5_1 = Yn_1 - Ym_5_1;
			float y_y6_0 = Yn_0 - Ym_6_0;
			float y_y6_1 = Yn_1 - Ym_6_1;
			float y_y7_0 = Yn_0 - Ym_7_0;
			float y_y7_1 = Yn_1 - Ym_7_1;

			float sum0_0 = y_y0_0 * mult_0;
			float sum0_1 = y_y0_1 * mult_0;
			float sum1_0 = y_y1_0 * mult_1;
			float sum1_1 = y_y1_1 * mult_1;
			float sum2_0 = y_y2_0 * mult_2;
			float sum2_1 = y_y2_1 * mult_2;
			float sum3_0 = y_y3_0 * mult_3;
			float sum3_1 = y_y3_1 * mult_3;
			float sum4_0 = y_y4_0 * mult_4;
			float sum4_1 = y_y4_1 * mult_4;
			float sum5_0 = y_y5_0 * mult_5;
			float sum5_1 = y_y5_1 * mult_5;
			float sum6_0 = y_y6_0 * mult_6;
			float sum6_1 = y_y6_1 * mult_6;
			float sum7_0 = y_y7_0 * mult_7;
			float sum7_1 = y_y7_1 * mult_7;

			float sum8_0 = sum0_0 + sum1_0;
			float sum8_1 = sum0_1 + sum1_1;
			float sum9_0 = sum2_0 + sum3_0;
			float sum9_1 = sum2_1 + sum3_1;
			float sum10_0 = sum4_0 + sum5_0;
			float sum10_1 = sum4_1 + sum5_1;
			float sum11_0 = sum6_0 + sum7_0;
			float sum11_1 = sum6_1 + sum7_1;

			float sum12_0 = sum8_0 + sum9_0;
			float sum12_1 = sum8_1 + sum9_1;
			float sum13_0 = sum10_0 + sum11_0;
			float sum13_1 = sum10_1 + sum11_1;

			float sum14_0 = sum12_0 + sum13_0;
			float sum14_1 = sum12_1 + sum13_1;

			dC_n_0 += sum14_0;
			dC_n_1 += sum14_1;

			mD += M*D;
		}

		dC[nD] = dC_n_0;
		dC[nD + 1] = dC_n_1;
		nN += N;
		nD += D;
	}
}

inline void unfold_d_unfold_mx4(float* Y, float* P, float* Q, float sum_Q,
						 int N, int D, float* dC) {
	int M = 4;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;

	float inv_sum_Q = 1 / sum_Q;

	for(int n = 0; n < N; n++) {
		int mD = 0;
		float dC_n_0 = 0, dC_n_1 = 0;

		float Yn_0 = Y[nD];
		float Yn_1 = Y[nD + 1];

		for (int m = 0; m < N; m += M) {
			float p_0 = P[nN + m];
			float p_1 = P[nN + m + 1];
			float p_2 = P[nN + m + 2];
			float p_3 = P[nN + m + 3];

			float q_0 = Q[nN + m];
			float q_1 = Q[nN + m + 1];
			float q_2 = Q[nN + m + 2];
			float q_3 = Q[nN + m + 3];

			float Ym_0_0 = Y[mD + 0*D + 0];
			float Ym_0_1 = Y[mD + 0*D + 1];
			float Ym_1_0 = Y[mD + 1*D + 0];
			float Ym_1_1 = Y[mD + 1*D + 1];
			float Ym_2_0 = Y[mD + 2*D + 0];
			float Ym_2_1 = Y[mD + 2*D + 1];
			float Ym_3_0 = Y[mD + 3*D + 0];
			float Ym_3_1 = Y[mD + 3*D + 1];

			float qq_0 = q_0 * inv_sum_Q;
			float qq_1 = q_1 * inv_sum_Q;
			float qq_2 = q_2 * inv_sum_Q;
			float qq_3 = q_3 * inv_sum_Q;

			float pq_0 = p_0 - qq_0;
			float pq_1 = p_1 - qq_1;
			float pq_2 = p_2 - qq_2;
			float pq_3 = p_3 - qq_3;

			float mult_0 = pq_0 * q_0;
			float mult_1 = pq_1 * q_1;
			float mult_2 = pq_2 * q_2;
			float mult_3 = pq_3 * q_3;

			float y_y0_0 = Yn_0 - Ym_0_0;
			float y_y0_1 = Yn_1 - Ym_0_1;
			float y_y1_0 = Yn_0 - Ym_1_0;
			float y_y1_1 = Yn_1 - Ym_1_1;
			float y_y2_0 = Yn_0 - Ym_2_0;
			float y_y2_1 = Yn_1 - Ym_2_1;
			float y_y3_0 = Yn_0 - Ym_3_0;
			float y_y3_1 = Yn_1 - Ym_3_1;

			float sum0_0 = y_y0_0 * mult_0;
			float sum0_1 = y_y0_1 * mult_0;
			float sum1_0 = y_y1_0 * mult_1;
			float sum1_1 = y_y1_1 * mult_1;
			float sum2_0 = y_y2_0 * mult_2;
			float sum2_1 = y_y2_1 * mult_2;
			float sum3_0 = y_y3_0 * mult_3;
			float sum3_1 = y_y3_1 * mult_3;

			float sum4_0 = sum0_0 + sum1_0;
			float sum4_1 = sum0_1 + sum1_1;
			float sum5_0 = sum2_0 + sum3_0;
			float sum5_1 = sum2_1 + sum3_1;

			float sum6_0 = sum4_0 + sum5_0;
			float sum6_1 = sum4_1 + sum5_1;

			dC_n_0 += sum6_0;
			dC_n_1 += sum6_1;

			mD += M*D;
		}

		dC[nD] = dC_n_0;
		dC[nD + 1] = dC_n_1;
		nN += N;
		nD += D;
	}
}

inline void unfold_d(float* Y, float* P, float* Q, float sum_Q,
						 int N, int D, float* dC) {
	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		int mD = 0;
		float dC_n_0 = 0, dC_n_1 = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				float mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				dC_n_0 += (Y[nD + 0] - Y[mD + 0]) * mult;
				dC_n_1 += (Y[nD + 1] - Y[mD + 1]) * mult;
			}
			mD += D;
		}
		dC[nD] = dC_n_0;
		dC[nD + 1] = dC_n_1;
		nN += N;
		nD += D;
	}
}

// Gradient computation dC_dy
inline void base_version(float* Y, float* P, float* Q, float sum_Q,
						 int N, int D, float* dC) {
	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		int mD = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				float mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for(int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
			}
			mD += D;
		}
		nN += N;
		nD += D;
	}
}

#endif
