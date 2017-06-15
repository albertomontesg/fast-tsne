#ifndef GRADIENT_COMPUTATION_H
#define GRADIENT_COMPUTATION_H

#include <stdio.h>
#include <immintrin.h>

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2,
        __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}


inline void unfold_d_unfold_nx4_mx8_vec(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC, float* uY,
									float momentum, float eta) {
	const int M = 8;
	const int K = 4;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;

	float inv_sum_Q = 1 / sum_Q;
	__m256 inv_q = _mm256_set1_ps(inv_sum_Q);
	__m256 m_eta = _mm256_set1_ps(-eta);
	__m256 mom = _mm256_set1_ps(momentum);

	for(int n = 0; n < N; n += 4*K) {
		int mD = 0;

		__m256 dCn_0 = _mm256_setzero_ps();
		__m256 dCn_1 = _mm256_setzero_ps();
		__m256 dCn_2 = _mm256_setzero_ps();
		__m256 dCn_3 = _mm256_setzero_ps();

		__m256 Yn_0 = _mm256_loadu_ps(Y + nD);
		__m256 Yn_1 = _mm256_loadu_ps(Y + nD + 8);
		__m256 Yn_2 = _mm256_loadu_ps(Y + nD + 16);
		__m256 Yn_3 = _mm256_loadu_ps(Y + nD + 24);

        __m256 uY_0 = _mm256_loadu_ps(uY+nD);
        __m256 uY_1 = _mm256_loadu_ps(uY+nD+8);
        __m256 uY_2 = _mm256_loadu_ps(uY+nD+16);
        __m256 uY_3 = _mm256_loadu_ps(uY+nD+24);

		for (int m = 0; m < N; m += M, mD += M * D) {
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

			// Load P
			__m256 p_00 = _mm256_load_ps(P+nN+m);
			__m256 p_01 = _mm256_load_ps(P+nN+N+m);
			__m256 p_02 = _mm256_load_ps(P+nN+2*N+m);
			__m256 p_03 = _mm256_load_ps(P+nN+3*N+m);

			__m256 p_10 = _mm256_load_ps(P+nN+4*N+m);
			__m256 p_11 = _mm256_load_ps(P+nN+5*N+m);
			__m256 p_12 = _mm256_load_ps(P+nN+6*N+m);
			__m256 p_13 = _mm256_load_ps(P+nN+7*N+m);

			__m256 p_20 = _mm256_load_ps(P+nN+8*N+m);
			__m256 p_21 = _mm256_load_ps(P+nN+9*N+m);
			__m256 p_22 = _mm256_load_ps(P+nN+10*N+m);
			__m256 p_23 = _mm256_load_ps(P+nN+11*N+m);

			__m256 p_30 = _mm256_load_ps(P+nN+12*N+m);
			__m256 p_31 = _mm256_load_ps(P+nN+13*N+m);
			__m256 p_32 = _mm256_load_ps(P+nN+14*N+m);
			__m256 p_33 = _mm256_load_ps(P+nN+15*N+m);

			// Load Q
			__m256 q_00 = _mm256_load_ps(Q+nN+m);
			__m256 q_01 = _mm256_load_ps(Q+nN+N+m);
			__m256 q_02 = _mm256_load_ps(Q+nN+2*N+m);
			__m256 q_03 = _mm256_load_ps(Q+nN+3*N+m);

			__m256 q_10 = _mm256_load_ps(Q+nN+4*N+m);
			__m256 q_11 = _mm256_load_ps(Q+nN+5*N+m);
			__m256 q_12 = _mm256_load_ps(Q+nN+6*N+m);
			__m256 q_13 = _mm256_load_ps(Q+nN+7*N+m);

			__m256 q_20 = _mm256_load_ps(Q+nN+8*N+m);
			__m256 q_21 = _mm256_load_ps(Q+nN+9*N+m);
			__m256 q_22 = _mm256_load_ps(Q+nN+10*N+m);
			__m256 q_23 = _mm256_load_ps(Q+nN+11*N+m);

			__m256 q_30 = _mm256_load_ps(Q+nN+12*N+m);
			__m256 q_31 = _mm256_load_ps(Q+nN+13*N+m);
			__m256 q_32 = _mm256_load_ps(Q+nN+14*N+m);
			__m256 q_33 = _mm256_load_ps(Q+nN+15*N+m);

			// Compute (q_ij)
			__m256 qq_00 = _mm256_mul_ps(q_00, inv_q);
			__m256 qq_01 = _mm256_mul_ps(q_01, inv_q);
			__m256 qq_02 = _mm256_mul_ps(q_02, inv_q);
			__m256 qq_03 = _mm256_mul_ps(q_03, inv_q);

			__m256 qq_10 = _mm256_mul_ps(q_10, inv_q);
			__m256 qq_11 = _mm256_mul_ps(q_11, inv_q);
			__m256 qq_12 = _mm256_mul_ps(q_12, inv_q);
			__m256 qq_13 = _mm256_mul_ps(q_13, inv_q);

			__m256 qq_20 = _mm256_mul_ps(q_20, inv_q);
			__m256 qq_21 = _mm256_mul_ps(q_21, inv_q);
			__m256 qq_22 = _mm256_mul_ps(q_22, inv_q);
			__m256 qq_23 = _mm256_mul_ps(q_23, inv_q);

			__m256 qq_30 = _mm256_mul_ps(q_30, inv_q);
			__m256 qq_31 = _mm256_mul_ps(q_31, inv_q);
			__m256 qq_32 = _mm256_mul_ps(q_32, inv_q);
			__m256 qq_33 = _mm256_mul_ps(q_33, inv_q);
			// }


			// Compute (y_i - y_j)
			__m256 Ynm_00 = _mm256_sub_ps(Yn_0, Ym_0);
			__m256 Ynm_01 = _mm256_sub_ps(Yn_0, Ym_1);
			__m256 Ynm_02 = _mm256_sub_ps(Yn_0, Ym_2);
			__m256 Ynm_03 = _mm256_sub_ps(Yn_0, Ym_3);
			__m256 Ynm_04 = _mm256_sub_ps(Yn_0, Ym_4);
			__m256 Ynm_05 = _mm256_sub_ps(Yn_0, Ym_5);
			__m256 Ynm_06 = _mm256_sub_ps(Yn_0, Ym_6);
			__m256 Ynm_07 = _mm256_sub_ps(Yn_0, Ym_7);

			__m256 Ynm_10 = _mm256_sub_ps(Yn_1, Ym_0);
			__m256 Ynm_11 = _mm256_sub_ps(Yn_1, Ym_1);
			__m256 Ynm_12 = _mm256_sub_ps(Yn_1, Ym_2);
			__m256 Ynm_13 = _mm256_sub_ps(Yn_1, Ym_3);
			__m256 Ynm_14 = _mm256_sub_ps(Yn_1, Ym_4);
			__m256 Ynm_15 = _mm256_sub_ps(Yn_1, Ym_5);
			__m256 Ynm_16 = _mm256_sub_ps(Yn_1, Ym_6);
			__m256 Ynm_17 = _mm256_sub_ps(Yn_1, Ym_7);

			__m256 Ynm_20 = _mm256_sub_ps(Yn_2, Ym_0);
			__m256 Ynm_21 = _mm256_sub_ps(Yn_2, Ym_1);
			__m256 Ynm_22 = _mm256_sub_ps(Yn_2, Ym_2);
			__m256 Ynm_23 = _mm256_sub_ps(Yn_2, Ym_3);
			__m256 Ynm_24 = _mm256_sub_ps(Yn_2, Ym_4);
			__m256 Ynm_25 = _mm256_sub_ps(Yn_2, Ym_5);
			__m256 Ynm_26 = _mm256_sub_ps(Yn_2, Ym_6);
			__m256 Ynm_27 = _mm256_sub_ps(Yn_2, Ym_7);

			__m256 Ynm_30 = _mm256_sub_ps(Yn_3, Ym_0);
			__m256 Ynm_31 = _mm256_sub_ps(Yn_3, Ym_1);
			__m256 Ynm_32 = _mm256_sub_ps(Yn_3, Ym_2);
			__m256 Ynm_33 = _mm256_sub_ps(Yn_3, Ym_3);
			__m256 Ynm_34 = _mm256_sub_ps(Yn_3, Ym_4);
			__m256 Ynm_35 = _mm256_sub_ps(Yn_3, Ym_5);
			__m256 Ynm_36 = _mm256_sub_ps(Yn_3, Ym_6);
			__m256 Ynm_37 = _mm256_sub_ps(Yn_3, Ym_7);
			// }


			// Compute (p_ij - q_ij)
			__m256 pq_00 = _mm256_sub_ps(p_00, qq_00);
			__m256 pq_01 = _mm256_sub_ps(p_01, qq_01);
			__m256 pq_02 = _mm256_sub_ps(p_02, qq_02);
			__m256 pq_03 = _mm256_sub_ps(p_03, qq_03);

			__m256 pq_10 = _mm256_sub_ps(p_10, qq_10);
			__m256 pq_11 = _mm256_sub_ps(p_11, qq_11);
			__m256 pq_12 = _mm256_sub_ps(p_12, qq_12);
			__m256 pq_13 = _mm256_sub_ps(p_13, qq_13);

			__m256 pq_20 = _mm256_sub_ps(p_20, qq_20);
			__m256 pq_21 = _mm256_sub_ps(p_21, qq_21);
			__m256 pq_22 = _mm256_sub_ps(p_22, qq_22);
			__m256 pq_23 = _mm256_sub_ps(p_23, qq_23);

			__m256 pq_30 = _mm256_sub_ps(p_30, qq_30);
			__m256 pq_31 = _mm256_sub_ps(p_31, qq_31);
			__m256 pq_32 = _mm256_sub_ps(p_32, qq_32);
			__m256 pq_33 = _mm256_sub_ps(p_33, qq_33);
			// }

            // Compute (p_ij - q_ij)(1-|y_i-y_j|^2)^-1
            __m256 pqq_00 = _mm256_mul_ps(pq_00, q_00);
			__m256 pqq_01 = _mm256_mul_ps(pq_01, q_01);
			__m256 pqq_02 = _mm256_mul_ps(pq_02, q_02);
			__m256 pqq_03 = _mm256_mul_ps(pq_03, q_03);

			__m256 pqq_10 = _mm256_mul_ps(pq_10, q_10);
			__m256 pqq_11 = _mm256_mul_ps(pq_11, q_11);
			__m256 pqq_12 = _mm256_mul_ps(pq_12, q_12);
			__m256 pqq_13 = _mm256_mul_ps(pq_13, q_13);

			__m256 pqq_20 = _mm256_mul_ps(pq_20, q_20);
			__m256 pqq_21 = _mm256_mul_ps(pq_21, q_21);
			__m256 pqq_22 = _mm256_mul_ps(pq_22, q_22);
			__m256 pqq_23 = _mm256_mul_ps(pq_23, q_23);

			__m256 pqq_30 = _mm256_mul_ps(pq_30, q_30);
			__m256 pqq_31 = _mm256_mul_ps(pq_31, q_31);
			__m256 pqq_32 = _mm256_mul_ps(pq_32, q_32);
			__m256 pqq_33 = _mm256_mul_ps(pq_33, q_33);


			__m256 f_00 = pqq_00, f_01 = pqq_00, f_02 = pqq_01, f_03 = pqq_01;
			__m256 f_04 = pqq_02, f_05 = pqq_02, f_06 = pqq_03, f_07 = pqq_03;

			__m256 f_10 = pqq_10, f_11 = pqq_10, f_12 = pqq_11, f_13 = pqq_11;
			__m256 f_14 = pqq_12, f_15 = pqq_12, f_16 = pqq_13, f_17 = pqq_13;

			__m256 f_20 = pqq_20, f_21 = pqq_20, f_22 = pqq_21, f_23 = pqq_21;
			__m256 f_24 = pqq_22, f_25 = pqq_22, f_26 = pqq_23, f_27 = pqq_23;

			__m256 f_30 = pqq_30, f_31 = pqq_30, f_32 = pqq_31, f_33 = pqq_31;
			__m256 f_34 = pqq_32, f_35 = pqq_32, f_36 = pqq_33, f_37 = pqq_33;

			// Transpose
			transpose8_ps(f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07);
			transpose8_ps(f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17);
			transpose8_ps(f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_27);
			transpose8_ps(f_30, f_31, f_32, f_33, f_34, f_35, f_36, f_37);

			// Compute the gradient dC
			__m256 dC_00 = _mm256_mul_ps(f_00, Ynm_00);
			__m256 dC_01 = _mm256_mul_ps(f_01, Ynm_01);
			__m256 dC_02 = _mm256_mul_ps(f_02, Ynm_02);
			__m256 dC_03 = _mm256_mul_ps(f_03, Ynm_03);
			__m256 dC_04 = _mm256_mul_ps(f_04, Ynm_04);
			__m256 dC_05 = _mm256_mul_ps(f_05, Ynm_05);
			__m256 dC_06 = _mm256_mul_ps(f_06, Ynm_06);
			__m256 dC_07 = _mm256_mul_ps(f_07, Ynm_07);
			__m256 _dC_00 = _mm256_add_ps(dC_00, dC_01);
			__m256 _dC_01 = _mm256_add_ps(dC_02, dC_03);
			__m256 _dC_02 = _mm256_add_ps(dC_04, dC_05);
			__m256 _dC_03 = _mm256_add_ps(dC_06, dC_07);
			__m256 _dC_04 = _mm256_add_ps(_dC_00, _dC_01);
			__m256 _dC_05 = _mm256_add_ps(_dC_02, _dC_03);
			__m256 _dC_06 = _mm256_add_ps(_dC_04, _dC_05);

			__m256 dC_10 = _mm256_mul_ps(f_10, Ynm_10);
			__m256 dC_11 = _mm256_mul_ps(f_11, Ynm_11);
			__m256 dC_12 = _mm256_mul_ps(f_12, Ynm_12);
			__m256 dC_13 = _mm256_mul_ps(f_13, Ynm_13);
			__m256 dC_14 = _mm256_mul_ps(f_14, Ynm_14);
			__m256 dC_15 = _mm256_mul_ps(f_15, Ynm_15);
			__m256 dC_16 = _mm256_mul_ps(f_16, Ynm_16);
			__m256 dC_17 = _mm256_mul_ps(f_17, Ynm_17);
			__m256 _dC_10 = _mm256_add_ps(dC_10, dC_11);
			__m256 _dC_11 = _mm256_add_ps(dC_12, dC_13);
			__m256 _dC_12 = _mm256_add_ps(dC_14, dC_15);
			__m256 _dC_13 = _mm256_add_ps(dC_16, dC_17);
			__m256 _dC_14 = _mm256_add_ps(_dC_10, _dC_11);
			__m256 _dC_15 = _mm256_add_ps(_dC_12, _dC_13);
			__m256 _dC_16 = _mm256_add_ps(_dC_14, _dC_15);

			__m256 dC_20 = _mm256_mul_ps(f_20, Ynm_20);
			__m256 dC_21 = _mm256_mul_ps(f_21, Ynm_21);
			__m256 dC_22 = _mm256_mul_ps(f_22, Ynm_22);
			__m256 dC_23 = _mm256_mul_ps(f_23, Ynm_23);
			__m256 dC_24 = _mm256_mul_ps(f_24, Ynm_24);
			__m256 dC_25 = _mm256_mul_ps(f_25, Ynm_25);
			__m256 dC_26 = _mm256_mul_ps(f_26, Ynm_26);
			__m256 dC_27 = _mm256_mul_ps(f_27, Ynm_27);
			__m256 _dC_20 = _mm256_add_ps(dC_20, dC_21);
			__m256 _dC_21 = _mm256_add_ps(dC_22, dC_23);
			__m256 _dC_22 = _mm256_add_ps(dC_24, dC_25);
			__m256 _dC_23 = _mm256_add_ps(dC_26, dC_27);
			__m256 _dC_24 = _mm256_add_ps(_dC_20, _dC_21);
			__m256 _dC_25 = _mm256_add_ps(_dC_22, _dC_23);
			__m256 _dC_26 = _mm256_add_ps(_dC_24, _dC_25);

			__m256 dC_30 = _mm256_mul_ps(f_30, Ynm_30);
			__m256 dC_31 = _mm256_mul_ps(f_31, Ynm_31);
			__m256 dC_32 = _mm256_mul_ps(f_32, Ynm_32);
			__m256 dC_33 = _mm256_mul_ps(f_33, Ynm_33);
			__m256 dC_34 = _mm256_mul_ps(f_34, Ynm_34);
			__m256 dC_35 = _mm256_mul_ps(f_35, Ynm_35);
			__m256 dC_36 = _mm256_mul_ps(f_36, Ynm_36);
			__m256 dC_37 = _mm256_mul_ps(f_37, Ynm_37);
			__m256 _dC_30 = _mm256_add_ps(dC_30, dC_31);
			__m256 _dC_31 = _mm256_add_ps(dC_32, dC_33);
			__m256 _dC_32 = _mm256_add_ps(dC_34, dC_35);
			__m256 _dC_33 = _mm256_add_ps(dC_36, dC_37);
			__m256 _dC_34 = _mm256_add_ps(_dC_30, _dC_31);
			__m256 _dC_35 = _mm256_add_ps(_dC_32, _dC_33);
			__m256 _dC_36 = _mm256_add_ps(_dC_34, _dC_35);


			dCn_0 = _mm256_add_ps(dCn_0, _dC_06);
			dCn_1 = _mm256_add_ps(dCn_1, _dC_16);
			dCn_2 = _mm256_add_ps(dCn_2, _dC_26);
			dCn_3 = _mm256_add_ps(dCn_3, _dC_36);

		}

		__m256 uYx_0 = _mm256_mul_ps(mom, uY_0);
		__m256 uYx_1 = _mm256_mul_ps(mom, uY_1);
		__m256 uYx_2 = _mm256_mul_ps(mom, uY_2);
		__m256 uYx_3 = _mm256_mul_ps(mom, uY_3);

		__m256 uYxx_0 = _mm256_fmadd_ps(m_eta, dCn_0, uYx_0);
		__m256 uYxx_1 = _mm256_fmadd_ps(m_eta, dCn_1, uYx_1);
		__m256 uYxx_2 = _mm256_fmadd_ps(m_eta, dCn_2, uYx_2);
		__m256 uYxx_3 = _mm256_fmadd_ps(m_eta, dCn_3, uYx_3);

		_mm256_storeu_ps(uY+nD, uYxx_0);
		_mm256_storeu_ps(uY+nD+8, uYxx_1);
		_mm256_storeu_ps(uY+nD+16, uYxx_2);
		_mm256_storeu_ps(uY+nD+24, uYxx_3);

		nN += 4 * K * N;
		nD += 4 * K * D;
	}

    int n;
    const int B = 8;
    for (n = 0; n + 8*B <= N*D; n += 8*B) {
        __m256 uY_0 = _mm256_loadu_ps(uY+n);
        __m256 uY_1 = _mm256_loadu_ps(uY+n+8);
		__m256 uY_2 = _mm256_loadu_ps(uY+n+16);
		__m256 uY_3 = _mm256_loadu_ps(uY+n+24);
		__m256 uY_4 = _mm256_loadu_ps(uY+n+32);
		__m256 uY_5 = _mm256_loadu_ps(uY+n+40);
		__m256 uY_6 = _mm256_loadu_ps(uY+n+48);
		__m256 uY_7 = _mm256_loadu_ps(uY+n+56);

        __m256 Y_0 = _mm256_loadu_ps(Y+n);
        __m256 Y_1 = _mm256_loadu_ps(Y+n+8);
		__m256 Y_2 = _mm256_loadu_ps(Y+n+16);
		__m256 Y_3 = _mm256_loadu_ps(Y+n+24);
		__m256 Y_4 = _mm256_loadu_ps(Y+n+32);
		__m256 Y_5 = _mm256_loadu_ps(Y+n+40);
		__m256 Y_6 = _mm256_loadu_ps(Y+n+48);
		__m256 Y_7 = _mm256_loadu_ps(Y+n+56);

        __m256 Yx_0 = _mm256_add_ps(Y_0, uY_0);
		__m256 Yx_1 = _mm256_add_ps(Y_1, uY_1);
		__m256 Yx_2 = _mm256_add_ps(Y_2, uY_2);
		__m256 Yx_3 = _mm256_add_ps(Y_3, uY_3);
		__m256 Yx_4 = _mm256_add_ps(Y_4, uY_4);
		__m256 Yx_5 = _mm256_add_ps(Y_5, uY_5);
		__m256 Yx_6 = _mm256_add_ps(Y_6, uY_6);
		__m256 Yx_7 = _mm256_add_ps(Y_7, uY_7);

        _mm256_storeu_ps(Y+n, Yx_0);
		_mm256_storeu_ps(Y+n+8, Yx_1);
		_mm256_storeu_ps(Y+n+16, Yx_2);
		_mm256_storeu_ps(Y+n+24, Yx_3);
		_mm256_storeu_ps(Y+n+32, Yx_4);
		_mm256_storeu_ps(Y+n+40, Yx_5);
		_mm256_storeu_ps(Y+n+48, Yx_6);
		_mm256_storeu_ps(Y+n+56, Yx_7);
    }
    for (; n + 8 <= N*D; n += 8) {
        __m256 uY_ = _mm256_loadu_ps(uY+n);
        __m256 Y_ = _mm256_loadu_ps(Y+n);
        __m256 Yx_ = _mm256_add_ps(Y_, uY_);
        _mm256_storeu_ps(Y+n, Yx_);
    }
    for (; n < N*D; n++) {
        Y[n] += uY[n];
    }
}

inline void unfold_d_unfold_nx4_mx4_vec(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC, float* uY,
									float momentum, float eta) {
	const int M = 4;
	const int K = 4;

	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;

	float inv_sum_Q = 1 / sum_Q;
	__m256 inv_q = _mm256_set1_ps(inv_sum_Q);
	__m256 m_eta = _mm256_set1_ps(-eta);
	__m256 mom = _mm256_set1_ps(momentum);

	for(int n = 0; n < N; n += 4*K) {
		int mD = 0;

		__m256 dCn_0 = _mm256_setzero_ps();
		__m256 dCn_1 = _mm256_setzero_ps();
		__m256 dCn_2 = _mm256_setzero_ps();
		__m256 dCn_3 = _mm256_setzero_ps();

		__m256 Yn_0 = _mm256_loadu_ps(Y + nD);
		__m256 Yn_1 = _mm256_loadu_ps(Y + nD + 8);
		__m256 Yn_2 = _mm256_loadu_ps(Y + nD + 16);
		__m256 Yn_3 = _mm256_loadu_ps(Y + nD + 24);

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

			__m256 p_00 = _mm256_set_ps(P[nN+3*N+m],   P[nN+3*N+m],
									    P[nN+2*N+m],   P[nN+2*N+m],
									    P[nN+N+m],     P[nN+N+m],
									    P[nN+m],       P[nN+m]);
			__m256 p_01 = _mm256_set_ps(P[nN+3*N+m+1], P[nN+3*N+m+1],
									    P[nN+2*N+m+1], P[nN+2*N+m+1],
									    P[nN+N+m+1],   P[nN+N+m+1],
									    P[nN+m+1],     P[nN+m+1]);
			__m256 p_02 = _mm256_set_ps(P[nN+3*N+m+2], P[nN+3*N+m+2],
									    P[nN+2*N+m+2], P[nN+2*N+m+2],
									    P[nN+N+m+2],   P[nN+N+m+2],
									    P[nN+m+2],     P[nN+m+2]);
			__m256 p_03 = _mm256_set_ps(P[nN+3*N+m+3], P[nN+3*N+m+3],
									    P[nN+2*N+m+3], P[nN+2*N+m+3],
									    P[nN+N+m+3],   P[nN+N+m+3],
									    P[nN+m+3],     P[nN+m+3]);

			__m256 p_10 = _mm256_set_ps(P[nN+7*N+m],   P[nN+7*N+m],
									    P[nN+6*N+m],   P[nN+6*N+m],
									    P[nN+5*N+m],   P[nN+5*N+m],
									    P[nN+4*N+m],   P[nN+4*N+m]);
			__m256 p_11 = _mm256_set_ps(P[nN+7*N+m+1], P[nN+7*N+m+1],
									    P[nN+6*N+m+1], P[nN+6*N+m+1],
									    P[nN+5*N+m+1], P[nN+5*N+m+1],
									    P[nN+4*N+m+1], P[nN+4*N+m+1]);
			__m256 p_12 = _mm256_set_ps(P[nN+7*N+m+2], P[nN+7*N+m+2],
									    P[nN+6*N+m+2], P[nN+6*N+m+2],
									    P[nN+5*N+m+2], P[nN+5*N+m+2],
									    P[nN+4*N+m+2], P[nN+4*N+m+2]);
			__m256 p_13 = _mm256_set_ps(P[nN+7*N+m+3], P[nN+7*N+m+3],
									    P[nN+6*N+m+3], P[nN+6*N+m+3],
									    P[nN+5*N+m+3], P[nN+5*N+m+3],
									    P[nN+4*N+m+3], P[nN+4*N+m+3]);

			__m256 p_20 = _mm256_set_ps(P[nN+11*N+m],  P[nN+11*N+m],
									    P[nN+10*N+m],  P[nN+10*N+m],
									    P[nN+9*N+m],   P[nN+9*N+m],
									    P[nN+8*N+m],   P[nN+8*N+m]);
			__m256 p_21 = _mm256_set_ps(P[nN+11*N+m+1],P[nN+11*N+m+1],
									    P[nN+10*N+m+1],P[nN+10*N+m+1],
									    P[nN+9*N+m+1], P[nN+9*N+m+1],
									    P[nN+8*N+m+1], P[nN+8*N+m+1]);
			__m256 p_22 = _mm256_set_ps(P[nN+11*N+m+2],P[nN+11*N+m+2],
									    P[nN+10*N+m+2],P[nN+10*N+m+2],
									    P[nN+9*N+m+2], P[nN+9*N+m+2],
									    P[nN+8*N+m+2], P[nN+8*N+m+2]);
			__m256 p_23 = _mm256_set_ps(P[nN+11*N+m+3],P[nN+11*N+m+3],
									    P[nN+10*N+m+3],P[nN+10*N+m+3],
									    P[nN+9*N+m+3], P[nN+9*N+m+3],
									    P[nN+8*N+m+3], P[nN+8*N+m+3]);

			__m256 p_30 = _mm256_set_ps(P[nN+15*N+m],  P[nN+15*N+m],
									    P[nN+14*N+m],  P[nN+14*N+m],
									    P[nN+13*N+m],  P[nN+13*N+m],
									    P[nN+12*N+m],  P[nN+12*N+m]);
			__m256 p_31 = _mm256_set_ps(P[nN+15*N+m+1],P[nN+15*N+m+1],
									    P[nN+14*N+m+1],P[nN+14*N+m+1],
									    P[nN+13*N+m+1],P[nN+13*N+m+1],
									    P[nN+12*N+m+1],P[nN+12*N+m+1]);
			__m256 p_32 = _mm256_set_ps(P[nN+15*N+m+2],P[nN+15*N+m+2],
									    P[nN+14*N+m+2],P[nN+14*N+m+2],
									    P[nN+13*N+m+2],P[nN+13*N+m+2],
									    P[nN+12*N+m+2],P[nN+12*N+m+2]);
			__m256 p_33 = _mm256_set_ps(P[nN+15*N+m+3],P[nN+15*N+m+3],
									    P[nN+14*N+m+3],P[nN+14*N+m+3],
									    P[nN+13*N+m+3],P[nN+13*N+m+3],
									    P[nN+12*N+m+3],P[nN+12*N+m+3]);

			__m256 q_00 = _mm256_set_ps(Q[nN+3*N+m],   Q[nN+3*N+m],
									    Q[nN+2*N+m],   Q[nN+2*N+m],
									    Q[nN+N+m],     Q[nN+N+m],
									    Q[nN+m],       Q[nN+m]);
			__m256 q_01 = _mm256_set_ps(Q[nN+3*N+m+1], Q[nN+3*N+m+1],
									    Q[nN+2*N+m+1], Q[nN+2*N+m+1],
									    Q[nN+N+m+1],   Q[nN+N+m+1],
									    Q[nN+m+1],     Q[nN+m+1]);
			__m256 q_02 = _mm256_set_ps(Q[nN+3*N+m+2], Q[nN+3*N+m+2],
									    Q[nN+2*N+m+2], Q[nN+2*N+m+2],
									    Q[nN+N+m+2],   Q[nN+N+m+2],
									    Q[nN+m+2],     Q[nN+m+2]);
			__m256 q_03 = _mm256_set_ps(Q[nN+3*N+m+3], Q[nN+3*N+m+3],
									    Q[nN+2*N+m+3], Q[nN+2*N+m+3],
									    Q[nN+N+m+3],   Q[nN+N+m+3],
									    Q[nN+m+3],     Q[nN+m+3]);

			__m256 q_10 = _mm256_set_ps(Q[nN+7*N+m],   Q[nN+7*N+m],
									    Q[nN+6*N+m],   Q[nN+6*N+m],
									    Q[nN+5*N+m],   Q[nN+5*N+m],
									    Q[nN+4*N+m],   Q[nN+4*N+m]);
			__m256 q_11 = _mm256_set_ps(Q[nN+7*N+m+1], Q[nN+7*N+m+1],
									    Q[nN+6*N+m+1], Q[nN+6*N+m+1],
									    Q[nN+5*N+m+1], Q[nN+5*N+m+1],
									    Q[nN+4*N+m+1], Q[nN+4*N+m+1]);
			__m256 q_12 = _mm256_set_ps(Q[nN+7*N+m+2], Q[nN+7*N+m+2],
									    Q[nN+6*N+m+2], Q[nN+6*N+m+2],
									    Q[nN+5*N+m+2], Q[nN+5*N+m+2],
									    Q[nN+4*N+m+2], Q[nN+4*N+m+2]);
			__m256 q_13 = _mm256_set_ps(Q[nN+7*N+m+3], Q[nN+7*N+m+3],
									    Q[nN+6*N+m+3], Q[nN+6*N+m+3],
									    Q[nN+5*N+m+3], Q[nN+5*N+m+3],
									    Q[nN+4*N+m+3], Q[nN+4*N+m+3]);

			__m256 q_20 = _mm256_set_ps(Q[nN+11*N+m],  Q[nN+11*N+m],
									    Q[nN+10*N+m],  Q[nN+10*N+m],
									    Q[nN+9*N+m],   Q[nN+9*N+m],
									    Q[nN+8*N+m],   Q[nN+8*N+m]);
			__m256 q_21 = _mm256_set_ps(Q[nN+11*N+m+1],Q[nN+11*N+m+1],
									    Q[nN+10*N+m+1],Q[nN+10*N+m+1],
									    Q[nN+9*N+m+1], Q[nN+9*N+m+1],
									    Q[nN+8*N+m+1], Q[nN+8*N+m+1]);
			__m256 q_22 = _mm256_set_ps(Q[nN+11*N+m+2],Q[nN+11*N+m+2],
									    Q[nN+10*N+m+2],Q[nN+10*N+m+2],
									    Q[nN+9*N+m+2], Q[nN+9*N+m+2],
									    Q[nN+8*N+m+2], Q[nN+8*N+m+2]);
			__m256 q_23 = _mm256_set_ps(Q[nN+11*N+m+3],Q[nN+11*N+m+3],
									    Q[nN+10*N+m+3],Q[nN+10*N+m+3],
									    Q[nN+9*N+m+3], Q[nN+9*N+m+3],
									    Q[nN+8*N+m+3], Q[nN+8*N+m+3]);

			__m256 q_30 = _mm256_set_ps(Q[nN+15*N+m],  Q[nN+15*N+m],
									    Q[nN+14*N+m],  Q[nN+14*N+m],
									    Q[nN+13*N+m],  Q[nN+13*N+m],
									    Q[nN+12*N+m],  Q[nN+12*N+m]);
			__m256 q_31 = _mm256_set_ps(Q[nN+15*N+m+1],Q[nN+15*N+m+1],
									    Q[nN+14*N+m+1],Q[nN+14*N+m+1],
									    Q[nN+13*N+m+1],Q[nN+13*N+m+1],
									    Q[nN+12*N+m+1],Q[nN+12*N+m+1]);
			__m256 q_32 = _mm256_set_ps(Q[nN+15*N+m+2],Q[nN+15*N+m+2],
									    Q[nN+14*N+m+2],Q[nN+14*N+m+2],
									    Q[nN+13*N+m+2],Q[nN+13*N+m+2],
									    Q[nN+12*N+m+2],Q[nN+12*N+m+2]);
			__m256 q_33 = _mm256_set_ps(Q[nN+15*N+m+3],Q[nN+15*N+m+3],
									    Q[nN+14*N+m+3],Q[nN+14*N+m+3],
									    Q[nN+13*N+m+3],Q[nN+13*N+m+3],
									    Q[nN+12*N+m+3],Q[nN+12*N+m+3]);


			__m256 Ynm_00 = _mm256_sub_ps(Yn_0, Ym_0);
			__m256 Ynm_01 = _mm256_sub_ps(Yn_0, Ym_1);
			__m256 Ynm_02 = _mm256_sub_ps(Yn_0, Ym_2);
			__m256 Ynm_03 = _mm256_sub_ps(Yn_0, Ym_3);

			__m256 Ynm_10 = _mm256_sub_ps(Yn_1, Ym_0);
			__m256 Ynm_11 = _mm256_sub_ps(Yn_1, Ym_1);
			__m256 Ynm_12 = _mm256_sub_ps(Yn_1, Ym_2);
			__m256 Ynm_13 = _mm256_sub_ps(Yn_1, Ym_3);

			__m256 Ynm_20 = _mm256_sub_ps(Yn_2, Ym_0);
			__m256 Ynm_21 = _mm256_sub_ps(Yn_2, Ym_1);
			__m256 Ynm_22 = _mm256_sub_ps(Yn_2, Ym_2);
			__m256 Ynm_23 = _mm256_sub_ps(Yn_2, Ym_3);

			__m256 Ynm_30 = _mm256_sub_ps(Yn_3, Ym_0);
			__m256 Ynm_31 = _mm256_sub_ps(Yn_3, Ym_1);
			__m256 Ynm_32 = _mm256_sub_ps(Yn_3, Ym_2);
			__m256 Ynm_33 = _mm256_sub_ps(Yn_3, Ym_3);



			__m256 qq_00 = _mm256_mul_ps(q_00, inv_q);
			__m256 qq_01 = _mm256_mul_ps(q_01, inv_q);
			__m256 qq_02 = _mm256_mul_ps(q_02, inv_q);
			__m256 qq_03 = _mm256_mul_ps(q_03, inv_q);

			__m256 qq_10 = _mm256_mul_ps(q_10, inv_q);
			__m256 qq_11 = _mm256_mul_ps(q_11, inv_q);
			__m256 qq_12 = _mm256_mul_ps(q_12, inv_q);
			__m256 qq_13 = _mm256_mul_ps(q_13, inv_q);

			__m256 qq_20 = _mm256_mul_ps(q_20, inv_q);
			__m256 qq_21 = _mm256_mul_ps(q_21, inv_q);
			__m256 qq_22 = _mm256_mul_ps(q_22, inv_q);
			__m256 qq_23 = _mm256_mul_ps(q_23, inv_q);

			__m256 qq_30 = _mm256_mul_ps(q_30, inv_q);
			__m256 qq_31 = _mm256_mul_ps(q_31, inv_q);
			__m256 qq_32 = _mm256_mul_ps(q_32, inv_q);
			__m256 qq_33 = _mm256_mul_ps(q_33, inv_q);


			__m256 pqq_00 = _mm256_sub_ps(p_00, qq_00);
			__m256 pqq_01 = _mm256_sub_ps(p_01, qq_01);
			__m256 pqq_02 = _mm256_sub_ps(p_02, qq_02);
			__m256 pqq_03 = _mm256_sub_ps(p_03, qq_03);

			__m256 pqq_10 = _mm256_sub_ps(p_10, qq_10);
			__m256 pqq_11 = _mm256_sub_ps(p_11, qq_11);
			__m256 pqq_12 = _mm256_sub_ps(p_12, qq_12);
			__m256 pqq_13 = _mm256_sub_ps(p_13, qq_13);

			__m256 pqq_20 = _mm256_sub_ps(p_20, qq_20);
			__m256 pqq_21 = _mm256_sub_ps(p_21, qq_21);
			__m256 pqq_22 = _mm256_sub_ps(p_22, qq_22);
			__m256 pqq_23 = _mm256_sub_ps(p_23, qq_23);

			__m256 pqq_30 = _mm256_sub_ps(p_30, qq_30);
			__m256 pqq_31 = _mm256_sub_ps(p_31, qq_31);
			__m256 pqq_32 = _mm256_sub_ps(p_32, qq_32);
			__m256 pqq_33 = _mm256_sub_ps(p_33, qq_33);



			__m256 yq_00 = _mm256_mul_ps(Ynm_00, q_00);
			__m256 yq_01 = _mm256_mul_ps(Ynm_01, q_01);
			__m256 yq_02 = _mm256_mul_ps(Ynm_02, q_02);
			__m256 yq_03 = _mm256_mul_ps(Ynm_03, q_03);

			__m256 yq_10 = _mm256_mul_ps(Ynm_10, q_10);
			__m256 yq_11 = _mm256_mul_ps(Ynm_11, q_11);
			__m256 yq_12 = _mm256_mul_ps(Ynm_12, q_12);
			__m256 yq_13 = _mm256_mul_ps(Ynm_13, q_13);

			__m256 yq_20 = _mm256_mul_ps(Ynm_20, q_20);
			__m256 yq_21 = _mm256_mul_ps(Ynm_21, q_21);
			__m256 yq_22 = _mm256_mul_ps(Ynm_22, q_22);
			__m256 yq_23 = _mm256_mul_ps(Ynm_23, q_23);

			__m256 yq_30 = _mm256_mul_ps(Ynm_30, q_30);
			__m256 yq_31 = _mm256_mul_ps(Ynm_31, q_31);
			__m256 yq_32 = _mm256_mul_ps(Ynm_32, q_32);
			__m256 yq_33 = _mm256_mul_ps(Ynm_33, q_33);


			__m256 dC_00 = _mm256_mul_ps(pqq_00, yq_00);
			__m256 dC_01 = _mm256_mul_ps(pqq_01, yq_01);
			__m256 dC_02 = _mm256_mul_ps(pqq_02, yq_02);
			__m256 dC_03 = _mm256_mul_ps(pqq_03, yq_03);
			__m256 dC_04 = _mm256_add_ps(dC_00, dC_01);
			__m256 dC_05 = _mm256_add_ps(dC_02, dC_03);
			__m256 dC_06 = _mm256_add_ps(dC_04, dC_05);

			__m256 dC_10 = _mm256_mul_ps(pqq_10, yq_10);
			__m256 dC_11 = _mm256_mul_ps(pqq_11, yq_11);
			__m256 dC_12 = _mm256_mul_ps(pqq_12, yq_12);
			__m256 dC_13 = _mm256_mul_ps(pqq_13, yq_13);
			__m256 dC_14 = _mm256_add_ps(dC_10, dC_11);
			__m256 dC_15 = _mm256_add_ps(dC_12, dC_13);
			__m256 dC_16 = _mm256_add_ps(dC_14, dC_15);

			__m256 dC_20 = _mm256_mul_ps(pqq_20, yq_20);
			__m256 dC_21 = _mm256_mul_ps(pqq_21, yq_21);
			__m256 dC_22 = _mm256_mul_ps(pqq_22, yq_22);
			__m256 dC_23 = _mm256_mul_ps(pqq_23, yq_23);
			__m256 dC_24 = _mm256_add_ps(dC_20, dC_21);
			__m256 dC_25 = _mm256_add_ps(dC_22, dC_23);
			__m256 dC_26 = _mm256_add_ps(dC_24, dC_25);

			__m256 dC_30 = _mm256_mul_ps(pqq_30, yq_30);
			__m256 dC_31 = _mm256_mul_ps(pqq_31, yq_31);
			__m256 dC_32 = _mm256_mul_ps(pqq_32, yq_32);
			__m256 dC_33 = _mm256_mul_ps(pqq_33, yq_33);
			__m256 dC_34 = _mm256_add_ps(dC_30, dC_31);
			__m256 dC_35 = _mm256_add_ps(dC_32, dC_33);
			__m256 dC_36 = _mm256_add_ps(dC_34, dC_35);

			dCn_0 = _mm256_add_ps(dCn_0, dC_06);
			dCn_1 = _mm256_add_ps(dCn_1, dC_16);
			dCn_2 = _mm256_add_ps(dCn_2, dC_26);
			dCn_3 = _mm256_add_ps(dCn_3, dC_36);

			mD += M * D;
		}

		__m256 uY_0 = _mm256_loadu_ps(uY+nD);
		__m256 uY_1 = _mm256_loadu_ps(uY+nD+8);
		__m256 uY_2 = _mm256_loadu_ps(uY+nD+16);
		__m256 uY_3 = _mm256_loadu_ps(uY+nD+24);

		__m256 uYx_0 = _mm256_mul_ps(mom, uY_0);
		__m256 uYx_1 = _mm256_mul_ps(mom, uY_1);
		__m256 uYx_2 = _mm256_mul_ps(mom, uY_2);
		__m256 uYx_3 = _mm256_mul_ps(mom, uY_3);

		__m256 uYxx_0 = _mm256_fmadd_ps(m_eta, dCn_0, uYx_0);
		__m256 uYxx_1 = _mm256_fmadd_ps(m_eta, dCn_1, uYx_1);
		__m256 uYxx_2 = _mm256_fmadd_ps(m_eta, dCn_2, uYx_2);
		__m256 uYxx_3 = _mm256_fmadd_ps(m_eta, dCn_3, uYx_3);

		_mm256_storeu_ps(uY+nD, uYxx_0);
		_mm256_storeu_ps(uY+nD+8, uYxx_1);
		_mm256_storeu_ps(uY+nD+16, uYxx_2);
		_mm256_storeu_ps(uY+nD+24, uYxx_3);

		nN += 4 * K * N;
		nD += 4 * K * D;
	}



	for (int n = 0; n < N*D; n += 8 * K) {
		__m256 uY_0 = _mm256_loadu_ps(uY+n);
		__m256 uY_1 = _mm256_loadu_ps(uY+n+8);
		__m256 uY_2 = _mm256_loadu_ps(uY+n+16);
		__m256 uY_3 = _mm256_loadu_ps(uY+n+24);

		__m256 Y_0 = _mm256_loadu_ps(Y+n);
		__m256 Y_1 = _mm256_loadu_ps(Y+n+8);
		__m256 Y_2 = _mm256_loadu_ps(Y+n+16);
		__m256 Y_3 = _mm256_loadu_ps(Y+n+24);

		__m256 Yx_0 = _mm256_add_ps(Y_0, uY_0);
		__m256 Yx_1 = _mm256_add_ps(Y_1, uY_1);
		__m256 Yx_2 = _mm256_add_ps(Y_2, uY_2);
		__m256 Yx_3 = _mm256_add_ps(Y_3, uY_3);

		_mm256_storeu_ps(Y+n, Yx_0);
		_mm256_storeu_ps(Y+n+8, Yx_1);
		_mm256_storeu_ps(Y+n+16, Yx_2);
		_mm256_storeu_ps(Y+n+24, Yx_3);
	}
}

inline void unfold_d_unfold_mx8_vec(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC, float* uY, float momentum, float eta) {
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
									int N, int D, float* dC, float* uY,
									float momentum, float eta) {
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

	__m256 m_eta = _mm256_set1_ps(-eta);
	__m256 mom = _mm256_set1_ps(momentum);

	const int K = 4; // Number of accumulators
	for (int n = 0; n < N*D; n += 8 * K) {
		__m256 uY_0 = _mm256_loadu_ps(uY+n);
		__m256 uY_1 = _mm256_loadu_ps(uY+n+8);
		__m256 uY_2 = _mm256_loadu_ps(uY+n+16);
		__m256 uY_3 = _mm256_loadu_ps(uY+n+24);

		__m256 dC_0 = _mm256_loadu_ps(dC+n);
		__m256 dC_1 = _mm256_loadu_ps(dC+n+8);
		__m256 dC_2 = _mm256_loadu_ps(dC+n+16);
		__m256 dC_3 = _mm256_loadu_ps(dC+n+24);

		__m256 Y_0 = _mm256_loadu_ps(Y+n);
		__m256 Y_1 = _mm256_loadu_ps(Y+n+8);
		__m256 Y_2 = _mm256_loadu_ps(Y+n+16);
		__m256 Y_3 = _mm256_loadu_ps(Y+n+24);

		__m256 uYx_0 = _mm256_mul_ps(mom, uY_0);
		__m256 uYx_1 = _mm256_mul_ps(mom, uY_1);
		__m256 uYx_2 = _mm256_mul_ps(mom, uY_2);
		__m256 uYx_3 = _mm256_mul_ps(mom, uY_3);

		__m256 uYxx_0 = _mm256_fmadd_ps(m_eta, dC_0, uYx_0);
		__m256 uYxx_1 = _mm256_fmadd_ps(m_eta, dC_1, uYx_1);
		__m256 uYxx_2 = _mm256_fmadd_ps(m_eta, dC_2, uYx_2);
		__m256 uYxx_3 = _mm256_fmadd_ps(m_eta, dC_3, uYx_3);

		__m256 Yx_0 = _mm256_add_ps(Y_0, uYxx_0);
		__m256 Yx_1 = _mm256_add_ps(Y_1, uYxx_1);
		__m256 Yx_2 = _mm256_add_ps(Y_2, uYxx_2);
		__m256 Yx_3 = _mm256_add_ps(Y_3, uYxx_3);

		_mm256_storeu_ps(uY+n, uYxx_0);
		_mm256_storeu_ps(uY+n+8, uYxx_1);
		_mm256_storeu_ps(uY+n+16, uYxx_2);
		_mm256_storeu_ps(uY+n+24, uYxx_3);

		_mm256_storeu_ps(Y+n, Yx_0);
		_mm256_storeu_ps(Y+n+8, Yx_1);
		_mm256_storeu_ps(Y+n+16, Yx_2);
		_mm256_storeu_ps(Y+n+24, Yx_3);
	}
}

inline void unfold_d_unfold_mx8(float* Y, float* P, float* Q, float sum_Q,
								int N, int D, float* dC, float* uY,
								float momentum, float eta) {
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

		uY[nD] = momentum * uY[nD] - eta * dC_n_0;
		uY[nD + 1] = momentum * uY[nD + 1] - eta * dC_n_1;

		nN += N;
		nD += D;
	}

	nD = 0;
	for (int n = 0; n < N; n++, nD += D) {
		Y[nD] = Y[nD] + uY[nD];
		Y[nD+1] = Y[nD+1] + uY[nD+1];
	}
}

inline void unfold_d_unfold_mx4(float* Y, float* P, float* Q, float sum_Q,
						 int N, int D, float* dC, float* uY, float momentum, float eta) {
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
		uY[nD] = momentum * uY[nD] - eta * dC_n_0;
		uY[nD + 1] = momentum * uY[nD + 1] - eta * dC_n_1;
		nN += N;
		nD += D;
	}

	nD = 0;
	for (int n = 0; n < N; n++, nD += D) {
		Y[nD] = Y[nD] + uY[nD];
		Y[nD+1] = Y[nD+1] + uY[nD+1];
	}
}

inline void unfold_d(float* Y, float* P, float* Q, float sum_Q,
						 int N, int D, float* dC, float* uY, float momentum, float eta) {
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
		uY[nD] = momentum * uY[nD] - eta * dC_n_0;
		uY[nD + 1] = momentum * uY[nD + 1] - eta * dC_n_1;

		nN += N;
		nD += D;
	}

	nD = 0;
	for (int n = 0; n < N; n++, nD += D) {
		Y[nD] = Y[nD] + uY[nD];
		Y[nD+1] = Y[nD+1] + uY[nD+1];
	}
}

// Gradient computation dC_dy
inline void gradient_computation(float* Y, float* P, float* Q, float sum_Q,
								 int N, int D, float* dC) {
	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		int mD = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				float mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for (int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
			}
			mD += D;
		}
		nN += N;
		nD += D;
	}
}

inline void gradient_update(float* Y, float* dC, float* uY, int N,
							int no_dims, float momentum, float eta){
	// Perform gradient update
	for(int i = 0; i < N * no_dims; i++)
		uY[i] = momentum * uY[i] - eta * dC[i];
	for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
}


inline void base_version(float* Y, float* P, float* Q, float sum_Q, int N,
						 int D, float* dC, float* uY, float momentum,
						 float eta) {
	gradient_computation(Y, P, Q, sum_Q, N, D, dC);
	gradient_update(Y, dC, uY, N, D, momentum, eta);
}

#endif
