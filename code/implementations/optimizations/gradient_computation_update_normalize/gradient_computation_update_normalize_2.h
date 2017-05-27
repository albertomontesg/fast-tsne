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


inline void unfold_accumulators_avx(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC, float* uY,
									float momentum, float eta) {

	// Perform the computation of the gradient
	float inv_sum_Q = 1 / sum_Q;
	__m256 inv_q = _mm256_set1_ps(inv_sum_Q);
	__m256 m_eta = _mm256_set1_ps(-eta);
	__m256 mom = _mm256_set1_ps(momentum);

    int n, nD, nN;
	for(n = 0, nD = 0, nN = 0; n + 16 <= N; n += 4*4, nD += 4*4*D, nN += 4*4*N) {

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

        int m, mD;
		for (m = 0, mD = 0; m + 8 <= N; m += 8, mD += 8*D) {
            __m256 Ym_00 = _mm256_broadcast_ss(Y+mD);
            __m256 Ym_01 = _mm256_broadcast_ss(Y+mD+1);
            __m256 Ym_10 = _mm256_broadcast_ss(Y+mD+2);
            __m256 Ym_11 = _mm256_broadcast_ss(Y+mD+3);
            __m256 Ym_20 = _mm256_broadcast_ss(Y+mD+4);
            __m256 Ym_21 = _mm256_broadcast_ss(Y+mD+5);
            __m256 Ym_30 = _mm256_broadcast_ss(Y+mD+6);
            __m256 Ym_31 = _mm256_broadcast_ss(Y+mD+7);
            __m256 Ym_40 = _mm256_broadcast_ss(Y+mD+8);
            __m256 Ym_41 = _mm256_broadcast_ss(Y+mD+9);
            __m256 Ym_50 = _mm256_broadcast_ss(Y+mD+10);
            __m256 Ym_51 = _mm256_broadcast_ss(Y+mD+11);
            __m256 Ym_60 = _mm256_broadcast_ss(Y+mD+12);
            __m256 Ym_61 = _mm256_broadcast_ss(Y+mD+13);
            __m256 Ym_70 = _mm256_broadcast_ss(Y+mD+14);
            __m256 Ym_71 = _mm256_broadcast_ss(Y+mD+15);

            __m256 Ym_0 = _mm256_blend_ps(Ym_00, Ym_01, 0b10101010);
            __m256 Ym_1 = _mm256_blend_ps(Ym_10, Ym_11, 0b10101010);
            __m256 Ym_2 = _mm256_blend_ps(Ym_20, Ym_21, 0b10101010);
            __m256 Ym_3 = _mm256_blend_ps(Ym_30, Ym_31, 0b10101010);
            __m256 Ym_4 = _mm256_blend_ps(Ym_40, Ym_41, 0b10101010);
            __m256 Ym_5 = _mm256_blend_ps(Ym_50, Ym_51, 0b10101010);
            __m256 Ym_6 = _mm256_blend_ps(Ym_60, Ym_61, 0b10101010);
            __m256 Ym_7 = _mm256_blend_ps(Ym_70, Ym_71, 0b10101010);

			// Load P

			__m256 P_00 = _mm256_loadu_ps(P+nN+m);
			__m256 P_01 = _mm256_loadu_ps(P+nN+N+m);
			__m256 P_02 = _mm256_loadu_ps(P+nN+2*N+m);
			__m256 P_03 = _mm256_loadu_ps(P+nN+3*N+m);

			__m256 P_10 = _mm256_loadu_ps(P+nN+4*N+m);
			__m256 P_11 = _mm256_loadu_ps(P+nN+5*N+m);
			__m256 P_12 = _mm256_loadu_ps(P+nN+6*N+m);
			__m256 P_13 = _mm256_loadu_ps(P+nN+7*N+m);

			__m256 P_20 = _mm256_loadu_ps(P+nN+8*N+m);
			__m256 P_21 = _mm256_loadu_ps(P+nN+9*N+m);
			__m256 P_22 = _mm256_loadu_ps(P+nN+10*N+m);
			__m256 P_23 = _mm256_loadu_ps(P+nN+11*N+m);

			__m256 P_30 = _mm256_loadu_ps(P+nN+12*N+m);
			__m256 P_31 = _mm256_loadu_ps(P+nN+13*N+m);
			__m256 P_32 = _mm256_loadu_ps(P+nN+14*N+m);
			__m256 P_33 = _mm256_loadu_ps(P+nN+15*N+m);
			// Load Q
			__m256 T_00 = _mm256_loadu_ps(Q+nN+m);
			__m256 T_01 = _mm256_loadu_ps(Q+nN+N+m);
			__m256 T_02 = _mm256_loadu_ps(Q+nN+2*N+m);
			__m256 T_03 = _mm256_loadu_ps(Q+nN+3*N+m);

			__m256 T_10 = _mm256_loadu_ps(Q+nN+4*N+m);
			__m256 T_11 = _mm256_loadu_ps(Q+nN+5*N+m);
			__m256 T_12 = _mm256_loadu_ps(Q+nN+6*N+m);
			__m256 T_13 = _mm256_loadu_ps(Q+nN+7*N+m);

			__m256 T_20 = _mm256_loadu_ps(Q+nN+8*N+m);
			__m256 T_21 = _mm256_loadu_ps(Q+nN+9*N+m);
			__m256 T_22 = _mm256_loadu_ps(Q+nN+10*N+m);
			__m256 T_23 = _mm256_loadu_ps(Q+nN+11*N+m);

			__m256 T_30 = _mm256_loadu_ps(Q+nN+12*N+m);
			__m256 T_31 = _mm256_loadu_ps(Q+nN+13*N+m);
			__m256 T_32 = _mm256_loadu_ps(Q+nN+14*N+m);
			__m256 T_33 = _mm256_loadu_ps(Q+nN+15*N+m);

			// Compute (q_ij)
			__m256 Q_00 = _mm256_mul_ps(T_00, inv_q);
			__m256 Q_01 = _mm256_mul_ps(T_01, inv_q);
			__m256 Q_02 = _mm256_mul_ps(T_02, inv_q);
			__m256 Q_03 = _mm256_mul_ps(T_03, inv_q);

			__m256 Q_10 = _mm256_mul_ps(T_10, inv_q);
			__m256 Q_11 = _mm256_mul_ps(T_11, inv_q);
			__m256 Q_12 = _mm256_mul_ps(T_12, inv_q);
			__m256 Q_13 = _mm256_mul_ps(T_13, inv_q);

			__m256 Q_20 = _mm256_mul_ps(T_20, inv_q);
			__m256 Q_21 = _mm256_mul_ps(T_21, inv_q);
			__m256 Q_22 = _mm256_mul_ps(T_22, inv_q);
			__m256 Q_23 = _mm256_mul_ps(T_23, inv_q);

			__m256 Q_30 = _mm256_mul_ps(T_30, inv_q);
			__m256 Q_31 = _mm256_mul_ps(T_31, inv_q);
			__m256 Q_32 = _mm256_mul_ps(T_32, inv_q);
			__m256 Q_33 = _mm256_mul_ps(T_33, inv_q);
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
			__m256 PQ_00 = _mm256_sub_ps(P_00, Q_00);
			__m256 PQ_01 = _mm256_sub_ps(P_01, Q_01);
			__m256 PQ_02 = _mm256_sub_ps(P_02, Q_02);
			__m256 PQ_03 = _mm256_sub_ps(P_03, Q_03);

			__m256 PQ_10 = _mm256_sub_ps(P_10, Q_10);
			__m256 PQ_11 = _mm256_sub_ps(P_11, Q_11);
			__m256 PQ_12 = _mm256_sub_ps(P_12, Q_12);
			__m256 PQ_13 = _mm256_sub_ps(P_13, Q_13);

			__m256 PQ_20 = _mm256_sub_ps(P_20, Q_20);
			__m256 PQ_21 = _mm256_sub_ps(P_21, Q_21);
			__m256 PQ_22 = _mm256_sub_ps(P_22, Q_22);
			__m256 PQ_23 = _mm256_sub_ps(P_23, Q_23);

			__m256 PQ_30 = _mm256_sub_ps(P_30, Q_30);
			__m256 PQ_31 = _mm256_sub_ps(P_31, Q_31);
			__m256 PQ_32 = _mm256_sub_ps(P_32, Q_32);
			__m256 PQ_33 = _mm256_sub_ps(P_33, Q_33);
			// }

            // Compute (p_ij - q_ij)(1-|y_i-y_j|^2)^-1
            __m256 pqq_00 = _mm256_mul_ps(PQ_00, T_00);
			__m256 pqq_01 = _mm256_mul_ps(PQ_01, T_01);
			__m256 pqq_02 = _mm256_mul_ps(PQ_02, T_02);
			__m256 pqq_03 = _mm256_mul_ps(PQ_03, T_03);

			__m256 pqq_10 = _mm256_mul_ps(PQ_10, T_10);
			__m256 pqq_11 = _mm256_mul_ps(PQ_11, T_11);
			__m256 pqq_12 = _mm256_mul_ps(PQ_12, T_12);
			__m256 pqq_13 = _mm256_mul_ps(PQ_13, T_13);

			__m256 pqq_20 = _mm256_mul_ps(PQ_20, T_20);
			__m256 pqq_21 = _mm256_mul_ps(PQ_21, T_21);
			__m256 pqq_22 = _mm256_mul_ps(PQ_22, T_22);
			__m256 pqq_23 = _mm256_mul_ps(PQ_23, T_23);

			__m256 pqq_30 = _mm256_mul_ps(PQ_30, T_30);
			__m256 pqq_31 = _mm256_mul_ps(PQ_31, T_31);
			__m256 pqq_32 = _mm256_mul_ps(PQ_32, T_32);
			__m256 pqq_33 = _mm256_mul_ps(PQ_33, T_33);


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

        for (; m < N; m++, mD += D) {
            __m256 Ym_0_0 = _mm256_broadcast_ss(Y+mD);
            __m256 Ym_0_1 = _mm256_broadcast_ss(Y+mD+1);
            __m256 Ym_0 = _mm256_blend_ps(Ym_0_0, Ym_0_1, 0b10101010);

            __m256 P_00 = _mm256_broadcast_ss(P+nN+m);
    		__m256 P_01 = _mm256_broadcast_ss(P+nN+N+m);
    		__m256 P_02 = _mm256_broadcast_ss(P+nN+2*N+m);
    		__m256 P_03 = _mm256_broadcast_ss(P+nN+3*N+m);

    		__m256 P_10 = _mm256_broadcast_ss(P+nN+4*N+m);
    		__m256 P_11 = _mm256_broadcast_ss(P+nN+5*N+m);
    		__m256 P_12 = _mm256_broadcast_ss(P+nN+6*N+m);
    		__m256 P_13 = _mm256_broadcast_ss(P+nN+7*N+m);

    		__m256 P_20 = _mm256_broadcast_ss(P+nN+8*N+m);
    		__m256 P_21 = _mm256_broadcast_ss(P+nN+9*N+m);
    		__m256 P_22 = _mm256_broadcast_ss(P+nN+10*N+m);
    		__m256 P_23 = _mm256_broadcast_ss(P+nN+11*N+m);

    		__m256 P_30 = _mm256_broadcast_ss(P+nN+12*N+m);
    		__m256 P_31 = _mm256_broadcast_ss(P+nN+13*N+m);
    		__m256 P_32 = _mm256_broadcast_ss(P+nN+14*N+m);
    		__m256 P_33 = _mm256_broadcast_ss(P+nN+15*N+m);

            __m256 P_0_lo = _mm256_blend_ps(P_00, P_01, 0b00001100);
            __m256 P_0_hi = _mm256_blend_ps(P_02, P_03, 0b11000000);
            __m256 P_0 = _mm256_blend_ps(P_0_lo, P_0_hi, 0b11110000);
            __m256 P_1_lo = _mm256_blend_ps(P_10, P_11, 0b00001100);
            __m256 P_1_hi = _mm256_blend_ps(P_12, P_13, 0b11000000);
            __m256 P_1 = _mm256_blend_ps(P_1_lo, P_1_hi, 0b11110000);
            __m256 P_2_lo = _mm256_blend_ps(P_20, P_21, 0b00001100);
            __m256 P_2_hi = _mm256_blend_ps(P_22, P_23, 0b11000000);
            __m256 P_2 = _mm256_blend_ps(P_2_lo, P_2_hi, 0b11110000);
            __m256 P_3_lo = _mm256_blend_ps(P_30, P_31, 0b00001100);
            __m256 P_3_hi = _mm256_blend_ps(P_32, P_33, 0b11000000);
            __m256 P_3 = _mm256_blend_ps(P_3_lo, P_3_hi, 0b11110000);

            __m256 T_00 = _mm256_broadcast_ss(Q+nN+m);
    		__m256 T_01 = _mm256_broadcast_ss(Q+nN+N+m);
    		__m256 T_02 = _mm256_broadcast_ss(Q+nN+2*N+m);
    		__m256 T_03 = _mm256_broadcast_ss(Q+nN+3*N+m);

    		__m256 T_10 = _mm256_broadcast_ss(Q+nN+4*N+m);
    		__m256 T_11 = _mm256_broadcast_ss(Q+nN+5*N+m);
    		__m256 T_12 = _mm256_broadcast_ss(Q+nN+6*N+m);
    		__m256 T_13 = _mm256_broadcast_ss(Q+nN+7*N+m);

    		__m256 T_20 = _mm256_broadcast_ss(Q+nN+8*N+m);
    		__m256 T_21 = _mm256_broadcast_ss(Q+nN+9*N+m);
    		__m256 T_22 = _mm256_broadcast_ss(Q+nN+10*N+m);
    		__m256 T_23 = _mm256_broadcast_ss(Q+nN+11*N+m);

    		__m256 T_30 = _mm256_broadcast_ss(Q+nN+12*N+m);
    		__m256 T_31 = _mm256_broadcast_ss(Q+nN+13*N+m);
    		__m256 T_32 = _mm256_broadcast_ss(Q+nN+14*N+m);
    		__m256 T_33 = _mm256_broadcast_ss(Q+nN+15*N+m);

            __m256 T_0_lo = _mm256_blend_ps(T_00, T_01, 0b00001100);
            __m256 T_0_hi = _mm256_blend_ps(T_02, T_03, 0b11000000);
            __m256 T_0 = _mm256_blend_ps(T_0_lo, T_0_hi, 0b11110000);
            __m256 T_1_lo = _mm256_blend_ps(T_10, T_11, 0b00001100);
            __m256 T_1_hi = _mm256_blend_ps(T_12, T_13, 0b11000000);
            __m256 T_1 = _mm256_blend_ps(T_1_lo, T_1_hi, 0b11110000);
            __m256 T_2_lo = _mm256_blend_ps(T_20, T_21, 0b00001100);
            __m256 T_2_hi = _mm256_blend_ps(T_22, T_23, 0b11000000);
            __m256 T_2 = _mm256_blend_ps(T_2_lo, T_2_hi, 0b11110000);
            __m256 T_3_lo = _mm256_blend_ps(T_30, T_31, 0b00001100);
            __m256 T_3_hi = _mm256_blend_ps(T_32, T_33, 0b11000000);
            __m256 T_3 = _mm256_blend_ps(T_3_lo, T_3_hi, 0b11110000);

            __m256 Q_0 = _mm256_mul_ps(T_0, inv_q);
            __m256 Q_1 = _mm256_mul_ps(T_1, inv_q);
            __m256 Q_2 = _mm256_mul_ps(T_2, inv_q);
            __m256 Q_3 = _mm256_mul_ps(T_3, inv_q);

            __m256 PQ_0 = _mm256_sub_ps(P_0, Q_0);
			__m256 PQ_1 = _mm256_sub_ps(P_1, Q_1);
			__m256 PQ_2 = _mm256_sub_ps(P_2, Q_2);
			__m256 PQ_3 = _mm256_sub_ps(P_3, Q_3);

            __m256 f_0 = _mm256_mul_ps(PQ_0, T_0);
            __m256 f_1 = _mm256_mul_ps(PQ_1, T_1);
            __m256 f_2 = _mm256_mul_ps(PQ_2, T_2);
            __m256 f_3 = _mm256_mul_ps(PQ_3, T_3);

            __m256 Ynm_00 = _mm256_sub_ps(Yn_0, Ym_0);
            __m256 Ynm_10 = _mm256_sub_ps(Yn_1, Ym_0);
            __m256 Ynm_20 = _mm256_sub_ps(Yn_2, Ym_0);
            __m256 Ynm_30 = _mm256_sub_ps(Yn_3, Ym_0);

            __m256 dC_0 = _mm256_mul_ps(Ynm_00, f_0);
            __m256 dC_1 = _mm256_mul_ps(Ynm_10, f_1);
            __m256 dC_2 = _mm256_mul_ps(Ynm_20, f_2);
            __m256 dC_3 = _mm256_mul_ps(Ynm_30, f_3);

            dCn_0 = _mm256_add_ps(dCn_0, dC_0);
            dCn_1 = _mm256_add_ps(dCn_1, dC_1);
            dCn_2 = _mm256_add_ps(dCn_2, dC_2);
            dCn_3 = _mm256_add_ps(dCn_3, dC_3);

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

	}
    for(; n + 4 <= N; n += 4, nD += 4*D, nN += 4*N) {

        float dC0_d0 = 0;
        float dC0_d1 = 0;
        float dC1_d0 = 0;
        float dC1_d1 = 0;
        float dC2_d0 = 0;
        float dC2_d1 = 0;
        float dC3_d0 = 0;
        float dC3_d1 = 0;

        float Yn0_d0 = Y[nD];
        float Yn0_d1 = Y[nD+1];
        float Yn1_d0 = Y[nD+2];
        float Yn1_d1 = Y[nD+3];
        float Yn2_d0 = Y[nD+4];
        float Yn2_d1 = Y[nD+5];
        float Yn3_d0 = Y[nD+6];
        float Yn3_d1 = Y[nD+7];

        float uY0_d0 = uY[nD];
        float uY0_d1 = uY[nD+1];
        float uY1_d0 = uY[nD+2];
        float uY1_d1 = uY[nD+3];
        float uY2_d0 = uY[nD+4];
        float uY2_d1 = uY[nD+5];
        float uY3_d0 = uY[nD+6];
        float uY3_d1 = uY[nD+7];

        int m = 0, mD = 0;
        for (; m + 8 <= N; m += 8, mD += 8*D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];
            float Ym1_d0 = Y[mD+2];
            float Ym1_d1 = Y[mD+3];
            float Ym2_d0 = Y[mD+4];
            float Ym2_d1 = Y[mD+5];
            float Ym3_d0 = Y[mD+6];
            float Ym3_d1 = Y[mD+7];
            float Ym4_d0 = Y[mD+8];
            float Ym4_d1 = Y[mD+9];
            float Ym5_d0 = Y[mD+10];
            float Ym5_d1 = Y[mD+11];
            float Ym6_d0 = Y[mD+12];
            float Ym6_d1 = Y[mD+13];
            float Ym7_d0 = Y[mD+14];
            float Ym7_d1 = Y[mD+15];

            float P_00 = P[nN+m];
            float P_01 = P[nN+m+1];
            float P_02 = P[nN+m+2];
            float P_03 = P[nN+m+3];
            float P_04 = P[nN+m+4];
            float P_05 = P[nN+m+5];
            float P_06 = P[nN+m+6];
            float P_07 = P[nN+m+7];
            float P_10 = P[nN+N+m];
            float P_11 = P[nN+N+m+1];
            float P_12 = P[nN+N+m+2];
            float P_13 = P[nN+N+m+3];
            float P_14 = P[nN+N+m+4];
            float P_15 = P[nN+N+m+5];
            float P_16 = P[nN+N+m+6];
            float P_17 = P[nN+N+m+7];
            float P_20 = P[nN+2*N+m];
            float P_21 = P[nN+2*N+m+1];
            float P_22 = P[nN+2*N+m+2];
            float P_23 = P[nN+2*N+m+3];
            float P_24 = P[nN+2*N+m+4];
            float P_25 = P[nN+2*N+m+5];
            float P_26 = P[nN+2*N+m+6];
            float P_27 = P[nN+2*N+m+7];
            float P_30 = P[nN+3*N+m];
            float P_31 = P[nN+3*N+m+1];
            float P_32 = P[nN+3*N+m+2];
            float P_33 = P[nN+3*N+m+3];
            float P_34 = P[nN+3*N+m+4];
            float P_35 = P[nN+3*N+m+5];
            float P_36 = P[nN+3*N+m+6];
            float P_37 = P[nN+3*N+m+7];

            float T_00 = Q[nN+m];
            float T_01 = Q[nN+m+1];
            float T_02 = Q[nN+m+2];
            float T_03 = Q[nN+m+3];
            float T_04 = Q[nN+m+4];
            float T_05 = Q[nN+m+5];
            float T_06 = Q[nN+m+6];
            float T_07 = Q[nN+m+7];
            float T_10 = Q[nN+N+m];
            float T_11 = Q[nN+N+m+1];
            float T_12 = Q[nN+N+m+2];
            float T_13 = Q[nN+N+m+3];
            float T_14 = Q[nN+N+m+4];
            float T_15 = Q[nN+N+m+5];
            float T_16 = Q[nN+N+m+6];
            float T_17 = Q[nN+N+m+7];
            float T_20 = Q[nN+2*N+m];
            float T_21 = Q[nN+2*N+m+1];
            float T_22 = Q[nN+2*N+m+2];
            float T_23 = Q[nN+2*N+m+3];
            float T_24 = Q[nN+2*N+m+4];
            float T_25 = Q[nN+2*N+m+5];
            float T_26 = Q[nN+2*N+m+6];
            float T_27 = Q[nN+2*N+m+7];
            float T_30 = Q[nN+3*N+m];
            float T_31 = Q[nN+3*N+m+1];
            float T_32 = Q[nN+3*N+m+2];
            float T_33 = Q[nN+3*N+m+3];
            float T_34 = Q[nN+3*N+m+4];
            float T_35 = Q[nN+3*N+m+5];
            float T_36 = Q[nN+3*N+m+6];
            float T_37 = Q[nN+3*N+m+7];

            float Q_00 = T_00 / sum_Q;
            float Q_01 = T_01 / sum_Q;
            float Q_02 = T_02 / sum_Q;
            float Q_03 = T_03 / sum_Q;
            float Q_04 = T_04 / sum_Q;
            float Q_05 = T_05 / sum_Q;
            float Q_06 = T_06 / sum_Q;
            float Q_07 = T_07 / sum_Q;
            float Q_10 = T_10 / sum_Q;
            float Q_11 = T_11 / sum_Q;
            float Q_12 = T_12 / sum_Q;
            float Q_13 = T_13 / sum_Q;
            float Q_14 = T_14 / sum_Q;
            float Q_15 = T_15 / sum_Q;
            float Q_16 = T_16 / sum_Q;
            float Q_17 = T_17 / sum_Q;
            float Q_20 = T_20 / sum_Q;
            float Q_21 = T_21 / sum_Q;
            float Q_22 = T_22 / sum_Q;
            float Q_23 = T_23 / sum_Q;
            float Q_24 = T_24 / sum_Q;
            float Q_25 = T_25 / sum_Q;
            float Q_26 = T_26 / sum_Q;
            float Q_27 = T_27 / sum_Q;
            float Q_30 = T_30 / sum_Q;
            float Q_31 = T_31 / sum_Q;
            float Q_32 = T_32 / sum_Q;
            float Q_33 = T_33 / sum_Q;
            float Q_34 = T_34 / sum_Q;
            float Q_35 = T_35 / sum_Q;
            float Q_36 = T_36 / sum_Q;
            float Q_37 = T_37 / sum_Q;

            float PQ_00 = P_00 - Q_00;
            float PQ_01 = P_01 - Q_01;
            float PQ_02 = P_02 - Q_02;
            float PQ_03 = P_03 - Q_03;
            float PQ_04 = P_04 - Q_04;
            float PQ_05 = P_05 - Q_05;
            float PQ_06 = P_06 - Q_06;
            float PQ_07 = P_07 - Q_07;
            float PQ_10 = P_10 - Q_10;
            float PQ_11 = P_11 - Q_11;
            float PQ_12 = P_12 - Q_12;
            float PQ_13 = P_13 - Q_13;
            float PQ_14 = P_14 - Q_14;
            float PQ_15 = P_15 - Q_15;
            float PQ_16 = P_16 - Q_16;
            float PQ_17 = P_17 - Q_17;
            float PQ_20 = P_20 - Q_20;
            float PQ_21 = P_21 - Q_21;
            float PQ_22 = P_22 - Q_22;
            float PQ_23 = P_23 - Q_23;
            float PQ_24 = P_24 - Q_24;
            float PQ_25 = P_25 - Q_25;
            float PQ_26 = P_26 - Q_26;
            float PQ_27 = P_27 - Q_27;
            float PQ_30 = P_30 - Q_30;
            float PQ_31 = P_31 - Q_31;
            float PQ_32 = P_32 - Q_32;
            float PQ_33 = P_33 - Q_33;
            float PQ_34 = P_34 - Q_34;
            float PQ_35 = P_35 - Q_35;
            float PQ_36 = P_36 - Q_36;
            float PQ_37 = P_37 - Q_37;

            float f_00 = PQ_00 * T_00;
            float f_01 = PQ_01 * T_01;
            float f_02 = PQ_02 * T_02;
            float f_03 = PQ_03 * T_03;
            float f_04 = PQ_04 * T_04;
            float f_05 = PQ_05 * T_05;
            float f_06 = PQ_06 * T_06;
            float f_07 = PQ_07 * T_07;
            float f_10 = PQ_10 * T_10;
            float f_11 = PQ_11 * T_11;
            float f_12 = PQ_12 * T_12;
            float f_13 = PQ_13 * T_13;
            float f_14 = PQ_14 * T_14;
            float f_15 = PQ_15 * T_15;
            float f_16 = PQ_16 * T_16;
            float f_17 = PQ_17 * T_17;
            float f_20 = PQ_20 * T_20;
            float f_21 = PQ_21 * T_21;
            float f_22 = PQ_22 * T_22;
            float f_23 = PQ_23 * T_23;
            float f_24 = PQ_24 * T_24;
            float f_25 = PQ_25 * T_25;
            float f_26 = PQ_26 * T_26;
            float f_27 = PQ_27 * T_27;
            float f_30 = PQ_30 * T_30;
            float f_31 = PQ_31 * T_31;
            float f_32 = PQ_32 * T_32;
            float f_33 = PQ_33 * T_33;
            float f_34 = PQ_34 * T_34;
            float f_35 = PQ_35 * T_35;
            float f_36 = PQ_36 * T_36;
            float f_37 = PQ_37 * T_37;

            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            float Ynm01_d0 = Yn0_d0 - Ym1_d0;
            float Ynm01_d1 = Yn0_d1 - Ym1_d1;
            float Ynm02_d0 = Yn0_d0 - Ym2_d0;
            float Ynm02_d1 = Yn0_d1 - Ym2_d1;
            float Ynm03_d0 = Yn0_d0 - Ym3_d0;
            float Ynm03_d1 = Yn0_d1 - Ym3_d1;
            float Ynm04_d0 = Yn0_d0 - Ym4_d0;
            float Ynm04_d1 = Yn0_d1 - Ym4_d1;
            float Ynm05_d0 = Yn0_d0 - Ym5_d0;
            float Ynm05_d1 = Yn0_d1 - Ym5_d1;
            float Ynm06_d0 = Yn0_d0 - Ym6_d0;
            float Ynm06_d1 = Yn0_d1 - Ym6_d1;
            float Ynm07_d0 = Yn0_d0 - Ym7_d0;
            float Ynm07_d1 = Yn0_d1 - Ym7_d1;
            float Ynm10_d0 = Yn1_d0 - Ym0_d0;
            float Ynm10_d1 = Yn1_d1 - Ym0_d1;
            float Ynm11_d0 = Yn1_d0 - Ym1_d0;
            float Ynm11_d1 = Yn1_d1 - Ym1_d1;
            float Ynm12_d0 = Yn1_d0 - Ym2_d0;
            float Ynm12_d1 = Yn1_d1 - Ym2_d1;
            float Ynm13_d0 = Yn1_d0 - Ym3_d0;
            float Ynm13_d1 = Yn1_d1 - Ym3_d1;
            float Ynm14_d0 = Yn1_d0 - Ym4_d0;
            float Ynm14_d1 = Yn1_d1 - Ym4_d1;
            float Ynm15_d0 = Yn1_d0 - Ym5_d0;
            float Ynm15_d1 = Yn1_d1 - Ym5_d1;
            float Ynm16_d0 = Yn1_d0 - Ym6_d0;
            float Ynm16_d1 = Yn1_d1 - Ym6_d1;
            float Ynm17_d0 = Yn1_d0 - Ym7_d0;
            float Ynm17_d1 = Yn1_d1 - Ym7_d1;
            float Ynm20_d0 = Yn2_d0 - Ym0_d0;
            float Ynm20_d1 = Yn2_d1 - Ym0_d1;
            float Ynm21_d0 = Yn2_d0 - Ym1_d0;
            float Ynm21_d1 = Yn2_d1 - Ym1_d1;
            float Ynm22_d0 = Yn2_d0 - Ym2_d0;
            float Ynm22_d1 = Yn2_d1 - Ym2_d1;
            float Ynm23_d0 = Yn2_d0 - Ym3_d0;
            float Ynm23_d1 = Yn2_d1 - Ym3_d1;
            float Ynm24_d0 = Yn2_d0 - Ym4_d0;
            float Ynm24_d1 = Yn2_d1 - Ym4_d1;
            float Ynm25_d0 = Yn2_d0 - Ym5_d0;
            float Ynm25_d1 = Yn2_d1 - Ym5_d1;
            float Ynm26_d0 = Yn2_d0 - Ym6_d0;
            float Ynm26_d1 = Yn2_d1 - Ym6_d1;
            float Ynm27_d0 = Yn2_d0 - Ym7_d0;
            float Ynm27_d1 = Yn2_d1 - Ym7_d1;
            float Ynm30_d0 = Yn3_d0 - Ym0_d0;
            float Ynm30_d1 = Yn3_d1 - Ym0_d1;
            float Ynm31_d0 = Yn3_d0 - Ym1_d0;
            float Ynm31_d1 = Yn3_d1 - Ym1_d1;
            float Ynm32_d0 = Yn3_d0 - Ym2_d0;
            float Ynm32_d1 = Yn3_d1 - Ym2_d1;
            float Ynm33_d0 = Yn3_d0 - Ym3_d0;
            float Ynm33_d1 = Yn3_d1 - Ym3_d1;
            float Ynm34_d0 = Yn3_d0 - Ym4_d0;
            float Ynm34_d1 = Yn3_d1 - Ym4_d1;
            float Ynm35_d0 = Yn3_d0 - Ym5_d0;
            float Ynm35_d1 = Yn3_d1 - Ym5_d1;
            float Ynm36_d0 = Yn3_d0 - Ym6_d0;
            float Ynm36_d1 = Yn3_d1 - Ym6_d1;
            float Ynm37_d0 = Yn3_d0 - Ym7_d0;
            float Ynm37_d1 = Yn3_d1 - Ym7_d1;


            float dC00_d0 = Ynm00_d0 * f_00;
            float dC00_d1 = Ynm00_d1 * f_00;
            float dC01_d0 = Ynm01_d0 * f_01;
            float dC01_d1 = Ynm01_d1 * f_01;
            float dC02_d0 = Ynm02_d0 * f_02;
            float dC02_d1 = Ynm02_d1 * f_02;
            float dC03_d0 = Ynm03_d0 * f_03;
            float dC03_d1 = Ynm03_d1 * f_03;
            float dC04_d0 = Ynm04_d0 * f_04;
            float dC04_d1 = Ynm04_d1 * f_04;
            float dC05_d0 = Ynm05_d0 * f_05;
            float dC05_d1 = Ynm05_d1 * f_05;
            float dC06_d0 = Ynm06_d0 * f_06;
            float dC06_d1 = Ynm06_d1 * f_06;
            float dC07_d0 = Ynm07_d0 * f_07;
            float dC07_d1 = Ynm07_d1 * f_07;
            float dC10_d0 = Ynm10_d0 * f_10;
            float dC10_d1 = Ynm10_d1 * f_10;
            float dC11_d0 = Ynm11_d0 * f_11;
            float dC11_d1 = Ynm11_d1 * f_11;
            float dC12_d0 = Ynm12_d0 * f_12;
            float dC12_d1 = Ynm12_d1 * f_12;
            float dC13_d0 = Ynm13_d0 * f_13;
            float dC13_d1 = Ynm13_d1 * f_13;
            float dC14_d0 = Ynm14_d0 * f_14;
            float dC14_d1 = Ynm14_d1 * f_14;
            float dC15_d0 = Ynm15_d0 * f_15;
            float dC15_d1 = Ynm15_d1 * f_15;
            float dC16_d0 = Ynm16_d0 * f_16;
            float dC16_d1 = Ynm16_d1 * f_16;
            float dC17_d0 = Ynm17_d0 * f_17;
            float dC17_d1 = Ynm17_d1 * f_17;
            float dC20_d0 = Ynm20_d0 * f_20;
            float dC20_d1 = Ynm20_d1 * f_20;
            float dC21_d0 = Ynm21_d0 * f_21;
            float dC21_d1 = Ynm21_d1 * f_21;
            float dC22_d0 = Ynm22_d0 * f_22;
            float dC22_d1 = Ynm22_d1 * f_22;
            float dC23_d0 = Ynm23_d0 * f_23;
            float dC23_d1 = Ynm23_d1 * f_23;
            float dC24_d0 = Ynm24_d0 * f_24;
            float dC24_d1 = Ynm24_d1 * f_24;
            float dC25_d0 = Ynm25_d0 * f_25;
            float dC25_d1 = Ynm25_d1 * f_25;
            float dC26_d0 = Ynm26_d0 * f_26;
            float dC26_d1 = Ynm26_d1 * f_26;
            float dC27_d0 = Ynm27_d0 * f_27;
            float dC27_d1 = Ynm27_d1 * f_27;
            float dC30_d0 = Ynm30_d0 * f_30;
            float dC30_d1 = Ynm30_d1 * f_30;
            float dC31_d0 = Ynm31_d0 * f_31;
            float dC31_d1 = Ynm31_d1 * f_31;
            float dC32_d0 = Ynm32_d0 * f_32;
            float dC32_d1 = Ynm32_d1 * f_32;
            float dC33_d0 = Ynm33_d0 * f_33;
            float dC33_d1 = Ynm33_d1 * f_33;
            float dC34_d0 = Ynm34_d0 * f_34;
            float dC34_d1 = Ynm34_d1 * f_34;
            float dC35_d0 = Ynm35_d0 * f_35;
            float dC35_d1 = Ynm35_d1 * f_35;
            float dC36_d0 = Ynm36_d0 * f_36;
            float dC36_d1 = Ynm36_d1 * f_36;
            float dC37_d0 = Ynm37_d0 * f_37;
            float dC37_d1 = Ynm37_d1 * f_37;

            // Accumulate the gradients
            float dC08_d0 = dC00_d0 + dC01_d0;
            float dC09_d0 = dC02_d0 + dC03_d0;
            float dC010_d0 = dC04_d0 + dC05_d0;
            float dC011_d0 = dC06_d0 + dC07_d0;
            float dC012_d0 = dC08_d0 + dC09_d0;
            float dC013_d0 = dC010_d0 + dC011_d0;
            float dC014_d0 = dC012_d0 + dC013_d0;
            float dC18_d0 = dC10_d0 + dC11_d0;
            float dC19_d0 = dC12_d0 + dC13_d0;
            float dC110_d0 = dC14_d0 + dC15_d0;
            float dC111_d0 = dC16_d0 + dC17_d0;
            float dC112_d0 = dC18_d0 + dC19_d0;
            float dC113_d0 = dC110_d0 + dC111_d0;
            float dC114_d0 = dC112_d0 + dC113_d0;
            float dC28_d0 = dC20_d0 + dC21_d0;
            float dC29_d0 = dC22_d0 + dC23_d0;
            float dC210_d0 = dC24_d0 + dC25_d0;
            float dC211_d0 = dC26_d0 + dC27_d0;
            float dC212_d0 = dC28_d0 + dC29_d0;
            float dC213_d0 = dC210_d0 + dC211_d0;
            float dC214_d0 = dC212_d0 + dC213_d0;
            float dC38_d0 = dC30_d0 + dC31_d0;
            float dC39_d0 = dC32_d0 + dC33_d0;
            float dC310_d0 = dC34_d0 + dC35_d0;
            float dC311_d0 = dC36_d0 + dC37_d0;
            float dC312_d0 = dC38_d0 + dC39_d0;
            float dC313_d0 = dC310_d0 + dC311_d0;
            float dC314_d0 = dC312_d0 + dC313_d0;
            dC0_d0 += dC014_d0;
            dC1_d0 += dC114_d0;
            dC2_d0 += dC214_d0;
            dC3_d0 += dC314_d0;
            float dC08_d1 = dC00_d1 + dC01_d1;
            float dC09_d1 = dC02_d1 + dC03_d1;
            float dC010_d1 = dC04_d1 + dC05_d1;
            float dC011_d1 = dC06_d1 + dC07_d1;
            float dC012_d1 = dC08_d1 + dC09_d1;
            float dC013_d1 = dC010_d1 + dC011_d1;
            float dC014_d1 = dC012_d1 + dC013_d1;
            float dC18_d1 = dC10_d1 + dC11_d1;
            float dC19_d1 = dC12_d1 + dC13_d1;
            float dC110_d1 = dC14_d1 + dC15_d1;
            float dC111_d1 = dC16_d1 + dC17_d1;
            float dC112_d1 = dC18_d1 + dC19_d1;
            float dC113_d1 = dC110_d1 + dC111_d1;
            float dC114_d1 = dC112_d1 + dC113_d1;
            float dC28_d1 = dC20_d1 + dC21_d1;
            float dC29_d1 = dC22_d1 + dC23_d1;
            float dC210_d1 = dC24_d1 + dC25_d1;
            float dC211_d1 = dC26_d1 + dC27_d1;
            float dC212_d1 = dC28_d1 + dC29_d1;
            float dC213_d1 = dC210_d1 + dC211_d1;
            float dC214_d1 = dC212_d1 + dC213_d1;
            float dC38_d1 = dC30_d1 + dC31_d1;
            float dC39_d1 = dC32_d1 + dC33_d1;
            float dC310_d1 = dC34_d1 + dC35_d1;
            float dC311_d1 = dC36_d1 + dC37_d1;
            float dC312_d1 = dC38_d1 + dC39_d1;
            float dC313_d1 = dC310_d1 + dC311_d1;
            float dC314_d1 = dC312_d1 + dC313_d1;
            dC0_d1 += dC014_d1;
            dC1_d1 += dC114_d1;
            dC2_d1 += dC214_d1;
            dC3_d1 += dC314_d1;

        }
        // Compute remaining
        for(; m < N; m++, mD += D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];

            float P_00 = P[nN+m];
            float P_10 = P[nN+N+m];
            float P_20 = P[nN+2*N+m];
            float P_30 = P[nN+3*N+m];

            float T_00 = Q[nN+m];
            float T_10 = Q[nN+N+m];
            float T_20 = Q[nN+2*N+m];
            float T_30 = Q[nN+3*N+m];

            float Q_00 = T_00 / sum_Q;
            float Q_10 = T_10 / sum_Q;
            float Q_20 = T_20 / sum_Q;
            float Q_30 = T_30 / sum_Q;

            float PQ_00 = P_00 - Q_00;
            float PQ_10 = P_10 - Q_10;
            float PQ_20 = P_20 - Q_20;
            float PQ_30 = P_30 - Q_30;

            float f_00 = PQ_00 * T_00;
            float f_10 = PQ_10 * T_10;
            float f_20 = PQ_20 * T_20;
            float f_30 = PQ_30 * T_30;

            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            float Ynm10_d0 = Yn1_d0 - Ym0_d0;
            float Ynm10_d1 = Yn1_d1 - Ym0_d1;
            float Ynm20_d0 = Yn2_d0 - Ym0_d0;
            float Ynm20_d1 = Yn2_d1 - Ym0_d1;
            float Ynm30_d0 = Yn3_d0 - Ym0_d0;
            float Ynm30_d1 = Yn3_d1 - Ym0_d1;

            dC0_d0 += Ynm00_d0 * f_00;
            dC0_d1 += Ynm00_d1 * f_00;
            dC1_d0 += Ynm10_d0 * f_10;
            dC1_d1 += Ynm10_d1 * f_10;
            dC2_d0 += Ynm20_d0 * f_20;
            dC2_d1 += Ynm20_d1 * f_20;
            dC3_d0 += Ynm30_d0 * f_30;
            dC3_d1 += Ynm30_d1 * f_30;
        }

        float edC0_d0 = eta * dC0_d0;
        float edC0_d1 = eta * dC0_d1;
        float edC1_d0 = eta * dC1_d0;
        float edC1_d1 = eta * dC1_d1;
        float edC2_d0 = eta * dC2_d0;
        float edC2_d1 = eta * dC2_d1;
        float edC3_d0 = eta * dC3_d0;
        float edC3_d1 = eta * dC3_d1;

        float muY0_d0 = momentum * uY0_d0;
        float muY0_d1 = momentum * uY0_d1;
        float muY1_d0 = momentum * uY1_d0;
        float muY1_d1 = momentum * uY1_d1;
        float muY2_d0 = momentum * uY2_d0;
        float muY2_d1 = momentum * uY2_d1;
        float muY3_d0 = momentum * uY3_d0;
        float muY3_d1 = momentum * uY3_d1;

        uY0_d0 = muY0_d0 - edC0_d0;
        uY0_d1 = muY0_d1 - edC0_d1;
        uY1_d0 = muY1_d0 - edC1_d0;
        uY1_d1 = muY1_d1 - edC1_d1;
        uY2_d0 = muY2_d0 - edC2_d0;
        uY2_d1 = muY2_d1 - edC2_d1;
        uY3_d0 = muY3_d0 - edC3_d0;
        uY3_d1 = muY3_d1 - edC3_d1;

        uY[nD] = uY0_d0;
        uY[nD+1] = uY0_d1;
        uY[nD+2] = uY1_d0;
        uY[nD+3] = uY1_d1;
        uY[nD+4] = uY2_d0;
        uY[nD+5] = uY2_d1;
        uY[nD+6] = uY3_d0;
        uY[nD+7] = uY3_d1;
    }
    // Now compute the gradient for the remaining points
    for(; n < N; n++, nD += D, nN += N) {

        float dC0_d0 = 0;
        float dC0_d1 = 0;
        float Yn0_d0 = Y[nD];
        float Yn0_d1 = Y[nD+1];
        float uY0_d0 = uY[nD];
        float uY0_d1 = uY[nD+1];

        int m = 0, mD = 0;
        for (; m + 8 <= N; m += 8, mD += 8*D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];
            float Ym1_d0 = Y[mD+2];
            float Ym1_d1 = Y[mD+3];
            float Ym2_d0 = Y[mD+4];
            float Ym2_d1 = Y[mD+5];
            float Ym3_d0 = Y[mD+6];
            float Ym3_d1 = Y[mD+7];
            float Ym4_d0 = Y[mD+8];
            float Ym4_d1 = Y[mD+9];
            float Ym5_d0 = Y[mD+10];
            float Ym5_d1 = Y[mD+11];
            float Ym6_d0 = Y[mD+12];
            float Ym6_d1 = Y[mD+13];
            float Ym7_d0 = Y[mD+14];
            float Ym7_d1 = Y[mD+15];


            float P_00 = P[nN+m];
            float P_01 = P[nN+m+1];
            float P_02 = P[nN+m+2];
            float P_03 = P[nN+m+3];
            float P_04 = P[nN+m+4];
            float P_05 = P[nN+m+5];
            float P_06 = P[nN+m+6];
            float P_07 = P[nN+m+7];

            float T_00 = Q[nN+m];
            float T_01 = Q[nN+m+1];
            float T_02 = Q[nN+m+2];
            float T_03 = Q[nN+m+3];
            float T_04 = Q[nN+m+4];
            float T_05 = Q[nN+m+5];
            float T_06 = Q[nN+m+6];
            float T_07 = Q[nN+m+7];

            float Q_00 = T_00 / sum_Q;
            float Q_01 = T_01 / sum_Q;
            float Q_02 = T_02 / sum_Q;
            float Q_03 = T_03 / sum_Q;
            float Q_04 = T_04 / sum_Q;
            float Q_05 = T_05 / sum_Q;
            float Q_06 = T_06 / sum_Q;
            float Q_07 = T_07 / sum_Q;

            float PQ_00 = P_00 - Q_00;
            float PQ_01 = P_01 - Q_01;
            float PQ_02 = P_02 - Q_02;
            float PQ_03 = P_03 - Q_03;
            float PQ_04 = P_04 - Q_04;
            float PQ_05 = P_05 - Q_05;
            float PQ_06 = P_06 - Q_06;
            float PQ_07 = P_07 - Q_07;

            float f_00 = PQ_00 * T_00;
            float f_01 = PQ_01 * T_01;
            float f_02 = PQ_02 * T_02;
            float f_03 = PQ_03 * T_03;
            float f_04 = PQ_04 * T_04;
            float f_05 = PQ_05 * T_05;
            float f_06 = PQ_06 * T_06;
            float f_07 = PQ_07 * T_07;

            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            float Ynm01_d0 = Yn0_d0 - Ym1_d0;
            float Ynm01_d1 = Yn0_d1 - Ym1_d1;
            float Ynm02_d0 = Yn0_d0 - Ym2_d0;
            float Ynm02_d1 = Yn0_d1 - Ym2_d1;
            float Ynm03_d0 = Yn0_d0 - Ym3_d0;
            float Ynm03_d1 = Yn0_d1 - Ym3_d1;
            float Ynm04_d0 = Yn0_d0 - Ym4_d0;
            float Ynm04_d1 = Yn0_d1 - Ym4_d1;
            float Ynm05_d0 = Yn0_d0 - Ym5_d0;
            float Ynm05_d1 = Yn0_d1 - Ym5_d1;
            float Ynm06_d0 = Yn0_d0 - Ym6_d0;
            float Ynm06_d1 = Yn0_d1 - Ym6_d1;
            float Ynm07_d0 = Yn0_d0 - Ym7_d0;
            float Ynm07_d1 = Yn0_d1 - Ym7_d1;


            float dC00_d0 = Ynm00_d0 * f_00;
            float dC00_d1 = Ynm00_d1 * f_00;
            float dC01_d0 = Ynm01_d0 * f_01;
            float dC01_d1 = Ynm01_d1 * f_01;
            float dC02_d0 = Ynm02_d0 * f_02;
            float dC02_d1 = Ynm02_d1 * f_02;
            float dC03_d0 = Ynm03_d0 * f_03;
            float dC03_d1 = Ynm03_d1 * f_03;
            float dC04_d0 = Ynm04_d0 * f_04;
            float dC04_d1 = Ynm04_d1 * f_04;
            float dC05_d0 = Ynm05_d0 * f_05;
            float dC05_d1 = Ynm05_d1 * f_05;
            float dC06_d0 = Ynm06_d0 * f_06;
            float dC06_d1 = Ynm06_d1 * f_06;
            float dC07_d0 = Ynm07_d0 * f_07;
            float dC07_d1 = Ynm07_d1 * f_07;

            // Accumulate the gradients
            float dC08_d0 = dC00_d0 + dC01_d0;
            float dC09_d0 = dC02_d0 + dC03_d0;
            float dC010_d0 = dC04_d0 + dC05_d0;
            float dC011_d0 = dC06_d0 + dC07_d0;
            float dC012_d0 = dC08_d0 + dC09_d0;
            float dC013_d0 = dC010_d0 + dC011_d0;
            float dC014_d0 = dC012_d0 + dC013_d0;
            dC0_d0 += dC014_d0;
            float dC08_d1 = dC00_d1 + dC01_d1;
            float dC09_d1 = dC02_d1 + dC03_d1;
            float dC010_d1 = dC04_d1 + dC05_d1;
            float dC011_d1 = dC06_d1 + dC07_d1;
            float dC012_d1 = dC08_d1 + dC09_d1;
            float dC013_d1 = dC010_d1 + dC011_d1;
            float dC014_d1 = dC012_d1 + dC013_d1;
            dC0_d1 += dC014_d1;
        }
        // Compute remaining
        for(; m < N; m++, mD += D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];
            float P_00 = P[nN+m];
            float T_00 = Q[nN+m];
            float Q_00 = T_00 / sum_Q;
            float PQ_00 = P_00 - Q_00;
            float f_00 = PQ_00 * T_00;
            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            dC0_d0 += Ynm00_d0 * f_00;
            dC0_d1 += Ynm00_d1 * f_00;
        }

        float edC0_d0 = eta * dC0_d0;
        float edC0_d1 = eta * dC0_d1;
        float muY0_d0 = momentum * uY0_d0;
        float muY0_d1 = momentum * uY0_d1;
        uY0_d0 = muY0_d0 - edC0_d0;
        uY0_d1 = muY0_d1 - edC0_d1;
        uY[nD] = uY0_d0;
        uY[nD+1] = uY0_d1;
    }


    __m256 _mean = _mm256_setzero_ps();
	__m256 _mean_0 = _mm256_setzero_ps();
	__m256 _mean_1 = _mm256_setzero_ps();
	__m256 _mean_2 = _mm256_setzero_ps();
	__m256 _mean_3 = _mm256_setzero_ps();
	__m256 _mean_4 = _mm256_setzero_ps();
	__m256 _mean_5 = _mm256_setzero_ps();
	__m256 _mean_6 = _mm256_setzero_ps();
	__m256 _mean_7 = _mm256_setzero_ps();

	__m256 inv_N = _mm256_set1_ps(1 / (double) N);

    for (n = 0, nD = 0; n + 32 <= N; n += 32, nD += 32*D) {
        __m256 uY_0 = _mm256_loadu_ps(uY+nD);
        __m256 uY_1 = _mm256_loadu_ps(uY+nD+8);
		__m256 uY_2 = _mm256_loadu_ps(uY+nD+16);
		__m256 uY_3 = _mm256_loadu_ps(uY+nD+24);
		__m256 uY_4 = _mm256_loadu_ps(uY+nD+32);
		__m256 uY_5 = _mm256_loadu_ps(uY+nD+40);
		__m256 uY_6 = _mm256_loadu_ps(uY+nD+48);
		__m256 uY_7 = _mm256_loadu_ps(uY+nD+56);

        __m256 Y_0 = _mm256_loadu_ps(Y+nD);
        __m256 Y_1 = _mm256_loadu_ps(Y+nD+8);
		__m256 Y_2 = _mm256_loadu_ps(Y+nD+16);
		__m256 Y_3 = _mm256_loadu_ps(Y+nD+24);
		__m256 Y_4 = _mm256_loadu_ps(Y+nD+32);
		__m256 Y_5 = _mm256_loadu_ps(Y+nD+40);
		__m256 Y_6 = _mm256_loadu_ps(Y+nD+48);
		__m256 Y_7 = _mm256_loadu_ps(Y+nD+56);

        __m256 Yx_0 = _mm256_add_ps(Y_0, uY_0);
		__m256 Yx_1 = _mm256_add_ps(Y_1, uY_1);
		__m256 Yx_2 = _mm256_add_ps(Y_2, uY_2);
		__m256 Yx_3 = _mm256_add_ps(Y_3, uY_3);
		__m256 Yx_4 = _mm256_add_ps(Y_4, uY_4);
		__m256 Yx_5 = _mm256_add_ps(Y_5, uY_5);
		__m256 Yx_6 = _mm256_add_ps(Y_6, uY_6);
		__m256 Yx_7 = _mm256_add_ps(Y_7, uY_7);

        _mean_0 = _mm256_add_ps(_mean_0, Yx_0);
		_mean_1 = _mm256_add_ps(_mean_1, Yx_1);
		_mean_2 = _mm256_add_ps(_mean_2, Yx_2);
		_mean_3 = _mm256_add_ps(_mean_3, Yx_3);
		_mean_4 = _mm256_add_ps(_mean_4, Yx_4);
		_mean_5 = _mm256_add_ps(_mean_5, Yx_5);
		_mean_6 = _mm256_add_ps(_mean_6, Yx_6);
		_mean_7 = _mm256_add_ps(_mean_7, Yx_7);

        _mm256_storeu_ps(Y+nD, Yx_0);
		_mm256_storeu_ps(Y+nD+8, Yx_1);
		_mm256_storeu_ps(Y+nD+16, Yx_2);
		_mm256_storeu_ps(Y+nD+24, Yx_3);
		_mm256_storeu_ps(Y+nD+32, Yx_4);
		_mm256_storeu_ps(Y+nD+40, Yx_5);
		_mm256_storeu_ps(Y+nD+48, Yx_6);
		_mm256_storeu_ps(Y+nD+56, Yx_7);
    }
    for (; n + 4 <= N; n += 4, nD += 4*D) {
        __m256 uY_ = _mm256_loadu_ps(uY+nD);
        __m256 Y_ = _mm256_loadu_ps(Y+nD);

        __m256 Yx_ = _mm256_add_ps(Y_, uY_);
        _mean = _mm256_add_ps(_mean, Yx_);

        _mm256_storeu_ps(Y+nD, Yx_);
    }
    for (; n < N; n++, nD += D) {
        Y[nD] += uY[nD];
        Y[nD+1] += uY[nD+1];
        _mean[0] += Y[nD];
		_mean[1] += Y[nD+1];
    }

    __m256 _mean_8 = _mm256_add_ps(_mean_0, _mean_1);
	__m256 _mean_9 = _mm256_add_ps(_mean_2, _mean_3);
	__m256 _mean_10 = _mm256_add_ps(_mean_4, _mean_5);
	__m256 _mean_11 = _mm256_add_ps(_mean_6, _mean_7);
	__m256 _mean_12 = _mm256_add_ps(_mean_8, _mean_9);
	__m256 _mean_13 = _mm256_add_ps(_mean_10, _mean_11);

	__m256 _mean_ = _mm256_add_ps(_mean_12, _mean_13);
	_mean_ = _mm256_add_ps(_mean, _mean_); // Sum the remaining samples

	__m256 _mean_permute = _mm256_permute2f128_ps(_mean_, _mean_, 1);
	__m256 _mean__ = _mm256_add_ps(_mean_, _mean_permute);

	__m256 _mean__shuffle = _mm256_shuffle_ps(_mean__, _mean__, 78);
	_mean = _mm256_add_ps(_mean__, _mean__shuffle);

	_mean = _mm256_mul_ps(_mean, inv_N);

	// Subtract data mean
	for (n = 0, nD = 0; n + 32 <= N; n += 32, nD += 32*D) {
		__m256 _Y_0 = _mm256_loadu_ps(Y+nD);
		__m256 _Y_1 = _mm256_loadu_ps(Y+nD+8);
		__m256 _Y_2 = _mm256_loadu_ps(Y+nD+16);
		__m256 _Y_3 = _mm256_loadu_ps(Y+nD+24);
		__m256 _Y_4 = _mm256_loadu_ps(Y+nD+32);
		__m256 _Y_5 = _mm256_loadu_ps(Y+nD+40);
		__m256 _Y_6 = _mm256_loadu_ps(Y+nD+48);
		__m256 _Y_7 = _mm256_loadu_ps(Y+nD+56);

		_Y_0 = _mm256_sub_ps(_Y_0, _mean);
		_Y_1 = _mm256_sub_ps(_Y_1, _mean);
		_Y_2 = _mm256_sub_ps(_Y_2, _mean);
		_Y_3 = _mm256_sub_ps(_Y_3, _mean);
		_Y_4 = _mm256_sub_ps(_Y_4, _mean);
		_Y_5 = _mm256_sub_ps(_Y_5, _mean);
		_Y_6 = _mm256_sub_ps(_Y_6, _mean);
		_Y_7 = _mm256_sub_ps(_Y_7, _mean);

		_mm256_storeu_ps(Y+nD, _Y_0);
		_mm256_storeu_ps(Y+nD+8, _Y_1);
		_mm256_storeu_ps(Y+nD+16, _Y_2);
		_mm256_storeu_ps(Y+nD+24, _Y_3);
		_mm256_storeu_ps(Y+nD+32, _Y_4);
		_mm256_storeu_ps(Y+nD+40, _Y_5);
		_mm256_storeu_ps(Y+nD+48, _Y_6);
		_mm256_storeu_ps(Y+nD+56, _Y_7);

	}
	for (; n + 4 <= N; n += 4, nD += 4*D) {
		__m256 _Y = _mm256_loadu_ps(Y+nD);
		_Y = _mm256_sub_ps(_Y, _mean);
		_mm256_storeu_ps(Y+nD, _Y);
	}
	for (; n < N; n++, nD += D) {
		Y[nD] -= _mean[0];
		Y[nD+1] -= _mean[1];
	}
}


inline void unfold_accumulators(float* Y, float* P, float* Q, float sum_Q, int N,
			 int D, float* dC, float* uY, float momentum,
			 float eta) {
    // Perform the computation of the gradient
    int n = 0, nD = 0, nN = 0;
    // Compute the gradient for 4 points at the same time
    for(; n + 4 <= N; n += 4, nD += 4*D, nN += 4*N) {

        float dC0_d0 = 0;
        float dC0_d1 = 0;
        float dC1_d0 = 0;
        float dC1_d1 = 0;
        float dC2_d0 = 0;
        float dC2_d1 = 0;
        float dC3_d0 = 0;
        float dC3_d1 = 0;

        float Yn0_d0 = Y[nD];
        float Yn0_d1 = Y[nD+1];
        float Yn1_d0 = Y[nD+2];
        float Yn1_d1 = Y[nD+3];
        float Yn2_d0 = Y[nD+4];
        float Yn2_d1 = Y[nD+5];
        float Yn3_d0 = Y[nD+6];
        float Yn3_d1 = Y[nD+7];

        float uY0_d0 = uY[nD];
        float uY0_d1 = uY[nD+1];
        float uY1_d0 = uY[nD+2];
        float uY1_d1 = uY[nD+3];
        float uY2_d0 = uY[nD+4];
        float uY2_d1 = uY[nD+5];
        float uY3_d0 = uY[nD+6];
        float uY3_d1 = uY[nD+7];

    	int m = 0, mD = 0;
        for (; m + 8 <= N; m += 8, mD += 8*D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];
            float Ym1_d0 = Y[mD+2];
            float Ym1_d1 = Y[mD+3];
            float Ym2_d0 = Y[mD+4];
            float Ym2_d1 = Y[mD+5];
            float Ym3_d0 = Y[mD+6];
            float Ym3_d1 = Y[mD+7];
            float Ym4_d0 = Y[mD+8];
            float Ym4_d1 = Y[mD+9];
            float Ym5_d0 = Y[mD+10];
            float Ym5_d1 = Y[mD+11];
            float Ym6_d0 = Y[mD+12];
            float Ym6_d1 = Y[mD+13];
            float Ym7_d0 = Y[mD+14];
            float Ym7_d1 = Y[mD+15];

            float P_00 = P[nN+m];
            float P_01 = P[nN+m+1];
            float P_02 = P[nN+m+2];
            float P_03 = P[nN+m+3];
            float P_04 = P[nN+m+4];
            float P_05 = P[nN+m+5];
            float P_06 = P[nN+m+6];
            float P_07 = P[nN+m+7];
            float P_10 = P[nN+N+m];
            float P_11 = P[nN+N+m+1];
            float P_12 = P[nN+N+m+2];
            float P_13 = P[nN+N+m+3];
            float P_14 = P[nN+N+m+4];
            float P_15 = P[nN+N+m+5];
            float P_16 = P[nN+N+m+6];
            float P_17 = P[nN+N+m+7];
            float P_20 = P[nN+2*N+m];
            float P_21 = P[nN+2*N+m+1];
            float P_22 = P[nN+2*N+m+2];
            float P_23 = P[nN+2*N+m+3];
            float P_24 = P[nN+2*N+m+4];
            float P_25 = P[nN+2*N+m+5];
            float P_26 = P[nN+2*N+m+6];
            float P_27 = P[nN+2*N+m+7];
            float P_30 = P[nN+3*N+m];
            float P_31 = P[nN+3*N+m+1];
            float P_32 = P[nN+3*N+m+2];
            float P_33 = P[nN+3*N+m+3];
            float P_34 = P[nN+3*N+m+4];
            float P_35 = P[nN+3*N+m+5];
            float P_36 = P[nN+3*N+m+6];
            float P_37 = P[nN+3*N+m+7];

            float T_00 = Q[nN+m];
            float T_01 = Q[nN+m+1];
            float T_02 = Q[nN+m+2];
            float T_03 = Q[nN+m+3];
            float T_04 = Q[nN+m+4];
            float T_05 = Q[nN+m+5];
            float T_06 = Q[nN+m+6];
            float T_07 = Q[nN+m+7];
            float T_10 = Q[nN+N+m];
            float T_11 = Q[nN+N+m+1];
            float T_12 = Q[nN+N+m+2];
            float T_13 = Q[nN+N+m+3];
            float T_14 = Q[nN+N+m+4];
            float T_15 = Q[nN+N+m+5];
            float T_16 = Q[nN+N+m+6];
            float T_17 = Q[nN+N+m+7];
            float T_20 = Q[nN+2*N+m];
            float T_21 = Q[nN+2*N+m+1];
            float T_22 = Q[nN+2*N+m+2];
            float T_23 = Q[nN+2*N+m+3];
            float T_24 = Q[nN+2*N+m+4];
            float T_25 = Q[nN+2*N+m+5];
            float T_26 = Q[nN+2*N+m+6];
            float T_27 = Q[nN+2*N+m+7];
            float T_30 = Q[nN+3*N+m];
            float T_31 = Q[nN+3*N+m+1];
            float T_32 = Q[nN+3*N+m+2];
            float T_33 = Q[nN+3*N+m+3];
            float T_34 = Q[nN+3*N+m+4];
            float T_35 = Q[nN+3*N+m+5];
            float T_36 = Q[nN+3*N+m+6];
            float T_37 = Q[nN+3*N+m+7];

            float Q_00 = T_00 / sum_Q;
            float Q_01 = T_01 / sum_Q;
            float Q_02 = T_02 / sum_Q;
            float Q_03 = T_03 / sum_Q;
            float Q_04 = T_04 / sum_Q;
            float Q_05 = T_05 / sum_Q;
            float Q_06 = T_06 / sum_Q;
            float Q_07 = T_07 / sum_Q;
            float Q_10 = T_10 / sum_Q;
            float Q_11 = T_11 / sum_Q;
            float Q_12 = T_12 / sum_Q;
            float Q_13 = T_13 / sum_Q;
            float Q_14 = T_14 / sum_Q;
            float Q_15 = T_15 / sum_Q;
            float Q_16 = T_16 / sum_Q;
            float Q_17 = T_17 / sum_Q;
            float Q_20 = T_20 / sum_Q;
            float Q_21 = T_21 / sum_Q;
            float Q_22 = T_22 / sum_Q;
            float Q_23 = T_23 / sum_Q;
            float Q_24 = T_24 / sum_Q;
            float Q_25 = T_25 / sum_Q;
            float Q_26 = T_26 / sum_Q;
            float Q_27 = T_27 / sum_Q;
            float Q_30 = T_30 / sum_Q;
            float Q_31 = T_31 / sum_Q;
            float Q_32 = T_32 / sum_Q;
            float Q_33 = T_33 / sum_Q;
            float Q_34 = T_34 / sum_Q;
            float Q_35 = T_35 / sum_Q;
            float Q_36 = T_36 / sum_Q;
            float Q_37 = T_37 / sum_Q;

            float PQ_00 = P_00 - Q_00;
            float PQ_01 = P_01 - Q_01;
            float PQ_02 = P_02 - Q_02;
            float PQ_03 = P_03 - Q_03;
            float PQ_04 = P_04 - Q_04;
            float PQ_05 = P_05 - Q_05;
            float PQ_06 = P_06 - Q_06;
            float PQ_07 = P_07 - Q_07;
            float PQ_10 = P_10 - Q_10;
            float PQ_11 = P_11 - Q_11;
            float PQ_12 = P_12 - Q_12;
            float PQ_13 = P_13 - Q_13;
            float PQ_14 = P_14 - Q_14;
            float PQ_15 = P_15 - Q_15;
            float PQ_16 = P_16 - Q_16;
            float PQ_17 = P_17 - Q_17;
            float PQ_20 = P_20 - Q_20;
            float PQ_21 = P_21 - Q_21;
            float PQ_22 = P_22 - Q_22;
            float PQ_23 = P_23 - Q_23;
            float PQ_24 = P_24 - Q_24;
            float PQ_25 = P_25 - Q_25;
            float PQ_26 = P_26 - Q_26;
            float PQ_27 = P_27 - Q_27;
            float PQ_30 = P_30 - Q_30;
            float PQ_31 = P_31 - Q_31;
            float PQ_32 = P_32 - Q_32;
            float PQ_33 = P_33 - Q_33;
            float PQ_34 = P_34 - Q_34;
            float PQ_35 = P_35 - Q_35;
            float PQ_36 = P_36 - Q_36;
            float PQ_37 = P_37 - Q_37;

            float f_00 = PQ_00 * T_00;
            float f_01 = PQ_01 * T_01;
            float f_02 = PQ_02 * T_02;
            float f_03 = PQ_03 * T_03;
            float f_04 = PQ_04 * T_04;
            float f_05 = PQ_05 * T_05;
            float f_06 = PQ_06 * T_06;
            float f_07 = PQ_07 * T_07;
            float f_10 = PQ_10 * T_10;
            float f_11 = PQ_11 * T_11;
            float f_12 = PQ_12 * T_12;
            float f_13 = PQ_13 * T_13;
            float f_14 = PQ_14 * T_14;
            float f_15 = PQ_15 * T_15;
            float f_16 = PQ_16 * T_16;
            float f_17 = PQ_17 * T_17;
            float f_20 = PQ_20 * T_20;
            float f_21 = PQ_21 * T_21;
            float f_22 = PQ_22 * T_22;
            float f_23 = PQ_23 * T_23;
            float f_24 = PQ_24 * T_24;
            float f_25 = PQ_25 * T_25;
            float f_26 = PQ_26 * T_26;
            float f_27 = PQ_27 * T_27;
            float f_30 = PQ_30 * T_30;
            float f_31 = PQ_31 * T_31;
            float f_32 = PQ_32 * T_32;
            float f_33 = PQ_33 * T_33;
            float f_34 = PQ_34 * T_34;
            float f_35 = PQ_35 * T_35;
            float f_36 = PQ_36 * T_36;
            float f_37 = PQ_37 * T_37;

            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            float Ynm01_d0 = Yn0_d0 - Ym1_d0;
            float Ynm01_d1 = Yn0_d1 - Ym1_d1;
            float Ynm02_d0 = Yn0_d0 - Ym2_d0;
            float Ynm02_d1 = Yn0_d1 - Ym2_d1;
            float Ynm03_d0 = Yn0_d0 - Ym3_d0;
            float Ynm03_d1 = Yn0_d1 - Ym3_d1;
            float Ynm04_d0 = Yn0_d0 - Ym4_d0;
            float Ynm04_d1 = Yn0_d1 - Ym4_d1;
            float Ynm05_d0 = Yn0_d0 - Ym5_d0;
            float Ynm05_d1 = Yn0_d1 - Ym5_d1;
            float Ynm06_d0 = Yn0_d0 - Ym6_d0;
            float Ynm06_d1 = Yn0_d1 - Ym6_d1;
            float Ynm07_d0 = Yn0_d0 - Ym7_d0;
            float Ynm07_d1 = Yn0_d1 - Ym7_d1;
            float Ynm10_d0 = Yn1_d0 - Ym0_d0;
            float Ynm10_d1 = Yn1_d1 - Ym0_d1;
            float Ynm11_d0 = Yn1_d0 - Ym1_d0;
            float Ynm11_d1 = Yn1_d1 - Ym1_d1;
            float Ynm12_d0 = Yn1_d0 - Ym2_d0;
            float Ynm12_d1 = Yn1_d1 - Ym2_d1;
            float Ynm13_d0 = Yn1_d0 - Ym3_d0;
            float Ynm13_d1 = Yn1_d1 - Ym3_d1;
            float Ynm14_d0 = Yn1_d0 - Ym4_d0;
            float Ynm14_d1 = Yn1_d1 - Ym4_d1;
            float Ynm15_d0 = Yn1_d0 - Ym5_d0;
            float Ynm15_d1 = Yn1_d1 - Ym5_d1;
            float Ynm16_d0 = Yn1_d0 - Ym6_d0;
            float Ynm16_d1 = Yn1_d1 - Ym6_d1;
            float Ynm17_d0 = Yn1_d0 - Ym7_d0;
            float Ynm17_d1 = Yn1_d1 - Ym7_d1;
            float Ynm20_d0 = Yn2_d0 - Ym0_d0;
            float Ynm20_d1 = Yn2_d1 - Ym0_d1;
            float Ynm21_d0 = Yn2_d0 - Ym1_d0;
            float Ynm21_d1 = Yn2_d1 - Ym1_d1;
            float Ynm22_d0 = Yn2_d0 - Ym2_d0;
            float Ynm22_d1 = Yn2_d1 - Ym2_d1;
            float Ynm23_d0 = Yn2_d0 - Ym3_d0;
            float Ynm23_d1 = Yn2_d1 - Ym3_d1;
            float Ynm24_d0 = Yn2_d0 - Ym4_d0;
            float Ynm24_d1 = Yn2_d1 - Ym4_d1;
            float Ynm25_d0 = Yn2_d0 - Ym5_d0;
            float Ynm25_d1 = Yn2_d1 - Ym5_d1;
            float Ynm26_d0 = Yn2_d0 - Ym6_d0;
            float Ynm26_d1 = Yn2_d1 - Ym6_d1;
            float Ynm27_d0 = Yn2_d0 - Ym7_d0;
            float Ynm27_d1 = Yn2_d1 - Ym7_d1;
            float Ynm30_d0 = Yn3_d0 - Ym0_d0;
            float Ynm30_d1 = Yn3_d1 - Ym0_d1;
            float Ynm31_d0 = Yn3_d0 - Ym1_d0;
            float Ynm31_d1 = Yn3_d1 - Ym1_d1;
            float Ynm32_d0 = Yn3_d0 - Ym2_d0;
            float Ynm32_d1 = Yn3_d1 - Ym2_d1;
            float Ynm33_d0 = Yn3_d0 - Ym3_d0;
            float Ynm33_d1 = Yn3_d1 - Ym3_d1;
            float Ynm34_d0 = Yn3_d0 - Ym4_d0;
            float Ynm34_d1 = Yn3_d1 - Ym4_d1;
            float Ynm35_d0 = Yn3_d0 - Ym5_d0;
            float Ynm35_d1 = Yn3_d1 - Ym5_d1;
            float Ynm36_d0 = Yn3_d0 - Ym6_d0;
            float Ynm36_d1 = Yn3_d1 - Ym6_d1;
            float Ynm37_d0 = Yn3_d0 - Ym7_d0;
            float Ynm37_d1 = Yn3_d1 - Ym7_d1;


            float dC00_d0 = Ynm00_d0 * f_00;
            float dC00_d1 = Ynm00_d1 * f_00;
            float dC01_d0 = Ynm01_d0 * f_01;
            float dC01_d1 = Ynm01_d1 * f_01;
            float dC02_d0 = Ynm02_d0 * f_02;
            float dC02_d1 = Ynm02_d1 * f_02;
            float dC03_d0 = Ynm03_d0 * f_03;
            float dC03_d1 = Ynm03_d1 * f_03;
            float dC04_d0 = Ynm04_d0 * f_04;
            float dC04_d1 = Ynm04_d1 * f_04;
            float dC05_d0 = Ynm05_d0 * f_05;
            float dC05_d1 = Ynm05_d1 * f_05;
            float dC06_d0 = Ynm06_d0 * f_06;
            float dC06_d1 = Ynm06_d1 * f_06;
            float dC07_d0 = Ynm07_d0 * f_07;
            float dC07_d1 = Ynm07_d1 * f_07;
            float dC10_d0 = Ynm10_d0 * f_10;
            float dC10_d1 = Ynm10_d1 * f_10;
            float dC11_d0 = Ynm11_d0 * f_11;
            float dC11_d1 = Ynm11_d1 * f_11;
            float dC12_d0 = Ynm12_d0 * f_12;
            float dC12_d1 = Ynm12_d1 * f_12;
            float dC13_d0 = Ynm13_d0 * f_13;
            float dC13_d1 = Ynm13_d1 * f_13;
            float dC14_d0 = Ynm14_d0 * f_14;
            float dC14_d1 = Ynm14_d1 * f_14;
            float dC15_d0 = Ynm15_d0 * f_15;
            float dC15_d1 = Ynm15_d1 * f_15;
            float dC16_d0 = Ynm16_d0 * f_16;
            float dC16_d1 = Ynm16_d1 * f_16;
            float dC17_d0 = Ynm17_d0 * f_17;
            float dC17_d1 = Ynm17_d1 * f_17;
            float dC20_d0 = Ynm20_d0 * f_20;
            float dC20_d1 = Ynm20_d1 * f_20;
            float dC21_d0 = Ynm21_d0 * f_21;
            float dC21_d1 = Ynm21_d1 * f_21;
            float dC22_d0 = Ynm22_d0 * f_22;
            float dC22_d1 = Ynm22_d1 * f_22;
            float dC23_d0 = Ynm23_d0 * f_23;
            float dC23_d1 = Ynm23_d1 * f_23;
            float dC24_d0 = Ynm24_d0 * f_24;
            float dC24_d1 = Ynm24_d1 * f_24;
            float dC25_d0 = Ynm25_d0 * f_25;
            float dC25_d1 = Ynm25_d1 * f_25;
            float dC26_d0 = Ynm26_d0 * f_26;
            float dC26_d1 = Ynm26_d1 * f_26;
            float dC27_d0 = Ynm27_d0 * f_27;
            float dC27_d1 = Ynm27_d1 * f_27;
            float dC30_d0 = Ynm30_d0 * f_30;
            float dC30_d1 = Ynm30_d1 * f_30;
            float dC31_d0 = Ynm31_d0 * f_31;
            float dC31_d1 = Ynm31_d1 * f_31;
            float dC32_d0 = Ynm32_d0 * f_32;
            float dC32_d1 = Ynm32_d1 * f_32;
            float dC33_d0 = Ynm33_d0 * f_33;
            float dC33_d1 = Ynm33_d1 * f_33;
            float dC34_d0 = Ynm34_d0 * f_34;
            float dC34_d1 = Ynm34_d1 * f_34;
            float dC35_d0 = Ynm35_d0 * f_35;
            float dC35_d1 = Ynm35_d1 * f_35;
            float dC36_d0 = Ynm36_d0 * f_36;
            float dC36_d1 = Ynm36_d1 * f_36;
            float dC37_d0 = Ynm37_d0 * f_37;
            float dC37_d1 = Ynm37_d1 * f_37;

            // Accumulate the gradients
            float dC08_d0 = dC00_d0 + dC01_d0;
            float dC09_d0 = dC02_d0 + dC03_d0;
            float dC010_d0 = dC04_d0 + dC05_d0;
            float dC011_d0 = dC06_d0 + dC07_d0;
            float dC012_d0 = dC08_d0 + dC09_d0;
            float dC013_d0 = dC010_d0 + dC011_d0;
            float dC014_d0 = dC012_d0 + dC013_d0;
            float dC18_d0 = dC10_d0 + dC11_d0;
            float dC19_d0 = dC12_d0 + dC13_d0;
            float dC110_d0 = dC14_d0 + dC15_d0;
            float dC111_d0 = dC16_d0 + dC17_d0;
            float dC112_d0 = dC18_d0 + dC19_d0;
            float dC113_d0 = dC110_d0 + dC111_d0;
            float dC114_d0 = dC112_d0 + dC113_d0;
            float dC28_d0 = dC20_d0 + dC21_d0;
            float dC29_d0 = dC22_d0 + dC23_d0;
            float dC210_d0 = dC24_d0 + dC25_d0;
            float dC211_d0 = dC26_d0 + dC27_d0;
            float dC212_d0 = dC28_d0 + dC29_d0;
            float dC213_d0 = dC210_d0 + dC211_d0;
            float dC214_d0 = dC212_d0 + dC213_d0;
            float dC38_d0 = dC30_d0 + dC31_d0;
            float dC39_d0 = dC32_d0 + dC33_d0;
            float dC310_d0 = dC34_d0 + dC35_d0;
            float dC311_d0 = dC36_d0 + dC37_d0;
            float dC312_d0 = dC38_d0 + dC39_d0;
            float dC313_d0 = dC310_d0 + dC311_d0;
            float dC314_d0 = dC312_d0 + dC313_d0;
            dC0_d0 += dC014_d0;
            dC1_d0 += dC114_d0;
            dC2_d0 += dC214_d0;
            dC3_d0 += dC314_d0;
            float dC08_d1 = dC00_d1 + dC01_d1;
            float dC09_d1 = dC02_d1 + dC03_d1;
            float dC010_d1 = dC04_d1 + dC05_d1;
            float dC011_d1 = dC06_d1 + dC07_d1;
            float dC012_d1 = dC08_d1 + dC09_d1;
            float dC013_d1 = dC010_d1 + dC011_d1;
            float dC014_d1 = dC012_d1 + dC013_d1;
            float dC18_d1 = dC10_d1 + dC11_d1;
            float dC19_d1 = dC12_d1 + dC13_d1;
            float dC110_d1 = dC14_d1 + dC15_d1;
            float dC111_d1 = dC16_d1 + dC17_d1;
            float dC112_d1 = dC18_d1 + dC19_d1;
            float dC113_d1 = dC110_d1 + dC111_d1;
            float dC114_d1 = dC112_d1 + dC113_d1;
            float dC28_d1 = dC20_d1 + dC21_d1;
            float dC29_d1 = dC22_d1 + dC23_d1;
            float dC210_d1 = dC24_d1 + dC25_d1;
            float dC211_d1 = dC26_d1 + dC27_d1;
            float dC212_d1 = dC28_d1 + dC29_d1;
            float dC213_d1 = dC210_d1 + dC211_d1;
            float dC214_d1 = dC212_d1 + dC213_d1;
            float dC38_d1 = dC30_d1 + dC31_d1;
            float dC39_d1 = dC32_d1 + dC33_d1;
            float dC310_d1 = dC34_d1 + dC35_d1;
            float dC311_d1 = dC36_d1 + dC37_d1;
            float dC312_d1 = dC38_d1 + dC39_d1;
            float dC313_d1 = dC310_d1 + dC311_d1;
            float dC314_d1 = dC312_d1 + dC313_d1;
            dC0_d1 += dC014_d1;
            dC1_d1 += dC114_d1;
            dC2_d1 += dC214_d1;
            dC3_d1 += dC314_d1;

        }
        // Compute remaining
    	for(; m < N; m++, mD += D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];

            float P_00 = P[nN+m];
            float P_10 = P[nN+N+m];
            float P_20 = P[nN+2*N+m];
            float P_30 = P[nN+3*N+m];

            float T_00 = Q[nN+m];
            float T_10 = Q[nN+N+m];
            float T_20 = Q[nN+2*N+m];
            float T_30 = Q[nN+3*N+m];

            float Q_00 = T_00 / sum_Q;
            float Q_10 = T_10 / sum_Q;
            float Q_20 = T_20 / sum_Q;
            float Q_30 = T_30 / sum_Q;

            float PQ_00 = P_00 - Q_00;
            float PQ_10 = P_10 - Q_10;
            float PQ_20 = P_20 - Q_20;
            float PQ_30 = P_30 - Q_30;

            float f_00 = PQ_00 * T_00;
            float f_10 = PQ_10 * T_10;
            float f_20 = PQ_20 * T_20;
            float f_30 = PQ_30 * T_30;

            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            float Ynm10_d0 = Yn1_d0 - Ym0_d0;
            float Ynm10_d1 = Yn1_d1 - Ym0_d1;
            float Ynm20_d0 = Yn2_d0 - Ym0_d0;
            float Ynm20_d1 = Yn2_d1 - Ym0_d1;
            float Ynm30_d0 = Yn3_d0 - Ym0_d0;
            float Ynm30_d1 = Yn3_d1 - Ym0_d1;

            dC0_d0 += Ynm00_d0 * f_00;
            dC0_d1 += Ynm00_d1 * f_00;
            dC1_d0 += Ynm10_d0 * f_10;
            dC1_d1 += Ynm10_d1 * f_10;
            dC2_d0 += Ynm20_d0 * f_20;
            dC2_d1 += Ynm20_d1 * f_20;
            dC3_d0 += Ynm30_d0 * f_30;
            dC3_d1 += Ynm30_d1 * f_30;
    	}

        float edC0_d0 = eta * dC0_d0;
        float edC0_d1 = eta * dC0_d1;
        float edC1_d0 = eta * dC1_d0;
        float edC1_d1 = eta * dC1_d1;
        float edC2_d0 = eta * dC2_d0;
        float edC2_d1 = eta * dC2_d1;
        float edC3_d0 = eta * dC3_d0;
        float edC3_d1 = eta * dC3_d1;

        float muY0_d0 = momentum * uY0_d0;
        float muY0_d1 = momentum * uY0_d1;
        float muY1_d0 = momentum * uY1_d0;
        float muY1_d1 = momentum * uY1_d1;
        float muY2_d0 = momentum * uY2_d0;
        float muY2_d1 = momentum * uY2_d1;
        float muY3_d0 = momentum * uY3_d0;
        float muY3_d1 = momentum * uY3_d1;

        uY0_d0 = muY0_d0 - edC0_d0;
        uY0_d1 = muY0_d1 - edC0_d1;
        uY1_d0 = muY1_d0 - edC1_d0;
        uY1_d1 = muY1_d1 - edC1_d1;
        uY2_d0 = muY2_d0 - edC2_d0;
        uY2_d1 = muY2_d1 - edC2_d1;
        uY3_d0 = muY3_d0 - edC3_d0;
        uY3_d1 = muY3_d1 - edC3_d1;

        uY[nD] = uY0_d0;
        uY[nD+1] = uY0_d1;
        uY[nD+2] = uY1_d0;
        uY[nD+3] = uY1_d1;
        uY[nD+4] = uY2_d0;
        uY[nD+5] = uY2_d1;
        uY[nD+6] = uY3_d0;
        uY[nD+7] = uY3_d1;
    }
    // Now compute the gradient for the remaining points
    for(; n < N; n++, nD += D, nN += N) {

        float dC0_d0 = 0;
        float dC0_d1 = 0;
        float Yn0_d0 = Y[nD];
        float Yn0_d1 = Y[nD+1];
        float uY0_d0 = uY[nD];
        float uY0_d1 = uY[nD+1];

    	int m = 0, mD = 0;
        for (; m + 8 <= N; m += 8, mD += 8*D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];
            float Ym1_d0 = Y[mD+2];
            float Ym1_d1 = Y[mD+3];
            float Ym2_d0 = Y[mD+4];
            float Ym2_d1 = Y[mD+5];
            float Ym3_d0 = Y[mD+6];
            float Ym3_d1 = Y[mD+7];
            float Ym4_d0 = Y[mD+8];
            float Ym4_d1 = Y[mD+9];
            float Ym5_d0 = Y[mD+10];
            float Ym5_d1 = Y[mD+11];
            float Ym6_d0 = Y[mD+12];
            float Ym6_d1 = Y[mD+13];
            float Ym7_d0 = Y[mD+14];
            float Ym7_d1 = Y[mD+15];


            float P_00 = P[nN+m];
            float P_01 = P[nN+m+1];
            float P_02 = P[nN+m+2];
            float P_03 = P[nN+m+3];
            float P_04 = P[nN+m+4];
            float P_05 = P[nN+m+5];
            float P_06 = P[nN+m+6];
            float P_07 = P[nN+m+7];

            float T_00 = Q[nN+m];
            float T_01 = Q[nN+m+1];
            float T_02 = Q[nN+m+2];
            float T_03 = Q[nN+m+3];
            float T_04 = Q[nN+m+4];
            float T_05 = Q[nN+m+5];
            float T_06 = Q[nN+m+6];
            float T_07 = Q[nN+m+7];

            float Q_00 = T_00 / sum_Q;
            float Q_01 = T_01 / sum_Q;
            float Q_02 = T_02 / sum_Q;
            float Q_03 = T_03 / sum_Q;
            float Q_04 = T_04 / sum_Q;
            float Q_05 = T_05 / sum_Q;
            float Q_06 = T_06 / sum_Q;
            float Q_07 = T_07 / sum_Q;

            float PQ_00 = P_00 - Q_00;
            float PQ_01 = P_01 - Q_01;
            float PQ_02 = P_02 - Q_02;
            float PQ_03 = P_03 - Q_03;
            float PQ_04 = P_04 - Q_04;
            float PQ_05 = P_05 - Q_05;
            float PQ_06 = P_06 - Q_06;
            float PQ_07 = P_07 - Q_07;

            float f_00 = PQ_00 * T_00;
            float f_01 = PQ_01 * T_01;
            float f_02 = PQ_02 * T_02;
            float f_03 = PQ_03 * T_03;
            float f_04 = PQ_04 * T_04;
            float f_05 = PQ_05 * T_05;
            float f_06 = PQ_06 * T_06;
            float f_07 = PQ_07 * T_07;

            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            float Ynm01_d0 = Yn0_d0 - Ym1_d0;
            float Ynm01_d1 = Yn0_d1 - Ym1_d1;
            float Ynm02_d0 = Yn0_d0 - Ym2_d0;
            float Ynm02_d1 = Yn0_d1 - Ym2_d1;
            float Ynm03_d0 = Yn0_d0 - Ym3_d0;
            float Ynm03_d1 = Yn0_d1 - Ym3_d1;
            float Ynm04_d0 = Yn0_d0 - Ym4_d0;
            float Ynm04_d1 = Yn0_d1 - Ym4_d1;
            float Ynm05_d0 = Yn0_d0 - Ym5_d0;
            float Ynm05_d1 = Yn0_d1 - Ym5_d1;
            float Ynm06_d0 = Yn0_d0 - Ym6_d0;
            float Ynm06_d1 = Yn0_d1 - Ym6_d1;
            float Ynm07_d0 = Yn0_d0 - Ym7_d0;
            float Ynm07_d1 = Yn0_d1 - Ym7_d1;


            float dC00_d0 = Ynm00_d0 * f_00;
            float dC00_d1 = Ynm00_d1 * f_00;
            float dC01_d0 = Ynm01_d0 * f_01;
            float dC01_d1 = Ynm01_d1 * f_01;
            float dC02_d0 = Ynm02_d0 * f_02;
            float dC02_d1 = Ynm02_d1 * f_02;
            float dC03_d0 = Ynm03_d0 * f_03;
            float dC03_d1 = Ynm03_d1 * f_03;
            float dC04_d0 = Ynm04_d0 * f_04;
            float dC04_d1 = Ynm04_d1 * f_04;
            float dC05_d0 = Ynm05_d0 * f_05;
            float dC05_d1 = Ynm05_d1 * f_05;
            float dC06_d0 = Ynm06_d0 * f_06;
            float dC06_d1 = Ynm06_d1 * f_06;
            float dC07_d0 = Ynm07_d0 * f_07;
            float dC07_d1 = Ynm07_d1 * f_07;

            // Accumulate the gradients
            float dC08_d0 = dC00_d0 + dC01_d0;
            float dC09_d0 = dC02_d0 + dC03_d0;
            float dC010_d0 = dC04_d0 + dC05_d0;
            float dC011_d0 = dC06_d0 + dC07_d0;
            float dC012_d0 = dC08_d0 + dC09_d0;
            float dC013_d0 = dC010_d0 + dC011_d0;
            float dC014_d0 = dC012_d0 + dC013_d0;
            dC0_d0 += dC014_d0;
            float dC08_d1 = dC00_d1 + dC01_d1;
            float dC09_d1 = dC02_d1 + dC03_d1;
            float dC010_d1 = dC04_d1 + dC05_d1;
            float dC011_d1 = dC06_d1 + dC07_d1;
            float dC012_d1 = dC08_d1 + dC09_d1;
            float dC013_d1 = dC010_d1 + dC011_d1;
            float dC014_d1 = dC012_d1 + dC013_d1;
            dC0_d1 += dC014_d1;
        }
        // Compute remaining
    	for(; m < N; m++, mD += D) {
            float Ym0_d0 = Y[mD];
            float Ym0_d1 = Y[mD+1];
            float P_00 = P[nN+m];
            float T_00 = Q[nN+m];
            float Q_00 = T_00 / sum_Q;
            float PQ_00 = P_00 - Q_00;
            float f_00 = PQ_00 * T_00;
            float Ynm00_d0 = Yn0_d0 - Ym0_d0;
            float Ynm00_d1 = Yn0_d1 - Ym0_d1;
            dC0_d0 += Ynm00_d0 * f_00;
            dC0_d1 += Ynm00_d1 * f_00;
    	}

        float edC0_d0 = eta * dC0_d0;
        float edC0_d1 = eta * dC0_d1;
        float muY0_d0 = momentum * uY0_d0;
        float muY0_d1 = momentum * uY0_d1;
        uY0_d0 = muY0_d0 - edC0_d0;
        uY0_d1 = muY0_d1 - edC0_d1;
        uY[nD] = uY0_d0;
        uY[nD+1] = uY0_d1;
    }

    float mean_d0 = 0;
    float mean_d1 = 0;
    // Compute the mean
    for (n = 0, nD = 0; n + 4 <= N; n += 4, nD += 4*D) {
        float Y0_d0 = Y[nD];
        float Y0_d1 = Y[nD+1];
        float Y1_d0 = Y[nD+2];
        float Y1_d1 = Y[nD+3];
        float Y2_d0 = Y[nD+4];
        float Y2_d1 = Y[nD+5];
        float Y3_d0 = Y[nD+6];
        float Y3_d1 = Y[nD+7];

        float uY0_d0 = uY[nD];
        float uY0_d1 = uY[nD+1];
        float uY1_d0 = uY[nD+2];
        float uY1_d1 = uY[nD+3];
        float uY2_d0 = uY[nD+4];
        float uY2_d1 = uY[nD+5];
        float uY3_d0 = uY[nD+6];
        float uY3_d1 = uY[nD+7];

        float nY0_d0 = Y0_d0 + uY0_d0;
        float nY0_d1 = Y0_d1 + uY0_d1;
        float nY1_d0 = Y1_d0 + uY1_d0;
        float nY1_d1 = Y1_d1 + uY1_d1;
        float nY2_d0 = Y2_d0 + uY2_d0;
        float nY2_d1 = Y2_d1 + uY2_d1;
        float nY3_d0 = Y3_d0 + uY3_d0;
        float nY3_d1 = Y3_d1 + uY3_d1;

        mean_d0 += nY0_d0;
        mean_d1 += nY0_d1;
        mean_d0 += nY1_d0;
        mean_d1 += nY1_d1;
        mean_d0 += nY2_d0;
        mean_d1 += nY2_d1;
        mean_d0 += nY3_d0;
        mean_d1 += nY3_d1;

        Y[nD] = nY0_d0;
        Y[nD+1] = nY0_d1;
        Y[nD+2] = nY1_d0;
        Y[nD+3] = nY1_d1;
        Y[nD+4] = nY2_d0;
        Y[nD+5] = nY2_d1;
        Y[nD+6] = nY3_d0;
        Y[nD+7] = nY3_d1;
    }
    for (; n < N; n++, nD += D) {
        float Y0_d0 = Y[nD];
        float Y0_d1 = Y[nD+1];
        float uY0_d0 = uY[nD];
        float uY0_d1 = uY[nD+1];

        float nY0_d0 = Y0_d0 + uY0_d0;
        float nY0_d1 = Y0_d1 + uY0_d1;

        mean_d0 += nY0_d0;
        mean_d1 += nY0_d1;

        Y[nD] = nY0_d0;
        Y[nD+1] = nY0_d1;
    }

    mean_d0 /= (double) N;
    mean_d1 /= (double) N;

    // Substract the mean
    for (n = 0, nD = 0; n + 4 <= N; n += 4, nD += 4*D) {
        float Y0_d0 = Y[nD];
        float Y0_d1 = Y[nD+1];
        float Y1_d0 = Y[nD+2];
        float Y1_d1 = Y[nD+3];
        float Y2_d0 = Y[nD+4];
        float Y2_d1 = Y[nD+5];
        float Y3_d0 = Y[nD+6];
        float Y3_d1 = Y[nD+7];

        float nY0_d0 = Y0_d0 - mean_d0;
        float nY0_d1 = Y0_d1 - mean_d1;
        float nY1_d0 = Y1_d0 - mean_d0;
        float nY1_d1 = Y1_d1 - mean_d1;
        float nY2_d0 = Y2_d0 - mean_d0;
        float nY2_d1 = Y2_d1 - mean_d1;
        float nY3_d0 = Y3_d0 - mean_d0;
        float nY3_d1 = Y3_d1 - mean_d1;

        Y[nD] = nY0_d0;
        Y[nD+1] = nY0_d1;
        Y[nD+2] = nY1_d0;
        Y[nD+3] = nY1_d1;
        Y[nD+4] = nY2_d0;
        Y[nD+5] = nY2_d1;
        Y[nD+6] = nY3_d0;
        Y[nD+7] = nY3_d1;
    }
    for (; n < N; n++, nD += D) {
        float Y0_d0 = Y[nD];
        float Y0_d1 = Y[nD+1];

        float nY0_d0 = Y0_d0 - mean_d0;
        float nY0_d1 = Y0_d1 - mean_d1;

        Y[nD] = nY0_d0;
        Y[nD+1] = nY0_d1;
    }
}

inline void fused(float* Y, float* P, float* Q, float sum_Q, int N,
			 int D, float* dC, float* uY, float momentum,
			 float eta) {
    // Perform the computation of the gradient
    int nN = 0;
    int nD = 0;
    for(int n = 0; n < N; n++) {
    	int mD = 0;
    	for(int m = 0; m < N; m++) {
			float mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
			for (int d = 0; d < D; d++) {
				dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
			}
    		mD += D;
    	}

        uY[nD] = momentum * uY[nD] - eta * dC[nD];
        uY[nD+1] = momentum * uY[nD+1] - eta * dC[nD+1];

    	nN += N;
    	nD += D;
    }
    float mean0 = 0;
	float mean1 = 0;
    nD = 0;
    for (int n = 0; n < N; n++, nD += D) {
        Y[nD] = Y[nD] +  uY[nD];
        Y[nD+1] = Y[nD+1] +  uY[nD+1];
        mean0 += Y[nD];
        mean1 += Y[nD+1];
    }

    mean0 /= (double) N;
    mean1 /= (double) N;

    // Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++, nD += D) {
		Y[nD] -= mean0;
		Y[nD+1] -= mean1;
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

// Normalize X substracting mean and
inline void normalize(float* X, int N, int D) {
	int nD = 0;

	float mean0 = 0;
	float mean1 = 0;

	for(int n = 0; n < N; n++) {
		mean0 += X[nD];
		mean1 += X[nD+1];
        nD += D;
	}
	mean0 /= (double) N;
	mean1 /= (double) N;
	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		X[nD] -= mean0;
		X[nD+1] -= mean1;
        nD += D;
	}
}


inline void base_version(float* Y, float* P, float* Q, float sum_Q, int N,
						 int D, float* dC, float* uY, float momentum,
						 float eta) {
	gradient_computation(Y, P, Q, sum_Q, N, D, dC);
	gradient_update(Y, dC, uY, N, D, momentum, eta);
    normalize(Y, N, D);
}

#endif
