#ifndef TRAINING_STEP_H
#define TRAINING_STEP_H

#include <stdio.h>
#include <immintrin.h>

static inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2,
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

static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline void unfold_d_unfold_nx8_mx8_vec(float* Y, float* P, float* Q, float sum_Q,
									int N, int D, float* dC, float* uY,
									float momentum, float eta) {
	const int M = 8;
	const int K = 8;

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
		__m256 dCn_4 = _mm256_setzero_ps();
		__m256 dCn_5 = _mm256_setzero_ps();
		__m256 dCn_6 = _mm256_setzero_ps();
		__m256 dCn_7 = _mm256_setzero_ps();

		__m256 Yn_0 = _mm256_loadu_ps(Y+nD);
		__m256 Yn_1 = _mm256_loadu_ps(Y+nD+8);
		__m256 Yn_2 = _mm256_loadu_ps(Y+nD+16);
		__m256 Yn_3 = _mm256_loadu_ps(Y+nD+24);
		__m256 Yn_4 = _mm256_loadu_ps(Y+nD+32);
		__m256 Yn_5 = _mm256_loadu_ps(Y+nD+40);
		__m256 Yn_6 = _mm256_loadu_ps(Y+nD+48);
		__m256 Yn_7 = _mm256_loadu_ps(Y+nD+56);

        __m256 uY_0 = _mm256_loadu_ps(uY+nD);
        __m256 uY_1 = _mm256_loadu_ps(uY+nD+8);
        __m256 uY_2 = _mm256_loadu_ps(uY+nD+16);
        __m256 uY_3 = _mm256_loadu_ps(uY+nD+24);
        __m256 uY_4 = _mm256_loadu_ps(uY+nD+32);
        __m256 uY_5 = _mm256_loadu_ps(uY+nD+40);
        __m256 uY_6 = _mm256_loadu_ps(uY+nD+48);
        __m256 uY_7 = _mm256_loadu_ps(uY+nD+56);

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

            __m256 p_40 = _mm256_load_ps(P+nN+16*N+m);
			__m256 p_41 = _mm256_load_ps(P+nN+17*N+m);
			__m256 p_42 = _mm256_load_ps(P+nN+18*N+m);
			__m256 p_43 = _mm256_load_ps(P+nN+19*N+m);

            __m256 p_50 = _mm256_load_ps(P+nN+20*N+m);
			__m256 p_51 = _mm256_load_ps(P+nN+21*N+m);
			__m256 p_52 = _mm256_load_ps(P+nN+22*N+m);
			__m256 p_53 = _mm256_load_ps(P+nN+23*N+m);

            __m256 p_60 = _mm256_load_ps(P+nN+24*N+m);
			__m256 p_61 = _mm256_load_ps(P+nN+25*N+m);
			__m256 p_62 = _mm256_load_ps(P+nN+26*N+m);
			__m256 p_63 = _mm256_load_ps(P+nN+27*N+m);

            __m256 p_70 = _mm256_load_ps(P+nN+28*N+m);
			__m256 p_71 = _mm256_load_ps(P+nN+29*N+m);
			__m256 p_72 = _mm256_load_ps(P+nN+30*N+m);
			__m256 p_73 = _mm256_load_ps(P+nN+31*N+m);


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

            __m256 q_40 = _mm256_load_ps(Q+nN+16*N+m);
			__m256 q_41 = _mm256_load_ps(Q+nN+17*N+m);
			__m256 q_42 = _mm256_load_ps(Q+nN+18*N+m);
			__m256 q_43 = _mm256_load_ps(Q+nN+19*N+m);

            __m256 q_50 = _mm256_load_ps(Q+nN+20*N+m);
			__m256 q_51 = _mm256_load_ps(Q+nN+21*N+m);
			__m256 q_52 = _mm256_load_ps(Q+nN+22*N+m);
			__m256 q_53 = _mm256_load_ps(Q+nN+23*N+m);

            __m256 q_60 = _mm256_load_ps(Q+nN+24*N+m);
			__m256 q_61 = _mm256_load_ps(Q+nN+25*N+m);
			__m256 q_62 = _mm256_load_ps(Q+nN+26*N+m);
			__m256 q_63 = _mm256_load_ps(Q+nN+27*N+m);

            __m256 q_70 = _mm256_load_ps(Q+nN+28*N+m);
			__m256 q_71 = _mm256_load_ps(Q+nN+29*N+m);
			__m256 q_72 = _mm256_load_ps(Q+nN+30*N+m);
			__m256 q_73 = _mm256_load_ps(Q+nN+31*N+m);

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

			__m256 qq_40 = _mm256_mul_ps(q_40, inv_q);
			__m256 qq_41 = _mm256_mul_ps(q_41, inv_q);
			__m256 qq_42 = _mm256_mul_ps(q_42, inv_q);
			__m256 qq_43 = _mm256_mul_ps(q_43, inv_q);

			__m256 qq_50 = _mm256_mul_ps(q_50, inv_q);
			__m256 qq_51 = _mm256_mul_ps(q_51, inv_q);
			__m256 qq_52 = _mm256_mul_ps(q_52, inv_q);
			__m256 qq_53 = _mm256_mul_ps(q_53, inv_q);

			__m256 qq_60 = _mm256_mul_ps(q_60, inv_q);
			__m256 qq_61 = _mm256_mul_ps(q_61, inv_q);
			__m256 qq_62 = _mm256_mul_ps(q_62, inv_q);
			__m256 qq_63 = _mm256_mul_ps(q_63, inv_q);

			__m256 qq_70 = _mm256_mul_ps(q_70, inv_q);
			__m256 qq_71 = _mm256_mul_ps(q_71, inv_q);
			__m256 qq_72 = _mm256_mul_ps(q_72, inv_q);
			__m256 qq_73 = _mm256_mul_ps(q_73, inv_q);
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

			__m256 Ynm_40 = _mm256_sub_ps(Yn_4, Ym_0);
			__m256 Ynm_41 = _mm256_sub_ps(Yn_4, Ym_1);
			__m256 Ynm_42 = _mm256_sub_ps(Yn_4, Ym_2);
			__m256 Ynm_43 = _mm256_sub_ps(Yn_4, Ym_3);
			__m256 Ynm_44 = _mm256_sub_ps(Yn_4, Ym_4);
			__m256 Ynm_45 = _mm256_sub_ps(Yn_4, Ym_5);
			__m256 Ynm_46 = _mm256_sub_ps(Yn_4, Ym_6);
			__m256 Ynm_47 = _mm256_sub_ps(Yn_4, Ym_7);

			__m256 Ynm_50 = _mm256_sub_ps(Yn_5, Ym_0);
			__m256 Ynm_51 = _mm256_sub_ps(Yn_5, Ym_1);
			__m256 Ynm_52 = _mm256_sub_ps(Yn_5, Ym_2);
			__m256 Ynm_53 = _mm256_sub_ps(Yn_5, Ym_3);
			__m256 Ynm_54 = _mm256_sub_ps(Yn_5, Ym_4);
			__m256 Ynm_55 = _mm256_sub_ps(Yn_5, Ym_5);
			__m256 Ynm_56 = _mm256_sub_ps(Yn_5, Ym_6);
			__m256 Ynm_57 = _mm256_sub_ps(Yn_5, Ym_7);

			__m256 Ynm_60 = _mm256_sub_ps(Yn_6, Ym_0);
			__m256 Ynm_61 = _mm256_sub_ps(Yn_6, Ym_1);
			__m256 Ynm_62 = _mm256_sub_ps(Yn_6, Ym_2);
			__m256 Ynm_63 = _mm256_sub_ps(Yn_6, Ym_3);
			__m256 Ynm_64 = _mm256_sub_ps(Yn_6, Ym_4);
			__m256 Ynm_65 = _mm256_sub_ps(Yn_6, Ym_5);
			__m256 Ynm_66 = _mm256_sub_ps(Yn_6, Ym_6);
			__m256 Ynm_67 = _mm256_sub_ps(Yn_6, Ym_7);

			__m256 Ynm_70 = _mm256_sub_ps(Yn_7, Ym_0);
			__m256 Ynm_71 = _mm256_sub_ps(Yn_7, Ym_1);
			__m256 Ynm_72 = _mm256_sub_ps(Yn_7, Ym_2);
			__m256 Ynm_73 = _mm256_sub_ps(Yn_7, Ym_3);
			__m256 Ynm_74 = _mm256_sub_ps(Yn_7, Ym_4);
			__m256 Ynm_75 = _mm256_sub_ps(Yn_7, Ym_5);
			__m256 Ynm_76 = _mm256_sub_ps(Yn_7, Ym_6);
			__m256 Ynm_77 = _mm256_sub_ps(Yn_7, Ym_7);
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

			__m256 pq_40 = _mm256_sub_ps(p_40, qq_40);
			__m256 pq_41 = _mm256_sub_ps(p_41, qq_41);
			__m256 pq_42 = _mm256_sub_ps(p_42, qq_42);
			__m256 pq_43 = _mm256_sub_ps(p_43, qq_43);

			__m256 pq_50 = _mm256_sub_ps(p_50, qq_50);
			__m256 pq_51 = _mm256_sub_ps(p_51, qq_51);
			__m256 pq_52 = _mm256_sub_ps(p_52, qq_52);
			__m256 pq_53 = _mm256_sub_ps(p_53, qq_53);

			__m256 pq_60 = _mm256_sub_ps(p_60, qq_60);
			__m256 pq_61 = _mm256_sub_ps(p_61, qq_61);
			__m256 pq_62 = _mm256_sub_ps(p_62, qq_62);
			__m256 pq_63 = _mm256_sub_ps(p_63, qq_63);

			__m256 pq_70 = _mm256_sub_ps(p_70, qq_70);
			__m256 pq_71 = _mm256_sub_ps(p_71, qq_71);
			__m256 pq_72 = _mm256_sub_ps(p_72, qq_72);
			__m256 pq_73 = _mm256_sub_ps(p_73, qq_73);
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

			__m256 pqq_40 = _mm256_mul_ps(pq_40, q_40);
			__m256 pqq_41 = _mm256_mul_ps(pq_41, q_41);
			__m256 pqq_42 = _mm256_mul_ps(pq_42, q_42);
			__m256 pqq_43 = _mm256_mul_ps(pq_43, q_43);

			__m256 pqq_50 = _mm256_mul_ps(pq_50, q_50);
			__m256 pqq_51 = _mm256_mul_ps(pq_51, q_51);
			__m256 pqq_52 = _mm256_mul_ps(pq_52, q_52);
			__m256 pqq_53 = _mm256_mul_ps(pq_53, q_53);

			__m256 pqq_60 = _mm256_mul_ps(pq_60, q_60);
			__m256 pqq_61 = _mm256_mul_ps(pq_61, q_61);
			__m256 pqq_62 = _mm256_mul_ps(pq_62, q_62);
			__m256 pqq_63 = _mm256_mul_ps(pq_63, q_63);

			__m256 pqq_70 = _mm256_mul_ps(pq_70, q_70);
			__m256 pqq_71 = _mm256_mul_ps(pq_71, q_71);
			__m256 pqq_72 = _mm256_mul_ps(pq_72, q_72);
			__m256 pqq_73 = _mm256_mul_ps(pq_73, q_73);


			__m256 f_00 = pqq_00, f_01 = pqq_00, f_02 = pqq_01, f_03 = pqq_01;
			__m256 f_04 = pqq_02, f_05 = pqq_02, f_06 = pqq_03, f_07 = pqq_03;

			__m256 f_10 = pqq_10, f_11 = pqq_10, f_12 = pqq_11, f_13 = pqq_11;
			__m256 f_14 = pqq_12, f_15 = pqq_12, f_16 = pqq_13, f_17 = pqq_13;

			__m256 f_20 = pqq_20, f_21 = pqq_20, f_22 = pqq_21, f_23 = pqq_21;
			__m256 f_24 = pqq_22, f_25 = pqq_22, f_26 = pqq_23, f_27 = pqq_23;

			__m256 f_30 = pqq_30, f_31 = pqq_30, f_32 = pqq_31, f_33 = pqq_31;
			__m256 f_34 = pqq_32, f_35 = pqq_32, f_36 = pqq_33, f_37 = pqq_33;

			__m256 f_40 = pqq_40, f_41 = pqq_40, f_42 = pqq_41, f_43 = pqq_41;
			__m256 f_44 = pqq_42, f_45 = pqq_42, f_46 = pqq_43, f_47 = pqq_43;

			__m256 f_50 = pqq_50, f_51 = pqq_50, f_52 = pqq_51, f_53 = pqq_51;
			__m256 f_54 = pqq_52, f_55 = pqq_52, f_56 = pqq_53, f_57 = pqq_53;

			__m256 f_60 = pqq_60, f_61 = pqq_60, f_62 = pqq_61, f_63 = pqq_61;
			__m256 f_64 = pqq_62, f_65 = pqq_62, f_66 = pqq_63, f_67 = pqq_63;

			__m256 f_70 = pqq_70, f_71 = pqq_70, f_72 = pqq_71, f_73 = pqq_71;
			__m256 f_74 = pqq_72, f_75 = pqq_72, f_76 = pqq_73, f_77 = pqq_73;

			// Transpose
			transpose8_ps(f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07);
			transpose8_ps(f_10, f_11, f_12, f_13, f_14, f_15, f_16, f_17);
			transpose8_ps(f_20, f_21, f_22, f_23, f_24, f_25, f_26, f_27);
			transpose8_ps(f_30, f_31, f_32, f_33, f_34, f_35, f_36, f_37);
			transpose8_ps(f_40, f_41, f_42, f_43, f_44, f_45, f_46, f_47);
			transpose8_ps(f_50, f_51, f_52, f_53, f_54, f_55, f_56, f_57);
			transpose8_ps(f_60, f_61, f_62, f_63, f_64, f_65, f_66, f_67);
			transpose8_ps(f_70, f_71, f_72, f_73, f_74, f_75, f_76, f_77);

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

			__m256 dC_40 = _mm256_mul_ps(f_40, Ynm_40);
			__m256 dC_41 = _mm256_mul_ps(f_41, Ynm_41);
			__m256 dC_42 = _mm256_mul_ps(f_42, Ynm_42);
			__m256 dC_43 = _mm256_mul_ps(f_43, Ynm_43);
			__m256 dC_44 = _mm256_mul_ps(f_44, Ynm_44);
			__m256 dC_45 = _mm256_mul_ps(f_45, Ynm_45);
			__m256 dC_46 = _mm256_mul_ps(f_46, Ynm_46);
			__m256 dC_47 = _mm256_mul_ps(f_47, Ynm_47);
			__m256 _dC_40 = _mm256_add_ps(dC_40, dC_41);
			__m256 _dC_41 = _mm256_add_ps(dC_42, dC_43);
			__m256 _dC_42 = _mm256_add_ps(dC_44, dC_45);
			__m256 _dC_43 = _mm256_add_ps(dC_46, dC_47);
			__m256 _dC_44 = _mm256_add_ps(_dC_40, _dC_41);
			__m256 _dC_45 = _mm256_add_ps(_dC_42, _dC_43);
			__m256 _dC_46 = _mm256_add_ps(_dC_44, _dC_45);

			__m256 dC_50 = _mm256_mul_ps(f_50, Ynm_50);
			__m256 dC_51 = _mm256_mul_ps(f_51, Ynm_51);
			__m256 dC_52 = _mm256_mul_ps(f_52, Ynm_52);
			__m256 dC_53 = _mm256_mul_ps(f_53, Ynm_53);
			__m256 dC_54 = _mm256_mul_ps(f_54, Ynm_54);
			__m256 dC_55 = _mm256_mul_ps(f_55, Ynm_55);
			__m256 dC_56 = _mm256_mul_ps(f_56, Ynm_56);
			__m256 dC_57 = _mm256_mul_ps(f_57, Ynm_57);
			__m256 _dC_50 = _mm256_add_ps(dC_50, dC_51);
			__m256 _dC_51 = _mm256_add_ps(dC_52, dC_53);
			__m256 _dC_52 = _mm256_add_ps(dC_54, dC_55);
			__m256 _dC_53 = _mm256_add_ps(dC_56, dC_57);
			__m256 _dC_54 = _mm256_add_ps(_dC_50, _dC_51);
			__m256 _dC_55 = _mm256_add_ps(_dC_52, _dC_53);
			__m256 _dC_56 = _mm256_add_ps(_dC_54, _dC_55);

			__m256 dC_60 = _mm256_mul_ps(f_60, Ynm_60);
			__m256 dC_61 = _mm256_mul_ps(f_61, Ynm_61);
			__m256 dC_62 = _mm256_mul_ps(f_62, Ynm_62);
			__m256 dC_63 = _mm256_mul_ps(f_63, Ynm_63);
			__m256 dC_64 = _mm256_mul_ps(f_64, Ynm_64);
			__m256 dC_65 = _mm256_mul_ps(f_65, Ynm_65);
			__m256 dC_66 = _mm256_mul_ps(f_66, Ynm_66);
			__m256 dC_67 = _mm256_mul_ps(f_67, Ynm_67);
			__m256 _dC_60 = _mm256_add_ps(dC_60, dC_61);
			__m256 _dC_61 = _mm256_add_ps(dC_62, dC_63);
			__m256 _dC_62 = _mm256_add_ps(dC_64, dC_65);
			__m256 _dC_63 = _mm256_add_ps(dC_66, dC_67);
			__m256 _dC_64 = _mm256_add_ps(_dC_60, _dC_61);
			__m256 _dC_65 = _mm256_add_ps(_dC_62, _dC_63);
			__m256 _dC_66 = _mm256_add_ps(_dC_64, _dC_65);

			__m256 dC_70 = _mm256_mul_ps(f_70, Ynm_70);
			__m256 dC_71 = _mm256_mul_ps(f_71, Ynm_71);
			__m256 dC_72 = _mm256_mul_ps(f_72, Ynm_72);
			__m256 dC_73 = _mm256_mul_ps(f_73, Ynm_73);
			__m256 dC_74 = _mm256_mul_ps(f_74, Ynm_74);
			__m256 dC_75 = _mm256_mul_ps(f_75, Ynm_75);
			__m256 dC_76 = _mm256_mul_ps(f_76, Ynm_76);
			__m256 dC_77 = _mm256_mul_ps(f_77, Ynm_77);
			__m256 _dC_70 = _mm256_add_ps(dC_70, dC_71);
			__m256 _dC_71 = _mm256_add_ps(dC_72, dC_73);
			__m256 _dC_72 = _mm256_add_ps(dC_74, dC_75);
			__m256 _dC_73 = _mm256_add_ps(dC_76, dC_77);
			__m256 _dC_74 = _mm256_add_ps(_dC_70, _dC_71);
			__m256 _dC_75 = _mm256_add_ps(_dC_72, _dC_73);
			__m256 _dC_76 = _mm256_add_ps(_dC_74, _dC_75);


			dCn_0 = _mm256_add_ps(dCn_0, _dC_06);
			dCn_1 = _mm256_add_ps(dCn_1, _dC_16);
			dCn_2 = _mm256_add_ps(dCn_2, _dC_26);
			dCn_3 = _mm256_add_ps(dCn_3, _dC_36);
			dCn_4 = _mm256_add_ps(dCn_4, _dC_46);
			dCn_5 = _mm256_add_ps(dCn_5, _dC_56);
			dCn_6 = _mm256_add_ps(dCn_6, _dC_66);
			dCn_7 = _mm256_add_ps(dCn_7, _dC_76);

		}

		__m256 uYx_0 = _mm256_mul_ps(mom, uY_0);
		__m256 uYx_1 = _mm256_mul_ps(mom, uY_1);
		__m256 uYx_2 = _mm256_mul_ps(mom, uY_2);
		__m256 uYx_3 = _mm256_mul_ps(mom, uY_3);
		__m256 uYx_4 = _mm256_mul_ps(mom, uY_4);
		__m256 uYx_5 = _mm256_mul_ps(mom, uY_5);
		__m256 uYx_6 = _mm256_mul_ps(mom, uY_6);
		__m256 uYx_7 = _mm256_mul_ps(mom, uY_7);

		__m256 uYxx_0 = _mm256_fmadd_ps(m_eta, dCn_0, uYx_0);
		__m256 uYxx_1 = _mm256_fmadd_ps(m_eta, dCn_1, uYx_1);
		__m256 uYxx_2 = _mm256_fmadd_ps(m_eta, dCn_2, uYx_2);
		__m256 uYxx_3 = _mm256_fmadd_ps(m_eta, dCn_3, uYx_3);
		__m256 uYxx_4 = _mm256_fmadd_ps(m_eta, dCn_4, uYx_4);
		__m256 uYxx_5 = _mm256_fmadd_ps(m_eta, dCn_5, uYx_5);
		__m256 uYxx_6 = _mm256_fmadd_ps(m_eta, dCn_6, uYx_6);
		__m256 uYxx_7 = _mm256_fmadd_ps(m_eta, dCn_7, uYx_7);

		_mm256_storeu_ps(uY+nD, uYxx_0);
		_mm256_storeu_ps(uY+nD+8, uYxx_1);
		_mm256_storeu_ps(uY+nD+16, uYxx_2);
		_mm256_storeu_ps(uY+nD+24, uYxx_3);
		_mm256_storeu_ps(uY+nD+32, uYxx_4);
		_mm256_storeu_ps(uY+nD+40, uYxx_5);
		_mm256_storeu_ps(uY+nD+48, uYxx_6);
		_mm256_storeu_ps(uY+nD+56, uYxx_7);

		nN += 4 * K * N;
		nD += 4 * K * D;
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

    int n;
    const int B = 8;
    const int V = 4; // samples that fit in a AVX register
    for (n = 0, nD = 0; n + V*B <= N; n += V*B, nD += V*B*D) {
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
    for (; n + V <= N; n += V, nD += V*D) {
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
	for (n = 0, nD = 0; n + V * K <= N; n += V*B, nD += V*B*D) {
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
	for (; n + V <= N; n += V, nD += V*D) {
		__m256 _Y = _mm256_loadu_ps(Y+nD);
		_Y = _mm256_sub_ps(_Y, _mean);
		_mm256_storeu_ps(Y+nD, _Y);
	}
	for(; n < N; n++, nD += D) {
		Y[nD] -= _mean[0];
		Y[nD+1] -= _mean[1];
	}
}

inline float low_dimensional_affinities_fast_vec_avx(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 8) ? 8: N;

	__m256 ones = _mm256_set1_ps(1.);
	__m256 zeros = _mm256_setzero_ps();
	__m256 cum_sum = _mm256_setzero_ps();

	// Loops for block size 32
	for (int iB = 0; iB < N; iB += B) {
		for (int jB = iB; jB < N; jB += B) {

			// Loops for microblocks of size 4
			for (int iK = iB; iK < iB + B; iK += K) {
				for (int jK = jB; jK < jB + B; jK += K) {
					// In case the result DD and symetric are the same
                    if (jK < iK) {continue;}

					// Case where the resulting block of BxB needs only to
					// be computed the upper triangle part
                    const float* XnD = X + iK * D;
					const float* XmD = X + jK * D;
					float* DDij = DD + iK * N + jK;
                    float* DDji = DD + jK * N + iK;

					__m256 Xn_0 = _mm256_set_ps(XnD[1], XnD[0], XnD[1], XnD[0], XnD[1], XnD[0], XnD[1], XnD[0]);
					__m256 Xn_1 = _mm256_set_ps(XnD[3], XnD[2], XnD[3], XnD[2], XnD[3], XnD[2], XnD[3], XnD[2]);
					__m256 Xn_2 = _mm256_set_ps(XnD[5], XnD[4], XnD[5], XnD[4], XnD[5], XnD[4], XnD[5], XnD[4]);
					__m256 Xn_3 = _mm256_set_ps(XnD[7], XnD[6], XnD[7], XnD[6], XnD[7], XnD[6], XnD[7], XnD[6]);
					__m256 Xn_4 = _mm256_set_ps(XnD[9], XnD[8], XnD[9], XnD[8], XnD[9], XnD[8], XnD[9], XnD[8]);
					__m256 Xn_5 = _mm256_set_ps(XnD[11], XnD[10], XnD[11], XnD[10], XnD[11], XnD[10], XnD[11], XnD[10]);
					__m256 Xn_6 = _mm256_set_ps(XnD[13], XnD[12], XnD[13], XnD[12], XnD[13], XnD[12], XnD[13], XnD[12]);
					__m256 Xn_7 = _mm256_set_ps(XnD[15], XnD[14], XnD[15], XnD[14], XnD[15], XnD[14], XnD[15], XnD[14]);

					__m256 Xm_0 = _mm256_loadu_ps(XmD);
					__m256 Xm_1 = _mm256_loadu_ps(XmD+8);

					__m256 diff_00 = _mm256_sub_ps(Xm_0, Xn_0);
					__m256 diff_01 = _mm256_sub_ps(Xm_0, Xn_1);
					__m256 diff_02 = _mm256_sub_ps(Xm_0, Xn_2);
					__m256 diff_03 = _mm256_sub_ps(Xm_0, Xn_3);
					__m256 diff_04 = _mm256_sub_ps(Xm_0, Xn_4);
					__m256 diff_05 = _mm256_sub_ps(Xm_0, Xn_5);
					__m256 diff_06 = _mm256_sub_ps(Xm_0, Xn_6);
					__m256 diff_07 = _mm256_sub_ps(Xm_0, Xn_7);
					__m256 diff_10 = _mm256_sub_ps(Xm_1, Xn_0);
					__m256 diff_11 = _mm256_sub_ps(Xm_1, Xn_1);
					__m256 diff_12 = _mm256_sub_ps(Xm_1, Xn_2);
					__m256 diff_13 = _mm256_sub_ps(Xm_1, Xn_3);
					__m256 diff_14 = _mm256_sub_ps(Xm_1, Xn_4);
					__m256 diff_15 = _mm256_sub_ps(Xm_1, Xn_5);
					__m256 diff_16 = _mm256_sub_ps(Xm_1, Xn_6);
					__m256 diff_17 = _mm256_sub_ps(Xm_1, Xn_7);

					__m256 diff_sq_00 = _mm256_mul_ps(diff_00, diff_00);
					__m256 diff_sq_01 = _mm256_mul_ps(diff_01, diff_01);
					__m256 diff_sq_02 = _mm256_mul_ps(diff_02, diff_02);
					__m256 diff_sq_03 = _mm256_mul_ps(diff_03, diff_03);
					__m256 diff_sq_04 = _mm256_mul_ps(diff_04, diff_04);
					__m256 diff_sq_05 = _mm256_mul_ps(diff_05, diff_05);
					__m256 diff_sq_06 = _mm256_mul_ps(diff_06, diff_06);
					__m256 diff_sq_07 = _mm256_mul_ps(diff_07, diff_07);
					__m256 diff_sq_10 = _mm256_mul_ps(diff_10, diff_10);
					__m256 diff_sq_11 = _mm256_mul_ps(diff_11, diff_11);
					__m256 diff_sq_12 = _mm256_mul_ps(diff_12, diff_12);
					__m256 diff_sq_13 = _mm256_mul_ps(diff_13, diff_13);
					__m256 diff_sq_14 = _mm256_mul_ps(diff_14, diff_14);
					__m256 diff_sq_15 = _mm256_mul_ps(diff_15, diff_15);
					__m256 diff_sq_16 = _mm256_mul_ps(diff_16, diff_16);
					__m256 diff_sq_17 = _mm256_mul_ps(diff_17, diff_17);

					__m256 diff_sq_shuf_00 = _mm256_shuffle_ps(diff_sq_00,
                        diff_sq_00, 177);
					__m256 diff_sq_shuf_01 = _mm256_shuffle_ps(diff_sq_01,
                        diff_sq_01, 177);
					__m256 diff_sq_shuf_02 = _mm256_shuffle_ps(diff_sq_02,
                        diff_sq_02, 177);
					__m256 diff_sq_shuf_03 = _mm256_shuffle_ps(diff_sq_03,
                        diff_sq_03, 177);
					__m256 diff_sq_shuf_04 = _mm256_shuffle_ps(diff_sq_04,
                        diff_sq_04, 177);
					__m256 diff_sq_shuf_05 = _mm256_shuffle_ps(diff_sq_05,
                        diff_sq_05, 177);
					__m256 diff_sq_shuf_06 = _mm256_shuffle_ps(diff_sq_06,
                        diff_sq_06, 177);
					__m256 diff_sq_shuf_07 = _mm256_shuffle_ps(diff_sq_07,
                        diff_sq_07, 177);
					__m256 diff_sq_shuf_10 = _mm256_shuffle_ps(diff_sq_10,
                        diff_sq_10, 177);
					__m256 diff_sq_shuf_11 = _mm256_shuffle_ps(diff_sq_11,
                        diff_sq_11, 177);
					__m256 diff_sq_shuf_12 = _mm256_shuffle_ps(diff_sq_12,
                        diff_sq_12, 177);
					__m256 diff_sq_shuf_13 = _mm256_shuffle_ps(diff_sq_13,
                        diff_sq_13, 177);
					__m256 diff_sq_shuf_14 = _mm256_shuffle_ps(diff_sq_14,
                        diff_sq_14, 177);
					__m256 diff_sq_shuf_15 = _mm256_shuffle_ps(diff_sq_15,
                        diff_sq_15, 177);
					__m256 diff_sq_shuf_16 = _mm256_shuffle_ps(diff_sq_16,
                        diff_sq_16, 177);
					__m256 diff_sq_shuf_17 = _mm256_shuffle_ps(diff_sq_17,
                        diff_sq_17, 177);

					__m256 norm_00x = _mm256_add_ps(diff_sq_00,
                                                    diff_sq_shuf_00);
					__m256 norm_01x = _mm256_add_ps(diff_sq_01,
                                                    diff_sq_shuf_01);
					__m256 norm_02x = _mm256_add_ps(diff_sq_02,
                                                    diff_sq_shuf_02);
					__m256 norm_03x = _mm256_add_ps(diff_sq_03,
                                                    diff_sq_shuf_03);
					__m256 norm_04x = _mm256_add_ps(diff_sq_04,
                                                    diff_sq_shuf_04);
					__m256 norm_05x = _mm256_add_ps(diff_sq_05,
                                                    diff_sq_shuf_05);
					__m256 norm_06x = _mm256_add_ps(diff_sq_06,
                                                    diff_sq_shuf_06);
					__m256 norm_07x = _mm256_add_ps(diff_sq_07,
                                                    diff_sq_shuf_07);
					__m256 norm_10x = _mm256_add_ps(diff_sq_10,
                                                    diff_sq_shuf_10);
					__m256 norm_11x = _mm256_add_ps(diff_sq_11,
                                                    diff_sq_shuf_11);
					__m256 norm_12x = _mm256_add_ps(diff_sq_12,
                                                    diff_sq_shuf_12);
					__m256 norm_13x = _mm256_add_ps(diff_sq_13,
                                                    diff_sq_shuf_13);
					__m256 norm_14x = _mm256_add_ps(diff_sq_14,
                                                    diff_sq_shuf_14);
					__m256 norm_15x = _mm256_add_ps(diff_sq_15,
                                                    diff_sq_shuf_15);
					__m256 norm_16x = _mm256_add_ps(diff_sq_16,
                                                    diff_sq_shuf_16);
					__m256 norm_17x = _mm256_add_ps(diff_sq_17,
                                                    diff_sq_shuf_17);


                    __m256 permute_00x = _mm256_permute2f128_ps(norm_00x,
                        norm_00x, 1);
                    __m256 permute_01x = _mm256_permute2f128_ps(norm_01x,
                        norm_01x, 1);
                    __m256 permute_02x = _mm256_permute2f128_ps(norm_02x,
                        norm_02x, 1);
                    __m256 permute_03x = _mm256_permute2f128_ps(norm_03x,
                        norm_03x, 1);
                    __m256 permute_04x = _mm256_permute2f128_ps(norm_04x,
                        norm_04x, 1);
                    __m256 permute_05x = _mm256_permute2f128_ps(norm_05x,
                        norm_05x, 1);
                    __m256 permute_06x = _mm256_permute2f128_ps(norm_06x,
                        norm_06x, 1);
                    __m256 permute_07x = _mm256_permute2f128_ps(norm_07x,
                        norm_07x, 1);
                    __m256 permute_10x = _mm256_permute2f128_ps(norm_10x,
                        norm_10x, 1);
                    __m256 permute_11x = _mm256_permute2f128_ps(norm_11x,
                        norm_11x, 1);
                    __m256 permute_12x = _mm256_permute2f128_ps(norm_12x,
                        norm_12x, 1);
                    __m256 permute_13x = _mm256_permute2f128_ps(norm_13x,
                        norm_13x, 1);
                    __m256 permute_14x = _mm256_permute2f128_ps(norm_14x,
                        norm_14x, 1);
                    __m256 permute_15x = _mm256_permute2f128_ps(norm_15x,
                        norm_15x, 1);
                    __m256 permute_16x = _mm256_permute2f128_ps(norm_16x,
                        norm_16x, 1);
                    __m256 permute_17x = _mm256_permute2f128_ps(norm_17x,
                        norm_07x, 1);

                    __m256 norm_0_low = _mm256_shuffle_ps(norm_00x, permute_00x, 136);
                    __m256 norm_0_high = _mm256_shuffle_ps(norm_10x, permute_10x, 136);
                    __m256 norm_1_low = _mm256_shuffle_ps(norm_01x, permute_01x, 136);
                    __m256 norm_1_high = _mm256_shuffle_ps(norm_11x, permute_11x, 136);
                    __m256 norm_2_low = _mm256_shuffle_ps(norm_02x, permute_02x, 136);
                    __m256 norm_2_high = _mm256_shuffle_ps(norm_12x, permute_12x, 136);
                    __m256 norm_3_low = _mm256_shuffle_ps(norm_03x, permute_03x, 136);
                    __m256 norm_3_high = _mm256_shuffle_ps(norm_13x, permute_13x, 136);
                    __m256 norm_4_low = _mm256_shuffle_ps(norm_04x, permute_04x, 136);
                    __m256 norm_4_high = _mm256_shuffle_ps(norm_14x, permute_14x, 136);
                    __m256 norm_5_low = _mm256_shuffle_ps(norm_05x, permute_05x, 136);
                    __m256 norm_5_high = _mm256_shuffle_ps(norm_15x, permute_15x, 136);
                    __m256 norm_6_low = _mm256_shuffle_ps(norm_06x, permute_06x, 136);
                    __m256 norm_6_high = _mm256_shuffle_ps(norm_16x, permute_16x, 136);
                    __m256 norm_7_low = _mm256_shuffle_ps(norm_07x, permute_07x, 136);
                    __m256 norm_7_high = _mm256_shuffle_ps(norm_17x, permute_17x, 136);

                    __m256 norm_0 = _mm256_permute2f128_ps(norm_0_low, norm_0_high, 32);
                    __m256 norm_1 = _mm256_permute2f128_ps(norm_1_low, norm_1_high, 32);
                    __m256 norm_2 = _mm256_permute2f128_ps(norm_2_low, norm_2_high, 32);
                    __m256 norm_3 = _mm256_permute2f128_ps(norm_3_low, norm_3_high, 32);
                    __m256 norm_4 = _mm256_permute2f128_ps(norm_4_low, norm_4_high, 32);
                    __m256 norm_5 = _mm256_permute2f128_ps(norm_5_low, norm_5_high, 32);
                    __m256 norm_6 = _mm256_permute2f128_ps(norm_6_low, norm_6_high, 32);
                    __m256 norm_7 = _mm256_permute2f128_ps(norm_7_low, norm_7_high, 32);

                    __m256 n_0 = _mm256_add_ps(norm_0, ones);
                    __m256 n_1 = _mm256_add_ps(norm_1, ones);
                    __m256 n_2 = _mm256_add_ps(norm_2, ones);
                    __m256 n_3 = _mm256_add_ps(norm_3, ones);
                    __m256 n_4 = _mm256_add_ps(norm_4, ones);
                    __m256 n_5 = _mm256_add_ps(norm_5, ones);
                    __m256 n_6 = _mm256_add_ps(norm_6, ones);
                    __m256 n_7 = _mm256_add_ps(norm_7, ones);

					// __m256 q_0 = _mm256_div_ps(ones, n_0);
					// __m256 q_1 = _mm256_div_ps(ones, n_1);
					// __m256 q_2 = _mm256_div_ps(ones, n_2);
					// __m256 q_3 = _mm256_div_ps(ones, n_3);
					// __m256 q_4 = _mm256_div_ps(ones, n_4);
					// __m256 q_5 = _mm256_div_ps(ones, n_5);
					// __m256 q_6 = _mm256_div_ps(ones, n_6);
					// __m256 q_7 = _mm256_div_ps(ones, n_7);
					__m256 q_0 = _mm256_rcp_ps(n_0);
					__m256 q_1 = _mm256_rcp_ps(n_1);
					__m256 q_2 = _mm256_rcp_ps(n_2);
					__m256 q_3 = _mm256_rcp_ps(n_3);
					__m256 q_4 = _mm256_rcp_ps(n_4);
					__m256 q_5 = _mm256_rcp_ps(n_5);
					__m256 q_6 = _mm256_rcp_ps(n_6);
					__m256 q_7 = _mm256_rcp_ps(n_7);

                    if (iK == jK) {
                        q_0 = _mm256_blend_ps(q_0, zeros, 1);
						q_1 = _mm256_blend_ps(q_1, zeros, 2);
						q_2 = _mm256_blend_ps(q_2, zeros, 4);
						q_3 = _mm256_blend_ps(q_3, zeros, 8);
						q_4 = _mm256_blend_ps(q_4, zeros, 16);
						q_5 = _mm256_blend_ps(q_5, zeros, 32);
						q_6 = _mm256_blend_ps(q_6, zeros, 64);
						q_7 = _mm256_blend_ps(q_7, zeros, 128);
                    }


					__m256 sum0 = _mm256_add_ps(q_0, q_1);
					__m256 sum1 = _mm256_add_ps(q_2, q_3);
					__m256 sum2 = _mm256_add_ps(q_4, q_5);
					__m256 sum3 = _mm256_add_ps(q_6, q_7);
                    __m256 sum4 = _mm256_add_ps(sum0, sum1);
                    __m256 sum5 = _mm256_add_ps(sum2, sum3);
                    __m256 sum6 = _mm256_add_ps(sum4, sum5);

					cum_sum = _mm256_add_ps(cum_sum, sum6);

					_mm256_store_ps(DDij, q_0);
					_mm256_store_ps(DDij+N, q_1);
					_mm256_store_ps(DDij+2*N, q_2);
					_mm256_store_ps(DDij+3*N, q_3);
					_mm256_store_ps(DDij+4*N, q_4);
					_mm256_store_ps(DDij+5*N, q_5);
					_mm256_store_ps(DDij+6*N, q_6);
					_mm256_store_ps(DDij+7*N, q_7);

	                if (jK > iK) {

                        transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                        cum_sum = _mm256_add_ps(cum_sum, sum6);

                        _mm256_store_ps(DDji, q_0);
					_mm256_store_ps(DDji+N, q_1);
					_mm256_store_ps(DDji+2*N, q_2);
					_mm256_store_ps(DDji+3*N, q_3);
					_mm256_store_ps(DDji+4*N, q_4);
					_mm256_store_ps(DDji+5*N, q_5);
					_mm256_store_ps(DDji+6*N, q_6);
					_mm256_store_ps(DDji+7*N, q_7);
                    }

				}
			} // End of K loop

		}
	} // End of B loop

	float sum_q = _mm256_reduce_add_ps(cum_sum);

	return sum_q;
}


inline void gradient_update_normalize_fast_vec_avx(float* Y, float* P,
                                    float* Q, float sum_Q,
									int N, int D, float* uY,
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

    int n;
    const int B = 8;
    const int V = 4; // samples that fit in a AVX register
    for (n = 0, nD = 0; n + V*B <= N; n += V*B, nD += V*B*D) {
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
    for (; n + V <= N; n += V, nD += V*D) {
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
	for (n = 0, nD = 0; n + V * K <= N; n += V*B, nD += V*B*D) {
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
	for (; n + V <= N; n += V, nD += V*D) {
		__m256 _Y = _mm256_loadu_ps(Y+nD);
		_Y = _mm256_sub_ps(_Y, _mean);
		_mm256_storeu_ps(Y+nD, _Y);
	}
	for(; n < N; n++, nD += D) {
		Y[nD] -= _mean[0];
		Y[nD+1] -= _mean[1];
	}
}


inline void fast_vectorize_avx(float* Y, float* P, float* Q, int N,
						 int D, float* dC, float* uY, float momentum,
						 float eta) {
    float sum_Q = low_dimensional_affinities_fast_vec_avx(Y, N, D, Q);
    gradient_update_normalize_fast_vec_avx(Y, P, Q, sum_Q, N, D, uY, momentum, eta);
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

inline float compute_low_dimensional_affinities(float* Y, int N, int no_dims,
                                                float* Q) {

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

inline void base_version(float* Y, float* P, float* Q, int N,
						 int D, float* dC, float* uY, float momentum,
						 float eta) {

    float sum_Q = compute_low_dimensional_affinities(Y, N, D, Q);
	gradient_computation(Y, P, Q, sum_Q, N, D, dC);
	gradient_update(Y, dC, uY, N, D, momentum, eta);
    normalize(Y, N, D);
}

#endif
