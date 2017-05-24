#ifndef COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H
#define COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H

#include <stdio.h>
#include <immintrin.h>

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

inline float blocking_32_block_8_unfold_sr_vec(float* X, int N, int D, float* DD) {
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

	// __m128 cum_sum_sh1 = _mm_shuffle_ps(cum_sum, cum_sum, 177);
	// cum_sum = _mm_add_ps(cum_sum, cum_sum_sh1);
	// __m128 cum_sum_sh2 = _mm_shuffle_ps(cum_sum, cum_sum, 78);
	// cum_sum = _mm_add_ps(cum_sum, cum_sum_sh2);

	float sum_q = _mm256_reduce_add_ps(cum_sum);
	// _mm_store_ss(&sum_q, cum_sum);

	return sum_q;
}

// Compute low dimensional affinities
inline float blocking_64_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;

	__m128 ones = _mm_set1_ps(1.);
	__m128 zeros = _mm_set1_ps(0.);
	__m128 cum_sum = _mm_set1_ps(0.);

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

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

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

						// __m128 q_0 = _mm_div_ps(ones, n_0);
						// __m128 q_1 = _mm_div_ps(ones, n_1);
						// __m128 q_2 = _mm_div_ps(ones, n_2);
						// __m128 q_3 = _mm_div_ps(ones, n_3);
						__m128 q_0 = _mm_rcp_ps(n_0);
						__m128 q_1 = _mm_rcp_ps(n_1);
						__m128 q_2 = _mm_rcp_ps(n_2);
						__m128 q_3 = _mm_rcp_ps(n_3);

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

						_MM_TRANSPOSE4_PS(q_0, q_1, q_2, q_3);

						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDji, q_0);
						_mm_store_ps(DDji+N, q_1);
						_mm_store_ps(DDji+2*N, q_2);
						_mm_store_ps(DDji+3*N, q_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop

	__m128 cum_sum_sh1 = _mm_shuffle_ps(cum_sum, cum_sum, 177);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh1);
	__m128 cum_sum_sh2 = _mm_shuffle_ps(cum_sum, cum_sum, 78);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh2);

	float sum_q;
	_mm_store_ss(&sum_q, cum_sum);

	return sum_q;
}

inline float blocking_32_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 32; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;

	__m128 ones = _mm_set1_ps(1.);
	__m128 zeros = _mm_set1_ps(0.);
	__m128 cum_sum = _mm_set1_ps(0.);

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

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

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

						// __m128 q_0 = _mm_div_ps(ones, n_0);
						// __m128 q_1 = _mm_div_ps(ones, n_1);
						// __m128 q_2 = _mm_div_ps(ones, n_2);
						// __m128 q_3 = _mm_div_ps(ones, n_3);
						__m128 q_0 = _mm_rcp_ps(n_0);
						__m128 q_1 = _mm_rcp_ps(n_1);
						__m128 q_2 = _mm_rcp_ps(n_2);
						__m128 q_3 = _mm_rcp_ps(n_3);

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

						_MM_TRANSPOSE4_PS(q_0, q_1, q_2, q_3);

						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDji, q_0);
						_mm_store_ps(DDji+N, q_1);
						_mm_store_ps(DDji+2*N, q_2);
						_mm_store_ps(DDji+3*N, q_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop

	__m128 cum_sum_sh1 = _mm_shuffle_ps(cum_sum, cum_sum, 177);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh1);
	__m128 cum_sum_sh2 = _mm_shuffle_ps(cum_sum, cum_sum, 78);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh2);

	float sum_q;
	_mm_store_ss(&sum_q, cum_sum);

	return sum_q;
}

inline float blocking_16_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 16; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;

	__m128 ones = _mm_set1_ps(1.);
	__m128 zeros = _mm_set1_ps(0.);
	__m128 cum_sum = _mm_set1_ps(0.);

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

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

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

						// __m128 q_0 = _mm_div_ps(ones, n_0);
						// __m128 q_1 = _mm_div_ps(ones, n_1);
						// __m128 q_2 = _mm_div_ps(ones, n_2);
						// __m128 q_3 = _mm_div_ps(ones, n_3);
						__m128 q_0 = _mm_rcp_ps(n_0);
						__m128 q_1 = _mm_rcp_ps(n_1);
						__m128 q_2 = _mm_rcp_ps(n_2);
						__m128 q_3 = _mm_rcp_ps(n_3);

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

						_MM_TRANSPOSE4_PS(q_0, q_1, q_2, q_3);

						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDji, q_0);
						_mm_store_ps(DDji+N, q_1);
						_mm_store_ps(DDji+2*N, q_2);
						_mm_store_ps(DDji+3*N, q_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop

	__m128 cum_sum_sh1 = _mm_shuffle_ps(cum_sum, cum_sum, 177);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh1);
	__m128 cum_sum_sh2 = _mm_shuffle_ps(cum_sum, cum_sum, 78);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh2);

	float sum_q;
	_mm_store_ss(&sum_q, cum_sum);

	return sum_q;
}

inline float blocking_8_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 8; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;

	__m128 ones = _mm_set1_ps(1.);
	__m128 zeros = _mm_set1_ps(0.);
	__m128 cum_sum = _mm_set1_ps(0.);

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

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

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

						// __m128 q_0 = _mm_div_ps(ones, n_0);
						// __m128 q_1 = _mm_div_ps(ones, n_1);
						// __m128 q_2 = _mm_div_ps(ones, n_2);
						// __m128 q_3 = _mm_div_ps(ones, n_3);
						__m128 q_0 = _mm_rcp_ps(n_0);
						__m128 q_1 = _mm_rcp_ps(n_1);
						__m128 q_2 = _mm_rcp_ps(n_2);
						__m128 q_3 = _mm_rcp_ps(n_3);

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

						_MM_TRANSPOSE4_PS(q_0, q_1, q_2, q_3);

						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDji, q_0);
						_mm_store_ps(DDji+N, q_1);
						_mm_store_ps(DDji+2*N, q_2);
						_mm_store_ps(DDji+3*N, q_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop

	__m128 cum_sum_sh1 = _mm_shuffle_ps(cum_sum, cum_sum, 177);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh1);
	__m128 cum_sum_sh2 = _mm_shuffle_ps(cum_sum, cum_sum, 78);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh2);

	float sum_q;
	_mm_store_ss(&sum_q, cum_sum);

	return sum_q;
}

inline float blocking_4_block_4_unfold_sr_vec(float* X, int N, int D, float* DD) {
	// Block size
	const int Bd = 4; // Desired Block size
	const int B =  (N >= Bd) ? Bd : N;
	const int K =  (N > 4) ? 4: N;

	__m128 ones = _mm_set1_ps(1.);
	__m128 zeros = _mm_set1_ps(0.);
	__m128 cum_sum = _mm_set1_ps(0.);

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

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

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

						// __m128 q_0 = _mm_div_ps(ones, n_0);
						// __m128 q_1 = _mm_div_ps(ones, n_1);
						// __m128 q_2 = _mm_div_ps(ones, n_2);
						// __m128 q_3 = _mm_div_ps(ones, n_3);
						__m128 q_0 = _mm_rcp_ps(n_0);
						__m128 q_1 = _mm_rcp_ps(n_1);
						__m128 q_2 = _mm_rcp_ps(n_2);
						__m128 q_3 = _mm_rcp_ps(n_3);

						__m128 sum1 = _mm_add_ps(q_0, q_1);
						__m128 sum2 = _mm_add_ps(q_2, q_3);
						__m128 sum3 = _mm_add_ps(sum1, sum2);
						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDij, q_0);
						_mm_store_ps(DDij+N, q_1);
						_mm_store_ps(DDij+2*N, q_2);
						_mm_store_ps(DDij+3*N, q_3);

						_MM_TRANSPOSE4_PS(q_0, q_1, q_2, q_3);

						cum_sum = _mm_add_ps(cum_sum, sum3);

						_mm_store_ps(DDji, q_0);
						_mm_store_ps(DDji+N, q_1);
						_mm_store_ps(DDji+2*N, q_2);
						_mm_store_ps(DDji+3*N, q_3);

					}
				}
			} // End of K loop

		}
	} // End of B loop

	__m128 cum_sum_sh1 = _mm_shuffle_ps(cum_sum, cum_sum, 177);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh1);
	__m128 cum_sum_sh2 = _mm_shuffle_ps(cum_sum, cum_sum, 78);
	cum_sum = _mm_add_ps(cum_sum, cum_sum_sh2);

	float sum_q;
	_mm_store_ss(&sum_q, cum_sum);

	return sum_q;
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
