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

inline float blocking2_avx(float* Y, int N, int D, float* Q) {
    const int K = 8; // Desired block size
    const int B = 32;

    __m256 ones = _mm256_set1_ps(1.);
	__m256 zeros = _mm256_setzero_ps();
	__m256 cum_sum = _mm256_setzero_ps();

    int iB;
    for (iB = 0; iB + B <= N; iB += B) {
        int jB;
        for (jB = iB; jB + B <= N; jB += B) {

            int iK;
            for (iK = iB; iK < iB + B; iK += K) {

                int nD = iK * D;
                __m256 Yn_0_d0 = _mm256_broadcast_ss(Y + nD + 0);
                __m256 Yn_0_d1 = _mm256_broadcast_ss(Y + nD + 1);
                __m256 Yn_1_d0 = _mm256_broadcast_ss(Y + nD + 2);
                __m256 Yn_1_d1 = _mm256_broadcast_ss(Y + nD + 3);
                __m256 Yn_2_d0 = _mm256_broadcast_ss(Y + nD + 4);
                __m256 Yn_2_d1 = _mm256_broadcast_ss(Y + nD + 5);
                __m256 Yn_3_d0 = _mm256_broadcast_ss(Y + nD + 6);
                __m256 Yn_3_d1 = _mm256_broadcast_ss(Y + nD + 7);
                __m256 Yn_4_d0 = _mm256_broadcast_ss(Y + nD + 8);
                __m256 Yn_4_d1 = _mm256_broadcast_ss(Y + nD + 9);
                __m256 Yn_5_d0 = _mm256_broadcast_ss(Y + nD + 10);
                __m256 Yn_5_d1 = _mm256_broadcast_ss(Y + nD + 11);
                __m256 Yn_6_d0 = _mm256_broadcast_ss(Y + nD + 12);
                __m256 Yn_6_d1 = _mm256_broadcast_ss(Y + nD + 13);
                __m256 Yn_7_d0 = _mm256_broadcast_ss(Y + nD + 14);
                __m256 Yn_7_d1 = _mm256_broadcast_ss(Y + nD + 15);

                int jK;
                // Compute the block
                for (jK = jB; jK < jB + B; jK += K) {
                    // In case the result Q and symetric are the same
                    if (jK < iK) { continue; }

                    int mD = jK * D;
                    int nN = iK * N + jK;
                    int mN = jK * N + iK;

                    __m256 Ym_0 = _mm256_load_ps(Y+mD);
                    __m256 Ym_1 = _mm256_load_ps(Y+mD+8);

                    __m256 Ym_lo = _mm256_shuffle_ps(Ym_0, Ym_1, 136);
                    __m256 Ym_hi = _mm256_shuffle_ps(Ym_0, Ym_1, 221);

                    __m256 Ym_lo_p = _mm256_permute2f128_ps(Ym_lo, Ym_lo, 1);
                    __m256 Ym_hi_p = _mm256_permute2f128_ps(Ym_hi, Ym_hi, 1);

                    __m256 Ym_lo_pp = _mm256_shuffle_ps(Ym_lo_p, Ym_lo_p, 78);
                    __m256 Ym_hi_pp = _mm256_shuffle_ps(Ym_hi_p, Ym_hi_p, 78);


                    __m256 Ym_d0 = _mm256_blend_ps(Ym_lo, Ym_lo_pp, 0b00111100);
                    __m256 Ym_d1 = _mm256_blend_ps(Ym_hi, Ym_hi_pp, 0b00111100);

                    __m256 diff_0_d0 = _mm256_sub_ps(Ym_d0, Yn_0_d0);
                    __m256 diff_1_d0 = _mm256_sub_ps(Ym_d0, Yn_1_d0);
                    __m256 diff_2_d0 = _mm256_sub_ps(Ym_d0, Yn_2_d0);
                    __m256 diff_3_d0 = _mm256_sub_ps(Ym_d0, Yn_3_d0);
                    __m256 diff_4_d0 = _mm256_sub_ps(Ym_d0, Yn_4_d0);
                    __m256 diff_5_d0 = _mm256_sub_ps(Ym_d0, Yn_5_d0);
                    __m256 diff_6_d0 = _mm256_sub_ps(Ym_d0, Yn_6_d0);
                    __m256 diff_7_d0 = _mm256_sub_ps(Ym_d0, Yn_7_d0);

                    __m256 diff_0_d1 = _mm256_sub_ps(Ym_d1, Yn_0_d1);
                    __m256 diff_1_d1 = _mm256_sub_ps(Ym_d1, Yn_1_d1);
                    __m256 diff_2_d1 = _mm256_sub_ps(Ym_d1, Yn_2_d1);
                    __m256 diff_3_d1 = _mm256_sub_ps(Ym_d1, Yn_3_d1);
                    __m256 diff_4_d1 = _mm256_sub_ps(Ym_d1, Yn_4_d1);
                    __m256 diff_5_d1 = _mm256_sub_ps(Ym_d1, Yn_5_d1);
                    __m256 diff_6_d1 = _mm256_sub_ps(Ym_d1, Yn_6_d1);
                    __m256 diff_7_d1 = _mm256_sub_ps(Ym_d1, Yn_7_d1);

                    __m256 df_sq_0_d0 = _mm256_mul_ps(diff_0_d0, diff_0_d0);
                    __m256 df_sq_1_d0 = _mm256_mul_ps(diff_1_d0, diff_1_d0);
                    __m256 df_sq_2_d0 = _mm256_mul_ps(diff_2_d0, diff_2_d0);
                    __m256 df_sq_3_d0 = _mm256_mul_ps(diff_3_d0, diff_3_d0);
                    __m256 df_sq_4_d0 = _mm256_mul_ps(diff_4_d0, diff_4_d0);
                    __m256 df_sq_5_d0 = _mm256_mul_ps(diff_5_d0, diff_5_d0);
                    __m256 df_sq_6_d0 = _mm256_mul_ps(diff_6_d0, diff_6_d0);
                    __m256 df_sq_7_d0 = _mm256_mul_ps(diff_7_d0, diff_7_d0);

                    __m256 norm_0 = _mm256_fmadd_ps(diff_0_d1, diff_0_d1,
                                                    df_sq_0_d0);
                    __m256 norm_1 = _mm256_fmadd_ps(diff_1_d1, diff_1_d1,
                                                    df_sq_1_d0);
                    __m256 norm_2 = _mm256_fmadd_ps(diff_2_d1, diff_2_d1,
                                                    df_sq_2_d0);
                    __m256 norm_3 = _mm256_fmadd_ps(diff_3_d1, diff_3_d1,
                                                    df_sq_3_d0);
                    __m256 norm_4 = _mm256_fmadd_ps(diff_4_d1, diff_4_d1,
                                                    df_sq_4_d0);
                    __m256 norm_5 = _mm256_fmadd_ps(diff_5_d1, diff_5_d1,
                                                    df_sq_5_d0);
                    __m256 norm_6 = _mm256_fmadd_ps(diff_6_d1, diff_6_d1,
                                                    df_sq_6_d0);
                    __m256 norm_7 = _mm256_fmadd_ps(diff_7_d1, diff_7_d1,
                                                    df_sq_7_d0);

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

                    _mm256_store_ps(Q+nN, q_0);
                    _mm256_store_ps(Q+nN+N, q_1);
                    _mm256_store_ps(Q+nN+2*N, q_2);
                    _mm256_store_ps(Q+nN+3*N, q_3);
                    _mm256_store_ps(Q+nN+4*N, q_4);
                    _mm256_store_ps(Q+nN+5*N, q_5);
                    _mm256_store_ps(Q+nN+6*N, q_6);
                    _mm256_store_ps(Q+nN+7*N, q_7);


                    // Check if we are in the diagonal or not
                    if (jK > iK) {

                        transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                        cum_sum = _mm256_add_ps(cum_sum, sum6);

                        _mm256_store_ps(Q+mN, q_0);
                        _mm256_store_ps(Q+mN+N, q_1);
                        _mm256_store_ps(Q+mN+2*N, q_2);
                        _mm256_store_ps(Q+mN+3*N, q_3);
                        _mm256_store_ps(Q+mN+4*N, q_4);
                        _mm256_store_ps(Q+mN+5*N, q_5);
                        _mm256_store_ps(Q+mN+6*N, q_6);
                        _mm256_store_ps(Q+mN+7*N, q_7);

                    }
                }

            }
        }
        // For the blocks which width is lower than 32
        for (int iiB = iB; iiB < iB + B; iiB += K) {

            int nD = iiB * D;
            __m256 Yn_0_d0 = _mm256_broadcast_ss(Y + nD + 0);
            __m256 Yn_0_d1 = _mm256_broadcast_ss(Y + nD + 1);
            __m256 Yn_1_d0 = _mm256_broadcast_ss(Y + nD + 2);
            __m256 Yn_1_d1 = _mm256_broadcast_ss(Y + nD + 3);
            __m256 Yn_2_d0 = _mm256_broadcast_ss(Y + nD + 4);
            __m256 Yn_2_d1 = _mm256_broadcast_ss(Y + nD + 5);
            __m256 Yn_3_d0 = _mm256_broadcast_ss(Y + nD + 6);
            __m256 Yn_3_d1 = _mm256_broadcast_ss(Y + nD + 7);
            __m256 Yn_4_d0 = _mm256_broadcast_ss(Y + nD + 8);
            __m256 Yn_4_d1 = _mm256_broadcast_ss(Y + nD + 9);
            __m256 Yn_5_d0 = _mm256_broadcast_ss(Y + nD + 10);
            __m256 Yn_5_d1 = _mm256_broadcast_ss(Y + nD + 11);
            __m256 Yn_6_d0 = _mm256_broadcast_ss(Y + nD + 12);
            __m256 Yn_6_d1 = _mm256_broadcast_ss(Y + nD + 13);
            __m256 Yn_7_d0 = _mm256_broadcast_ss(Y + nD + 14);
            __m256 Yn_7_d1 = _mm256_broadcast_ss(Y + nD + 15);

            int jjB;
            for (jjB = jB; jjB + K <= N; jjB += K) {
                int mD = jjB * D;
                int nN = iiB * N + jjB;
                int mN = jjB * N + iiB;

                __m256 Ym_0 = _mm256_load_ps(Y+mD);
                __m256 Ym_1 = _mm256_load_ps(Y+mD+8);

                __m256 Ym_lo = _mm256_shuffle_ps(Ym_0, Ym_1, 136);
                __m256 Ym_hi = _mm256_shuffle_ps(Ym_0, Ym_1, 221);

                __m256 Ym_lo_p = _mm256_permute2f128_ps(Ym_lo, Ym_lo, 1);
                __m256 Ym_hi_p = _mm256_permute2f128_ps(Ym_hi, Ym_hi, 1);

                __m256 Ym_lo_pp = _mm256_shuffle_ps(Ym_lo_p, Ym_lo_p, 78);
                __m256 Ym_hi_pp = _mm256_shuffle_ps(Ym_hi_p, Ym_hi_p, 78);


                __m256 Ym_d0 = _mm256_blend_ps(Ym_lo, Ym_lo_pp, 0b00111100);
                __m256 Ym_d1 = _mm256_blend_ps(Ym_hi, Ym_hi_pp, 0b00111100);

                __m256 diff_0_d0 = _mm256_sub_ps(Ym_d0, Yn_0_d0);
                __m256 diff_1_d0 = _mm256_sub_ps(Ym_d0, Yn_1_d0);
                __m256 diff_2_d0 = _mm256_sub_ps(Ym_d0, Yn_2_d0);
                __m256 diff_3_d0 = _mm256_sub_ps(Ym_d0, Yn_3_d0);
                __m256 diff_4_d0 = _mm256_sub_ps(Ym_d0, Yn_4_d0);
                __m256 diff_5_d0 = _mm256_sub_ps(Ym_d0, Yn_5_d0);
                __m256 diff_6_d0 = _mm256_sub_ps(Ym_d0, Yn_6_d0);
                __m256 diff_7_d0 = _mm256_sub_ps(Ym_d0, Yn_7_d0);

                __m256 diff_0_d1 = _mm256_sub_ps(Ym_d1, Yn_0_d1);
                __m256 diff_1_d1 = _mm256_sub_ps(Ym_d1, Yn_1_d1);
                __m256 diff_2_d1 = _mm256_sub_ps(Ym_d1, Yn_2_d1);
                __m256 diff_3_d1 = _mm256_sub_ps(Ym_d1, Yn_3_d1);
                __m256 diff_4_d1 = _mm256_sub_ps(Ym_d1, Yn_4_d1);
                __m256 diff_5_d1 = _mm256_sub_ps(Ym_d1, Yn_5_d1);
                __m256 diff_6_d1 = _mm256_sub_ps(Ym_d1, Yn_6_d1);
                __m256 diff_7_d1 = _mm256_sub_ps(Ym_d1, Yn_7_d1);

                __m256 df_sq_0_d0 = _mm256_mul_ps(diff_0_d0, diff_0_d0);
                __m256 df_sq_1_d0 = _mm256_mul_ps(diff_1_d0, diff_1_d0);
                __m256 df_sq_2_d0 = _mm256_mul_ps(diff_2_d0, diff_2_d0);
                __m256 df_sq_3_d0 = _mm256_mul_ps(diff_3_d0, diff_3_d0);
                __m256 df_sq_4_d0 = _mm256_mul_ps(diff_4_d0, diff_4_d0);
                __m256 df_sq_5_d0 = _mm256_mul_ps(diff_5_d0, diff_5_d0);
                __m256 df_sq_6_d0 = _mm256_mul_ps(diff_6_d0, diff_6_d0);
                __m256 df_sq_7_d0 = _mm256_mul_ps(diff_7_d0, diff_7_d0);

                __m256 norm_0 = _mm256_fmadd_ps(diff_0_d1, diff_0_d1,
                                                df_sq_0_d0);
                __m256 norm_1 = _mm256_fmadd_ps(diff_1_d1, diff_1_d1,
                                                df_sq_1_d0);
                __m256 norm_2 = _mm256_fmadd_ps(diff_2_d1, diff_2_d1,
                                                df_sq_2_d0);
                __m256 norm_3 = _mm256_fmadd_ps(diff_3_d1, diff_3_d1,
                                                df_sq_3_d0);
                __m256 norm_4 = _mm256_fmadd_ps(diff_4_d1, diff_4_d1,
                                                df_sq_4_d0);
                __m256 norm_5 = _mm256_fmadd_ps(diff_5_d1, diff_5_d1,
                                                df_sq_5_d0);
                __m256 norm_6 = _mm256_fmadd_ps(diff_6_d1, diff_6_d1,
                                                df_sq_6_d0);
                __m256 norm_7 = _mm256_fmadd_ps(diff_7_d1, diff_7_d1,
                                                df_sq_7_d0);

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

                __m256 sum0 = _mm256_add_ps(q_0, q_1);
                __m256 sum1 = _mm256_add_ps(q_2, q_3);
                __m256 sum2 = _mm256_add_ps(q_4, q_5);
                __m256 sum3 = _mm256_add_ps(q_6, q_7);
                __m256 sum4 = _mm256_add_ps(sum0, sum1);
                __m256 sum5 = _mm256_add_ps(sum2, sum3);
                __m256 sum6 = _mm256_add_ps(sum4, sum5);

                cum_sum = _mm256_add_ps(cum_sum, sum6);

                _mm256_store_ps(Q+nN, q_0);
                _mm256_store_ps(Q+nN+N, q_1);
                _mm256_store_ps(Q+nN+2*N, q_2);
                _mm256_store_ps(Q+nN+3*N, q_3);
                _mm256_store_ps(Q+nN+4*N, q_4);
                _mm256_store_ps(Q+nN+5*N, q_5);
                _mm256_store_ps(Q+nN+6*N, q_6);
                _mm256_store_ps(Q+nN+7*N, q_7);

                transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                cum_sum = _mm256_add_ps(cum_sum, sum6);

                _mm256_store_ps(Q+mN, q_0);
                _mm256_store_ps(Q+mN+N, q_1);
                _mm256_store_ps(Q+mN+2*N, q_2);
                _mm256_store_ps(Q+mN+3*N, q_3);
                _mm256_store_ps(Q+mN+4*N, q_4);
                _mm256_store_ps(Q+mN+5*N, q_5);
                _mm256_store_ps(Q+mN+6*N, q_6);
                _mm256_store_ps(Q+mN+7*N, q_7);


            }

            // Compute the remaining
            for (; jjB < N; jjB++) {
                int mD = jjB * D;
                int nN = iiB * N + jjB;
                int mN = jjB * N + iiB;

                float elem_0 = 0;
                float elem_1 = 0;
                float elem_2 = 0;
                float elem_3 = 0;
                float elem_4 = 0;
                float elem_5 = 0;
                float elem_6 = 0;
                float elem_7 = 0;

                float Ym_d0 = Y[mD];
                float Ym_d1 = Y[mD+1];

                float Ynm_0_d0 = Yn_0_d0[0] - Ym_d0;
                float Ynm_0_d1 = Yn_0_d1[0] - Ym_d1;
                float Ynm_1_d0 = Yn_1_d0[0] - Ym_d0;
                float Ynm_1_d1 = Yn_1_d1[0] - Ym_d1;
                float Ynm_2_d0 = Yn_2_d0[0] - Ym_d0;
                float Ynm_2_d1 = Yn_2_d1[0] - Ym_d1;
                float Ynm_3_d0 = Yn_3_d0[0] - Ym_d0;
                float Ynm_3_d1 = Yn_3_d1[0] - Ym_d1;
                float Ynm_4_d0 = Yn_4_d0[0] - Ym_d0;
                float Ynm_4_d1 = Yn_4_d1[0] - Ym_d1;
                float Ynm_5_d0 = Yn_5_d0[0] - Ym_d0;
                float Ynm_5_d1 = Yn_5_d1[0] - Ym_d1;
                float Ynm_6_d0 = Yn_6_d0[0] - Ym_d0;
                float Ynm_6_d1 = Yn_6_d1[0] - Ym_d1;
                float Ynm_7_d0 = Yn_7_d0[0] - Ym_d0;
                float Ynm_7_d1 = Yn_7_d1[0] - Ym_d1;

                elem_0 += Ynm_0_d0 * Ynm_0_d0;
                elem_1 += Ynm_1_d0 * Ynm_1_d0;
                elem_2 += Ynm_2_d0 * Ynm_2_d0;
                elem_3 += Ynm_3_d0 * Ynm_3_d0;
                elem_4 += Ynm_4_d0 * Ynm_4_d0;
                elem_5 += Ynm_5_d0 * Ynm_5_d0;
                elem_6 += Ynm_6_d0 * Ynm_6_d0;
                elem_7 += Ynm_7_d0 * Ynm_7_d0;

                elem_0 += Ynm_0_d1 * Ynm_0_d1;
                elem_1 += Ynm_1_d1 * Ynm_1_d1;
                elem_2 += Ynm_2_d1 * Ynm_2_d1;
                elem_3 += Ynm_3_d1 * Ynm_3_d1;
                elem_4 += Ynm_4_d1 * Ynm_4_d1;
                elem_5 += Ynm_5_d1 * Ynm_5_d1;
                elem_6 += Ynm_6_d1 * Ynm_6_d1;
                elem_7 += Ynm_7_d1 * Ynm_7_d1;

                elem_0 += 1;
                elem_1 += 1;
                elem_2 += 1;
                elem_3 += 1;
                elem_4 += 1;
                elem_5 += 1;
                elem_6 += 1;
                elem_7 += 1;

                float elem_inv_0 = 1 / elem_0;
                float elem_inv_1 = 1 / elem_1;
                float elem_inv_2 = 1 / elem_2;
                float elem_inv_3 = 1 / elem_3;
                float elem_inv_4 = 1 / elem_4;
                float elem_inv_5 = 1 / elem_5;
                float elem_inv_6 = 1 / elem_6;
                float elem_inv_7 = 1 / elem_7;

                float sum_0 = elem_inv_0 + elem_inv_1;
                float sum_1 = elem_inv_2 + elem_inv_2;
                float sum_2 = elem_inv_4 + elem_inv_5;
                float sum_3 = elem_inv_6 + elem_inv_7;
                float sum_4 = sum_0 + sum_1;
                float sum_5 = sum_2 + sum_3;
                float sum_6 = sum_4 + sum_5;
                cum_sum[0] += 2 * sum_6;

                Q[nN] = elem_inv_0;
                Q[nN+N] = elem_inv_1;
                Q[nN+2*N] = elem_inv_2;
                Q[nN+3*N] = elem_inv_3;
                Q[nN+4*N] = elem_inv_4;
                Q[nN+5*N] = elem_inv_5;
                Q[nN+6*N] = elem_inv_6;
                Q[nN+7*N] = elem_inv_7;

                Q[mN] = elem_inv_0;
                Q[mN+1] = elem_inv_1;
                Q[mN+2] = elem_inv_2;
                Q[mN+3] = elem_inv_3;
                Q[mN+4] = elem_inv_4;
                Q[mN+5] = elem_inv_5;
                Q[mN+6] = elem_inv_6;
                Q[mN+7] = elem_inv_7;
            }
        }
    }
    // Remaining positions in blocks of size 8
    for (; iB + K <= N; iB += K) {

        int nD = iB * D;
        __m256 Yn_0_d0 = _mm256_broadcast_ss(Y + nD + 0);
        __m256 Yn_0_d1 = _mm256_broadcast_ss(Y + nD + 1);
        __m256 Yn_1_d0 = _mm256_broadcast_ss(Y + nD + 2);
        __m256 Yn_1_d1 = _mm256_broadcast_ss(Y + nD + 3);
        __m256 Yn_2_d0 = _mm256_broadcast_ss(Y + nD + 4);
        __m256 Yn_2_d1 = _mm256_broadcast_ss(Y + nD + 5);
        __m256 Yn_3_d0 = _mm256_broadcast_ss(Y + nD + 6);
        __m256 Yn_3_d1 = _mm256_broadcast_ss(Y + nD + 7);
        __m256 Yn_4_d0 = _mm256_broadcast_ss(Y + nD + 8);
        __m256 Yn_4_d1 = _mm256_broadcast_ss(Y + nD + 9);
        __m256 Yn_5_d0 = _mm256_broadcast_ss(Y + nD + 10);
        __m256 Yn_5_d1 = _mm256_broadcast_ss(Y + nD + 11);
        __m256 Yn_6_d0 = _mm256_broadcast_ss(Y + nD + 12);
        __m256 Yn_6_d1 = _mm256_broadcast_ss(Y + nD + 13);
        __m256 Yn_7_d0 = _mm256_broadcast_ss(Y + nD + 14);
        __m256 Yn_7_d1 = _mm256_broadcast_ss(Y + nD + 15);

        int jB;
        // Compute the block
        for (jB = iB; jB + K <= N; jB += K) {

            int mD = jB * D;
            int nN = iB * N + jB;
            int mN = jB * N + iB;

            __m256 Ym_0 = _mm256_load_ps(Y+mD);
            __m256 Ym_1 = _mm256_load_ps(Y+mD+8);

            __m256 Ym_lo = _mm256_shuffle_ps(Ym_0, Ym_1, 136);
            __m256 Ym_hi = _mm256_shuffle_ps(Ym_0, Ym_1, 221);

            __m256 Ym_lo_p = _mm256_permute2f128_ps(Ym_lo, Ym_lo, 1);
            __m256 Ym_hi_p = _mm256_permute2f128_ps(Ym_hi, Ym_hi, 1);

            __m256 Ym_lo_pp = _mm256_shuffle_ps(Ym_lo_p, Ym_lo_p, 78);
            __m256 Ym_hi_pp = _mm256_shuffle_ps(Ym_hi_p, Ym_hi_p, 78);


            __m256 Ym_d0 = _mm256_blend_ps(Ym_lo, Ym_lo_pp, 0b00111100);
            __m256 Ym_d1 = _mm256_blend_ps(Ym_hi, Ym_hi_pp, 0b00111100);

            __m256 diff_0_d0 = _mm256_sub_ps(Ym_d0, Yn_0_d0);
            __m256 diff_1_d0 = _mm256_sub_ps(Ym_d0, Yn_1_d0);
            __m256 diff_2_d0 = _mm256_sub_ps(Ym_d0, Yn_2_d0);
            __m256 diff_3_d0 = _mm256_sub_ps(Ym_d0, Yn_3_d0);
            __m256 diff_4_d0 = _mm256_sub_ps(Ym_d0, Yn_4_d0);
            __m256 diff_5_d0 = _mm256_sub_ps(Ym_d0, Yn_5_d0);
            __m256 diff_6_d0 = _mm256_sub_ps(Ym_d0, Yn_6_d0);
            __m256 diff_7_d0 = _mm256_sub_ps(Ym_d0, Yn_7_d0);

            __m256 diff_0_d1 = _mm256_sub_ps(Ym_d1, Yn_0_d1);
            __m256 diff_1_d1 = _mm256_sub_ps(Ym_d1, Yn_1_d1);
            __m256 diff_2_d1 = _mm256_sub_ps(Ym_d1, Yn_2_d1);
            __m256 diff_3_d1 = _mm256_sub_ps(Ym_d1, Yn_3_d1);
            __m256 diff_4_d1 = _mm256_sub_ps(Ym_d1, Yn_4_d1);
            __m256 diff_5_d1 = _mm256_sub_ps(Ym_d1, Yn_5_d1);
            __m256 diff_6_d1 = _mm256_sub_ps(Ym_d1, Yn_6_d1);
            __m256 diff_7_d1 = _mm256_sub_ps(Ym_d1, Yn_7_d1);

            __m256 df_sq_0_d0 = _mm256_mul_ps(diff_0_d0, diff_0_d0);
            __m256 df_sq_1_d0 = _mm256_mul_ps(diff_1_d0, diff_1_d0);
            __m256 df_sq_2_d0 = _mm256_mul_ps(diff_2_d0, diff_2_d0);
            __m256 df_sq_3_d0 = _mm256_mul_ps(diff_3_d0, diff_3_d0);
            __m256 df_sq_4_d0 = _mm256_mul_ps(diff_4_d0, diff_4_d0);
            __m256 df_sq_5_d0 = _mm256_mul_ps(diff_5_d0, diff_5_d0);
            __m256 df_sq_6_d0 = _mm256_mul_ps(diff_6_d0, diff_6_d0);
            __m256 df_sq_7_d0 = _mm256_mul_ps(diff_7_d0, diff_7_d0);

            __m256 norm_0 = _mm256_fmadd_ps(diff_0_d1, diff_0_d1,
                                            df_sq_0_d0);
            __m256 norm_1 = _mm256_fmadd_ps(diff_1_d1, diff_1_d1,
                                            df_sq_1_d0);
            __m256 norm_2 = _mm256_fmadd_ps(diff_2_d1, diff_2_d1,
                                            df_sq_2_d0);
            __m256 norm_3 = _mm256_fmadd_ps(diff_3_d1, diff_3_d1,
                                            df_sq_3_d0);
            __m256 norm_4 = _mm256_fmadd_ps(diff_4_d1, diff_4_d1,
                                            df_sq_4_d0);
            __m256 norm_5 = _mm256_fmadd_ps(diff_5_d1, diff_5_d1,
                                            df_sq_5_d0);
            __m256 norm_6 = _mm256_fmadd_ps(diff_6_d1, diff_6_d1,
                                            df_sq_6_d0);
            __m256 norm_7 = _mm256_fmadd_ps(diff_7_d1, diff_7_d1,
                                            df_sq_7_d0);

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

            if (iB == jB) {
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

            _mm256_store_ps(Q+nN, q_0);
            _mm256_store_ps(Q+nN+N, q_1);
            _mm256_store_ps(Q+nN+2*N, q_2);
            _mm256_store_ps(Q+nN+3*N, q_3);
            _mm256_store_ps(Q+nN+4*N, q_4);
            _mm256_store_ps(Q+nN+5*N, q_5);
            _mm256_store_ps(Q+nN+6*N, q_6);
            _mm256_store_ps(Q+nN+7*N, q_7);


            // Check if we are in the diagonal or not
            if (jB > iB) {

                transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                cum_sum = _mm256_add_ps(cum_sum, sum6);

                _mm256_store_ps(Q+mN, q_0);
                _mm256_store_ps(Q+mN+N, q_1);
                _mm256_store_ps(Q+mN+2*N, q_2);
                _mm256_store_ps(Q+mN+3*N, q_3);
                _mm256_store_ps(Q+mN+4*N, q_4);
                _mm256_store_ps(Q+mN+5*N, q_5);
                _mm256_store_ps(Q+mN+6*N, q_6);
                _mm256_store_ps(Q+mN+7*N, q_7);

            }
        }
        // Compute the remaining
        for (; jB < N; jB++) {
            int mD = jB * D;
            int nN = iB * N + jB;
            int mN = jB * N + iB;

            float elem_0 = 0;
            float elem_1 = 0;
            float elem_2 = 0;
            float elem_3 = 0;
            float elem_4 = 0;
            float elem_5 = 0;
            float elem_6 = 0;
            float elem_7 = 0;

            float Ym_d0 = Y[mD];
            float Ym_d1 = Y[mD+1];

            float Ynm_0_d0 = Yn_0_d0[0] - Ym_d0;
            float Ynm_0_d1 = Yn_0_d1[0] - Ym_d1;
            float Ynm_1_d0 = Yn_1_d0[0] - Ym_d0;
            float Ynm_1_d1 = Yn_1_d1[0] - Ym_d1;
            float Ynm_2_d0 = Yn_2_d0[0] - Ym_d0;
            float Ynm_2_d1 = Yn_2_d1[0] - Ym_d1;
            float Ynm_3_d0 = Yn_3_d0[0] - Ym_d0;
            float Ynm_3_d1 = Yn_3_d1[0] - Ym_d1;
            float Ynm_4_d0 = Yn_4_d0[0] - Ym_d0;
            float Ynm_4_d1 = Yn_4_d1[0] - Ym_d1;
            float Ynm_5_d0 = Yn_5_d0[0] - Ym_d0;
            float Ynm_5_d1 = Yn_5_d1[0] - Ym_d1;
            float Ynm_6_d0 = Yn_6_d0[0] - Ym_d0;
            float Ynm_6_d1 = Yn_6_d1[0] - Ym_d1;
            float Ynm_7_d0 = Yn_7_d0[0] - Ym_d0;
            float Ynm_7_d1 = Yn_7_d1[0] - Ym_d1;

            elem_0 += Ynm_0_d0 * Ynm_0_d0;
            elem_1 += Ynm_1_d0 * Ynm_1_d0;
            elem_2 += Ynm_2_d0 * Ynm_2_d0;
            elem_3 += Ynm_3_d0 * Ynm_3_d0;
            elem_4 += Ynm_4_d0 * Ynm_4_d0;
            elem_5 += Ynm_5_d0 * Ynm_5_d0;
            elem_6 += Ynm_6_d0 * Ynm_6_d0;
            elem_7 += Ynm_7_d0 * Ynm_7_d0;

            elem_0 += Ynm_0_d1 * Ynm_0_d1;
            elem_1 += Ynm_1_d1 * Ynm_1_d1;
            elem_2 += Ynm_2_d1 * Ynm_2_d1;
            elem_3 += Ynm_3_d1 * Ynm_3_d1;
            elem_4 += Ynm_4_d1 * Ynm_4_d1;
            elem_5 += Ynm_5_d1 * Ynm_5_d1;
            elem_6 += Ynm_6_d1 * Ynm_6_d1;
            elem_7 += Ynm_7_d1 * Ynm_7_d1;

            elem_0 += 1;
            elem_1 += 1;
            elem_2 += 1;
            elem_3 += 1;
            elem_4 += 1;
            elem_5 += 1;
            elem_6 += 1;
            elem_7 += 1;

            float elem_inv_0 = 1 / elem_0;
            float elem_inv_1 = 1 / elem_1;
            float elem_inv_2 = 1 / elem_2;
            float elem_inv_3 = 1 / elem_3;
            float elem_inv_4 = 1 / elem_4;
            float elem_inv_5 = 1 / elem_5;
            float elem_inv_6 = 1 / elem_6;
            float elem_inv_7 = 1 / elem_7;

            float sum_0 = elem_inv_0 + elem_inv_1;
            float sum_1 = elem_inv_2 + elem_inv_2;
            float sum_2 = elem_inv_4 + elem_inv_5;
            float sum_3 = elem_inv_6 + elem_inv_7;
            float sum_4 = sum_0 + sum_1;
            float sum_5 = sum_2 + sum_3;
            float sum_6 = sum_4 + sum_5;
            cum_sum[0] += 2 * sum_6;

            Q[nN] = elem_inv_0;
            Q[nN+N] = elem_inv_1;
            Q[nN+2*N] = elem_inv_2;
            Q[nN+3*N] = elem_inv_3;
            Q[nN+4*N] = elem_inv_4;
            Q[nN+5*N] = elem_inv_5;
            Q[nN+6*N] = elem_inv_6;
            Q[nN+7*N] = elem_inv_7;

            Q[mN] = elem_inv_0;
            Q[mN+1] = elem_inv_1;
            Q[mN+2] = elem_inv_2;
            Q[mN+3] = elem_inv_3;
            Q[mN+4] = elem_inv_4;
            Q[mN+5] = elem_inv_5;
            Q[mN+6] = elem_inv_6;
            Q[mN+7] = elem_inv_7;
        }

    }
    // Remaining positions
    for (; iB < N; iB++) {
        int nD = iB*D;
        int mD = nD + D;
        int nN = iB*N + iB;
        int mN = nN + N;

        Q[nN++] = 0;

        float Yn_0 = Y[nD];
        float Yn_1 = Y[nD+1];

        for (int jB = iB + 1; jB < N; jB++, mD += D, nN++,
                mN += N) {

            float elem = 0;
            float Ym_0 = Y[mD];
            float Ym_1 = Y[mD+1];
            float Ynm_0 = Yn_0 - Ym_0;
            float Ynm_1 = Yn_1 - Ym_1;
            elem += Ynm_0 * Ynm_0;
            elem += Ynm_1 * Ynm_1;
            elem += 1;
            float elem_inv = 1 / elem;
            Q[nN] = elem_inv;
            Q[mN] = elem_inv;
            cum_sum[0] += 2 * elem_inv;
        }
    }

    float sum_Q = _mm256_reduce_add_ps(cum_sum);


    return sum_Q;
}

inline float blocking_avx(float* Y, int N, int D, float* Q) {
    const int K = 8; // Desired block size

    __m256 ones = _mm256_set1_ps(1.);
	__m256 zeros = _mm256_setzero_ps();
	__m256 cum_sum = _mm256_setzero_ps();

    int iK;
    for (iK = 0; iK + K <= N; iK += K) {

        int nD = iK * D;
        __m256 Yn_0_d0 = _mm256_broadcast_ss(Y + nD + 0);
        __m256 Yn_0_d1 = _mm256_broadcast_ss(Y + nD + 1);
        __m256 Yn_1_d0 = _mm256_broadcast_ss(Y + nD + 2);
        __m256 Yn_1_d1 = _mm256_broadcast_ss(Y + nD + 3);
        __m256 Yn_2_d0 = _mm256_broadcast_ss(Y + nD + 4);
        __m256 Yn_2_d1 = _mm256_broadcast_ss(Y + nD + 5);
        __m256 Yn_3_d0 = _mm256_broadcast_ss(Y + nD + 6);
        __m256 Yn_3_d1 = _mm256_broadcast_ss(Y + nD + 7);
        __m256 Yn_4_d0 = _mm256_broadcast_ss(Y + nD + 8);
        __m256 Yn_4_d1 = _mm256_broadcast_ss(Y + nD + 9);
        __m256 Yn_5_d0 = _mm256_broadcast_ss(Y + nD + 10);
        __m256 Yn_5_d1 = _mm256_broadcast_ss(Y + nD + 11);
        __m256 Yn_6_d0 = _mm256_broadcast_ss(Y + nD + 12);
        __m256 Yn_6_d1 = _mm256_broadcast_ss(Y + nD + 13);
        __m256 Yn_7_d0 = _mm256_broadcast_ss(Y + nD + 14);
        __m256 Yn_7_d1 = _mm256_broadcast_ss(Y + nD + 15);

        int jK;
        // Compute the block
        for (jK = iK; jK + K <= N; jK += K) {

            int mD = jK * D;
            int nN = iK * N + jK;
            int mN = jK * N + iK;

            __m256 Ym_0 = _mm256_load_ps(Y+mD);
            __m256 Ym_1 = _mm256_load_ps(Y+mD+8);

            __m256 Ym_lo = _mm256_shuffle_ps(Ym_0, Ym_1, 136);
            __m256 Ym_hi = _mm256_shuffle_ps(Ym_0, Ym_1, 221);

            __m256 Ym_lo_p = _mm256_permute2f128_ps(Ym_lo, Ym_lo, 1);
            __m256 Ym_hi_p = _mm256_permute2f128_ps(Ym_hi, Ym_hi, 1);

            __m256 Ym_lo_pp = _mm256_shuffle_ps(Ym_lo_p, Ym_lo_p, 78);
            __m256 Ym_hi_pp = _mm256_shuffle_ps(Ym_hi_p, Ym_hi_p, 78);


            __m256 Ym_d0 = _mm256_blend_ps(Ym_lo, Ym_lo_pp, 0b00111100);
            __m256 Ym_d1 = _mm256_blend_ps(Ym_hi, Ym_hi_pp, 0b00111100);

            __m256 diff_0_d0 = _mm256_sub_ps(Ym_d0, Yn_0_d0);
            __m256 diff_1_d0 = _mm256_sub_ps(Ym_d0, Yn_1_d0);
            __m256 diff_2_d0 = _mm256_sub_ps(Ym_d0, Yn_2_d0);
            __m256 diff_3_d0 = _mm256_sub_ps(Ym_d0, Yn_3_d0);
            __m256 diff_4_d0 = _mm256_sub_ps(Ym_d0, Yn_4_d0);
            __m256 diff_5_d0 = _mm256_sub_ps(Ym_d0, Yn_5_d0);
            __m256 diff_6_d0 = _mm256_sub_ps(Ym_d0, Yn_6_d0);
            __m256 diff_7_d0 = _mm256_sub_ps(Ym_d0, Yn_7_d0);

            __m256 diff_0_d1 = _mm256_sub_ps(Ym_d1, Yn_0_d1);
            __m256 diff_1_d1 = _mm256_sub_ps(Ym_d1, Yn_1_d1);
            __m256 diff_2_d1 = _mm256_sub_ps(Ym_d1, Yn_2_d1);
            __m256 diff_3_d1 = _mm256_sub_ps(Ym_d1, Yn_3_d1);
            __m256 diff_4_d1 = _mm256_sub_ps(Ym_d1, Yn_4_d1);
            __m256 diff_5_d1 = _mm256_sub_ps(Ym_d1, Yn_5_d1);
            __m256 diff_6_d1 = _mm256_sub_ps(Ym_d1, Yn_6_d1);
            __m256 diff_7_d1 = _mm256_sub_ps(Ym_d1, Yn_7_d1);

            __m256 df_sq_0_d0 = _mm256_mul_ps(diff_0_d0, diff_0_d0);
            __m256 df_sq_1_d0 = _mm256_mul_ps(diff_1_d0, diff_1_d0);
            __m256 df_sq_2_d0 = _mm256_mul_ps(diff_2_d0, diff_2_d0);
            __m256 df_sq_3_d0 = _mm256_mul_ps(diff_3_d0, diff_3_d0);
            __m256 df_sq_4_d0 = _mm256_mul_ps(diff_4_d0, diff_4_d0);
            __m256 df_sq_5_d0 = _mm256_mul_ps(diff_5_d0, diff_5_d0);
            __m256 df_sq_6_d0 = _mm256_mul_ps(diff_6_d0, diff_6_d0);
            __m256 df_sq_7_d0 = _mm256_mul_ps(diff_7_d0, diff_7_d0);

            __m256 norm_0 = _mm256_fmadd_ps(diff_0_d1, diff_0_d1,
                                            df_sq_0_d0);
            __m256 norm_1 = _mm256_fmadd_ps(diff_1_d1, diff_1_d1,
                                            df_sq_1_d0);
            __m256 norm_2 = _mm256_fmadd_ps(diff_2_d1, diff_2_d1,
                                            df_sq_2_d0);
            __m256 norm_3 = _mm256_fmadd_ps(diff_3_d1, diff_3_d1,
                                            df_sq_3_d0);
            __m256 norm_4 = _mm256_fmadd_ps(diff_4_d1, diff_4_d1,
                                            df_sq_4_d0);
            __m256 norm_5 = _mm256_fmadd_ps(diff_5_d1, diff_5_d1,
                                            df_sq_5_d0);
            __m256 norm_6 = _mm256_fmadd_ps(diff_6_d1, diff_6_d1,
                                            df_sq_6_d0);
            __m256 norm_7 = _mm256_fmadd_ps(diff_7_d1, diff_7_d1,
                                            df_sq_7_d0);

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

            _mm256_store_ps(Q+nN, q_0);
            _mm256_store_ps(Q+nN+N, q_1);
            _mm256_store_ps(Q+nN+2*N, q_2);
            _mm256_store_ps(Q+nN+3*N, q_3);
            _mm256_store_ps(Q+nN+4*N, q_4);
            _mm256_store_ps(Q+nN+5*N, q_5);
            _mm256_store_ps(Q+nN+6*N, q_6);
            _mm256_store_ps(Q+nN+7*N, q_7);


            // Check if we are in the diagonal or not
            if (jK > iK) {

                transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                cum_sum = _mm256_add_ps(cum_sum, sum6);

                _mm256_store_ps(Q+mN, q_0);
                _mm256_store_ps(Q+mN+N, q_1);
                _mm256_store_ps(Q+mN+2*N, q_2);
                _mm256_store_ps(Q+mN+3*N, q_3);
                _mm256_store_ps(Q+mN+4*N, q_4);
                _mm256_store_ps(Q+mN+5*N, q_5);
                _mm256_store_ps(Q+mN+6*N, q_6);
                _mm256_store_ps(Q+mN+7*N, q_7);

            }
        }
        // Compute the remaining
        for (; jK < N; jK++) {
            int mD = jK * D;
            int nN = iK * N + jK;
            int mN = jK * N + iK;

            float elem_0 = 0;
            float elem_1 = 0;
            float elem_2 = 0;
            float elem_3 = 0;
            float elem_4 = 0;
            float elem_5 = 0;
            float elem_6 = 0;
            float elem_7 = 0;

            float Ym_d0 = Y[mD];
            float Ym_d1 = Y[mD+1];

            float Ynm_0_d0 = Yn_0_d0[0] - Ym_d0;
            float Ynm_0_d1 = Yn_0_d1[0] - Ym_d1;
            float Ynm_1_d0 = Yn_1_d0[0] - Ym_d0;
            float Ynm_1_d1 = Yn_1_d1[0] - Ym_d1;
            float Ynm_2_d0 = Yn_2_d0[0] - Ym_d0;
            float Ynm_2_d1 = Yn_2_d1[0] - Ym_d1;
            float Ynm_3_d0 = Yn_3_d0[0] - Ym_d0;
            float Ynm_3_d1 = Yn_3_d1[0] - Ym_d1;
            float Ynm_4_d0 = Yn_4_d0[0] - Ym_d0;
            float Ynm_4_d1 = Yn_4_d1[0] - Ym_d1;
            float Ynm_5_d0 = Yn_5_d0[0] - Ym_d0;
            float Ynm_5_d1 = Yn_5_d1[0] - Ym_d1;
            float Ynm_6_d0 = Yn_6_d0[0] - Ym_d0;
            float Ynm_6_d1 = Yn_6_d1[0] - Ym_d1;
            float Ynm_7_d0 = Yn_7_d0[0] - Ym_d0;
            float Ynm_7_d1 = Yn_7_d1[0] - Ym_d1;

            elem_0 += Ynm_0_d0 * Ynm_0_d0;
            elem_1 += Ynm_1_d0 * Ynm_1_d0;
            elem_2 += Ynm_2_d0 * Ynm_2_d0;
            elem_3 += Ynm_3_d0 * Ynm_3_d0;
            elem_4 += Ynm_4_d0 * Ynm_4_d0;
            elem_5 += Ynm_5_d0 * Ynm_5_d0;
            elem_6 += Ynm_6_d0 * Ynm_6_d0;
            elem_7 += Ynm_7_d0 * Ynm_7_d0;

            elem_0 += Ynm_0_d1 * Ynm_0_d1;
            elem_1 += Ynm_1_d1 * Ynm_1_d1;
            elem_2 += Ynm_2_d1 * Ynm_2_d1;
            elem_3 += Ynm_3_d1 * Ynm_3_d1;
            elem_4 += Ynm_4_d1 * Ynm_4_d1;
            elem_5 += Ynm_5_d1 * Ynm_5_d1;
            elem_6 += Ynm_6_d1 * Ynm_6_d1;
            elem_7 += Ynm_7_d1 * Ynm_7_d1;

            elem_0 += 1;
            elem_1 += 1;
            elem_2 += 1;
            elem_3 += 1;
            elem_4 += 1;
            elem_5 += 1;
            elem_6 += 1;
            elem_7 += 1;

            float elem_inv_0 = 1 / elem_0;
            float elem_inv_1 = 1 / elem_1;
            float elem_inv_2 = 1 / elem_2;
            float elem_inv_3 = 1 / elem_3;
            float elem_inv_4 = 1 / elem_4;
            float elem_inv_5 = 1 / elem_5;
            float elem_inv_6 = 1 / elem_6;
            float elem_inv_7 = 1 / elem_7;

            float sum_0 = elem_inv_0 + elem_inv_1;
            float sum_1 = elem_inv_2 + elem_inv_2;
            float sum_2 = elem_inv_4 + elem_inv_5;
            float sum_3 = elem_inv_6 + elem_inv_7;
            float sum_4 = sum_0 + sum_1;
            float sum_5 = sum_2 + sum_3;
            float sum_6 = sum_4 + sum_5;
            cum_sum[0] += 2 * sum_6;

            Q[nN] = elem_inv_0;
            Q[nN+N] = elem_inv_1;
            Q[nN+2*N] = elem_inv_2;
            Q[nN+3*N] = elem_inv_3;
            Q[nN+4*N] = elem_inv_4;
            Q[nN+5*N] = elem_inv_5;
            Q[nN+6*N] = elem_inv_6;
            Q[nN+7*N] = elem_inv_7;

            Q[mN] = elem_inv_0;
            Q[mN+1] = elem_inv_1;
            Q[mN+2] = elem_inv_2;
            Q[mN+3] = elem_inv_3;
            Q[mN+4] = elem_inv_4;
            Q[mN+5] = elem_inv_5;
            Q[mN+6] = elem_inv_6;
            Q[mN+7] = elem_inv_7;
        }

    }
    // Remaining positions
    for (; iK < N; iK++) {
        int nD = iK*D;
        int mD = nD + D;
        int nN = iK*N + iK;
        int mN = nN + N;

        Q[nN++] = 0;

        float Yn_0 = Y[nD];
        float Yn_1 = Y[nD+1];

        for (int jK = iK + 1; jK < N; jK++, mD += D, nN++, mN += N) {
            float elem = 0;
            float Ym_0 = Y[mD];
            float Ym_1 = Y[mD+1];
            float Ynm_0 = Yn_0 - Ym_0;
            float Ynm_1 = Yn_1 - Ym_1;
            elem += Ynm_0 * Ynm_0;
            elem += Ynm_1 * Ynm_1;
            elem += 1;
            float elem_inv = 1 / elem;
            Q[nN] = elem_inv;
            Q[mN] = elem_inv;
            cum_sum[0] += 2 * elem_inv;
        }
    }

    float sum_Q = _mm256_reduce_add_ps(cum_sum);


    return sum_Q;
}

inline float blocking(float* Y, int N, int D, float* Q) {
    float sum_Q = 0.;
    const int K = 4; // Desired block size

    int iK, nD;
    int nN = 0;
    for (iK = 0, nD = 0; iK + K <= N; iK += K, nD += K*D, nN += K*N) {

        float Yn_0_d0 = Y[nD];
        float Yn_0_d1 = Y[nD+1];
        float Yn_1_d0 = Y[nD+2];
        float Yn_1_d1 = Y[nD+3];
        float Yn_2_d0 = Y[nD+4];
        float Yn_2_d1 = Y[nD+5];
        float Yn_3_d0 = Y[nD+6];
        float Yn_3_d1 = Y[nD+7];

        int jK;
        int mD = nD;
        // Compute the block
        for (jK = iK; jK + K <= N; jK += K, mD += 4*D) {
            int nN = iK * N + jK;
            int mN = jK * N + iK;

            float Ym_0_d0 = Y[mD];
            float Ym_0_d1 = Y[mD+1];
            float Ym_1_d0 = Y[mD+2];
            float Ym_1_d1 = Y[mD+3];
            float Ym_2_d0 = Y[mD+4];
            float Ym_2_d1 = Y[mD+5];
            float Ym_3_d0 = Y[mD+6];
            float Ym_3_d1 = Y[mD+7];

            float elem_00 = 0, elem_01 = 0, elem_02 = 0, elem_03 = 0;
            float elem_10 = 0, elem_11 = 0, elem_12 = 0, elem_13 = 0;
            float elem_20 = 0, elem_21 = 0, elem_22 = 0, elem_23 = 0;
            float elem_30 = 0, elem_31 = 0, elem_32 = 0, elem_33 = 0;

            float Ynm_00_d0 = Yn_0_d0 - Ym_0_d0;
            float Ynm_01_d0 = Yn_0_d0 - Ym_1_d0;
            float Ynm_02_d0 = Yn_0_d0 - Ym_2_d0;
            float Ynm_03_d0 = Yn_0_d0 - Ym_3_d0;
            float Ynm_10_d0 = Yn_1_d0 - Ym_0_d0;
            float Ynm_11_d0 = Yn_1_d0 - Ym_1_d0;
            float Ynm_12_d0 = Yn_1_d0 - Ym_2_d0;
            float Ynm_13_d0 = Yn_1_d0 - Ym_3_d0;
            float Ynm_20_d0 = Yn_2_d0 - Ym_0_d0;
            float Ynm_21_d0 = Yn_2_d0 - Ym_1_d0;
            float Ynm_22_d0 = Yn_2_d0 - Ym_2_d0;
            float Ynm_23_d0 = Yn_2_d0 - Ym_3_d0;
            float Ynm_30_d0 = Yn_3_d0 - Ym_0_d0;
            float Ynm_31_d0 = Yn_3_d0 - Ym_1_d0;
            float Ynm_32_d0 = Yn_3_d0 - Ym_2_d0;
            float Ynm_33_d0 = Yn_3_d0 - Ym_3_d0;

            float Ynm_00_d1 = Yn_0_d1 - Ym_0_d1;
            float Ynm_01_d1 = Yn_0_d1 - Ym_1_d1;
            float Ynm_02_d1 = Yn_0_d1 - Ym_2_d1;
            float Ynm_03_d1 = Yn_0_d1 - Ym_3_d1;
            float Ynm_10_d1 = Yn_1_d1 - Ym_0_d1;
            float Ynm_11_d1 = Yn_1_d1 - Ym_1_d1;
            float Ynm_12_d1 = Yn_1_d1 - Ym_2_d1;
            float Ynm_13_d1 = Yn_1_d1 - Ym_3_d1;
            float Ynm_20_d1 = Yn_2_d1 - Ym_0_d1;
            float Ynm_21_d1 = Yn_2_d1 - Ym_1_d1;
            float Ynm_22_d1 = Yn_2_d1 - Ym_2_d1;
            float Ynm_23_d1 = Yn_2_d1 - Ym_3_d1;
            float Ynm_30_d1 = Yn_3_d1 - Ym_0_d1;
            float Ynm_31_d1 = Yn_3_d1 - Ym_1_d1;
            float Ynm_32_d1 = Yn_3_d1 - Ym_2_d1;
            float Ynm_33_d1 = Yn_3_d1 - Ym_3_d1;

            elem_00 += Ynm_00_d0 * Ynm_00_d0;
            elem_01 += Ynm_01_d0 * Ynm_01_d0;
            elem_02 += Ynm_02_d0 * Ynm_02_d0;
            elem_03 += Ynm_03_d0 * Ynm_03_d0;
            elem_10 += Ynm_10_d0 * Ynm_10_d0;
            elem_11 += Ynm_11_d0 * Ynm_11_d0;
            elem_12 += Ynm_12_d0 * Ynm_12_d0;
            elem_13 += Ynm_13_d0 * Ynm_13_d0;
            elem_20 += Ynm_20_d0 * Ynm_20_d0;
            elem_21 += Ynm_21_d0 * Ynm_21_d0;
            elem_22 += Ynm_22_d0 * Ynm_22_d0;
            elem_23 += Ynm_23_d0 * Ynm_23_d0;
            elem_30 += Ynm_30_d0 * Ynm_30_d0;
            elem_31 += Ynm_31_d0 * Ynm_31_d0;
            elem_32 += Ynm_32_d0 * Ynm_32_d0;
            elem_33 += Ynm_33_d0 * Ynm_33_d0;

            elem_00 += Ynm_00_d1 * Ynm_00_d1;
            elem_01 += Ynm_01_d1 * Ynm_01_d1;
            elem_02 += Ynm_02_d1 * Ynm_02_d1;
            elem_03 += Ynm_03_d1 * Ynm_03_d1;
            elem_10 += Ynm_10_d1 * Ynm_10_d1;
            elem_11 += Ynm_11_d1 * Ynm_11_d1;
            elem_12 += Ynm_12_d1 * Ynm_12_d1;
            elem_13 += Ynm_13_d1 * Ynm_13_d1;
            elem_20 += Ynm_20_d1 * Ynm_20_d1;
            elem_21 += Ynm_21_d1 * Ynm_21_d1;
            elem_22 += Ynm_22_d1 * Ynm_22_d1;
            elem_23 += Ynm_23_d1 * Ynm_23_d1;
            elem_30 += Ynm_30_d1 * Ynm_30_d1;
            elem_31 += Ynm_31_d1 * Ynm_31_d1;
            elem_32 += Ynm_32_d1 * Ynm_32_d1;
            elem_33 += Ynm_33_d1 * Ynm_33_d1;

            elem_00 += 1;
            elem_01 += 1;
            elem_02 += 1;
            elem_03 += 1;
            elem_10 += 1;
            elem_11 += 1;
            elem_12 += 1;
            elem_13 += 1;
            elem_20 += 1;
            elem_21 += 1;
            elem_22 += 1;
            elem_23 += 1;
            elem_30 += 1;
            elem_31 += 1;
            elem_32 += 1;
            elem_33 += 1;

            float elem_inv_00 = 1 / elem_00;
            float elem_inv_01 = 1 / elem_01;
            float elem_inv_02 = 1 / elem_02;
            float elem_inv_03 = 1 / elem_03;
            float elem_inv_10 = 1 / elem_10;
            float elem_inv_11 = 1 / elem_11;
            float elem_inv_12 = 1 / elem_12;
            float elem_inv_13 = 1 / elem_13;
            float elem_inv_20 = 1 / elem_20;
            float elem_inv_21 = 1 / elem_21;
            float elem_inv_22 = 1 / elem_22;
            float elem_inv_23 = 1 / elem_23;
            float elem_inv_30 = 1 / elem_30;
            float elem_inv_31 = 1 / elem_31;
            float elem_inv_32 = 1 / elem_32;
            float elem_inv_33 = 1 / elem_33;

            if (iK == jK) {
                elem_inv_00 = 0;
                elem_inv_11 = 0;
                elem_inv_22 = 0;
                elem_inv_33 = 0;
            }

            float sum_00 = elem_inv_00 + elem_inv_01;
            float sum_01 = elem_inv_02 + elem_inv_03;
            float sum_10 = elem_inv_10 + elem_inv_11;
            float sum_11 = elem_inv_12 + elem_inv_13;
            float sum_20 = elem_inv_20 + elem_inv_21;
            float sum_21 = elem_inv_22 + elem_inv_23;
            float sum_30 = elem_inv_30 + elem_inv_31;
            float sum_31 = elem_inv_32 + elem_inv_33;

            float sum_02 = sum_00 + sum_01;
            float sum_12 = sum_10 + sum_11;
            float sum_22 = sum_20 + sum_21;
            float sum_32 = sum_30 + sum_31;

            float sum_0 = sum_02 + sum_12;
            float sum_1 = sum_22 + sum_32;
            float sum = sum_0 + sum_1;

            sum_Q += sum;

            Q[nN] = elem_inv_00;
            Q[nN+1] = elem_inv_01;
            Q[nN+2] = elem_inv_02;
            Q[nN+3] = elem_inv_03;
            Q[nN+N] = elem_inv_10;
            Q[nN+N+1] = elem_inv_11;
            Q[nN+N+2] = elem_inv_12;
            Q[nN+N+3] = elem_inv_13;
            Q[nN+2*N] = elem_inv_20;
            Q[nN+2*N+1] = elem_inv_21;
            Q[nN+2*N+2] = elem_inv_22;
            Q[nN+2*N+3] = elem_inv_23;
            Q[nN+3*N] = elem_inv_30;
            Q[nN+3*N+1] = elem_inv_31;
            Q[nN+3*N+2] = elem_inv_32;
            Q[nN+3*N+3] = elem_inv_33;

            // Check if we are in the diagonal or not
            if (jK > iK) {
                sum_Q += sum;

                Q[mN] = elem_inv_00;
                Q[mN+1] = elem_inv_10;
                Q[mN+2] = elem_inv_20;
                Q[mN+3] = elem_inv_30;
                Q[mN+N] = elem_inv_01;
                Q[mN+N+1] = elem_inv_11;
                Q[mN+N+2] = elem_inv_21;
                Q[mN+N+3] = elem_inv_31;
                Q[mN+2*N] = elem_inv_02;
                Q[mN+2*N+1] = elem_inv_12;
                Q[mN+2*N+2] = elem_inv_22;
                Q[mN+2*N+3] = elem_inv_32;
                Q[mN+3*N] = elem_inv_03;
                Q[mN+3*N+1] = elem_inv_13;
                Q[mN+3*N+2] = elem_inv_23;
                Q[mN+3*N+3] = elem_inv_33;
            }
        }
        // Compute the remaining
        for (; jK < N; jK++) {
            int mD = jK * D;
            float elem_0 = 0;
            float elem_1 = 0;
            float elem_2 = 0;
            float elem_3 = 0;

            float Ym_d0 = Y[mD];
            float Ym_d1 = Y[mD+1];

            float Ynm_0_d0 = Yn_0_d0 - Ym_d0;
            float Ynm_0_d1 = Yn_0_d1 - Ym_d1;
            float Ynm_1_d0 = Yn_1_d0 - Ym_d0;
            float Ynm_1_d1 = Yn_1_d1 - Ym_d1;
            float Ynm_2_d0 = Yn_2_d0 - Ym_d0;
            float Ynm_2_d1 = Yn_2_d1 - Ym_d1;
            float Ynm_3_d0 = Yn_3_d0 - Ym_d0;
            float Ynm_3_d1 = Yn_3_d1 - Ym_d1;

            elem_0 += Ynm_0_d0 * Ynm_0_d0;
            elem_1 += Ynm_1_d0 * Ynm_1_d0;
            elem_2 += Ynm_2_d0 * Ynm_2_d0;
            elem_3 += Ynm_3_d0 * Ynm_3_d0;
            elem_0 += Ynm_0_d1 * Ynm_0_d1;
            elem_1 += Ynm_1_d1 * Ynm_1_d1;
            elem_2 += Ynm_2_d1 * Ynm_2_d1;
            elem_3 += Ynm_3_d1 * Ynm_3_d1;

            elem_0 += 1;
            elem_1 += 1;
            elem_2 += 1;
            elem_3 += 1;

            float elem_inv_0 = 1 / elem_0;
            float elem_inv_1 = 1 / elem_1;
            float elem_inv_2 = 1 / elem_2;
            float elem_inv_3 = 1 / elem_3;

            float sum_0 = elem_inv_0 + elem_inv_1;
            float sum_1 = elem_inv_2 + elem_inv_2;
            float sum_2 = sum_0 + sum_1;
            sum_Q += 2 * sum_2;

            int pos = iK * N + jK;
            int pos_sym = jK * N + iK;

            Q[pos] = elem_inv_0;
            Q[pos+N] = elem_inv_1;
            Q[pos+2*N] = elem_inv_2;
            Q[pos+3*N] = elem_inv_3;

            Q[pos_sym] = elem_inv_0;
            Q[pos_sym+1] = elem_inv_1;
            Q[pos_sym+2] = elem_inv_2;
            Q[pos_sym+3] = elem_inv_3;
        }

    }
    // Remaining positions
    for (; iK < N; iK++) {
        int nD = iK*D;
        int mD = nD + D;
        int pos = iK*N + iK;
        int pos_sym = pos + N;

        Q[pos++] = 0;

        float Yn_0 = Y[nD];
        float Yn_1 = Y[nD+1];

        for (int jK = iK + 1; jK < N; jK++, mD += D, pos++,
                pos_sym += N) {
            float elem = 0;
            float Ym_0 = Y[mD];
            float Ym_1 = Y[mD+1];
            float Ynm_0 = Yn_0 - Ym_0;
            float Ynm_1 = Yn_1 - Ym_1;
            elem += Ynm_0 * Ynm_0;
            elem += Ynm_1 * Ynm_1;
            elem += 1;
            float elem_inv = 1 / elem;
            Q[pos] = elem_inv;
            Q[pos_sym] = elem_inv;
            sum_Q += 2 * elem_inv;
        }
    }

    return sum_Q;
}

inline float unfold_sr(float* Y, int N, int D, float* Q) {
    float sum_Q = 0.;


    int n, nD;
    for(n = 0, nD = 0; n < N; ++n, nD += D) {
        int mD = nD + D;

        int pos = n*N + n;
        int pos_sym = pos + N;
        Q[pos++] = 0;

        float Yn_0 = Y[nD];
        float Yn_1 = Y[nD+1];

        int m;
        // Unroll with 8 accumulators
        for (m = n + 1; m + 8 <= N; m += 8, mD += 8*D, pos += 8, pos_sym += 8*N) {
            float elem_0 = 0;
            float elem_1 = 0;
            float elem_2 = 0;
            float elem_3 = 0;
            float elem_4 = 0;
            float elem_5 = 0;
            float elem_6 = 0;
            float elem_7 = 0;

            float Ym0_0 = Y[mD];
            float Ym0_1 = Y[mD+1];
            float Ym1_0 = Y[mD+2];
            float Ym1_1 = Y[mD+3];
            float Ym2_0 = Y[mD+4];
            float Ym2_1 = Y[mD+5];
            float Ym3_0 = Y[mD+6];
            float Ym3_1 = Y[mD+7];
            float Ym4_0 = Y[mD+8];
            float Ym4_1 = Y[mD+9];
            float Ym5_0 = Y[mD+10];
            float Ym5_1 = Y[mD+11];
            float Ym6_0 = Y[mD+12];
            float Ym6_1 = Y[mD+13];
            float Ym7_0 = Y[mD+14];
            float Ym7_1 = Y[mD+15];

            float Ynm0_0 = Yn_0 - Ym0_0;
            float Ynm0_1 = Yn_1 - Ym0_1;
            float Ynm1_0 = Yn_0 - Ym1_0;
            float Ynm1_1 = Yn_1 - Ym1_1;
            float Ynm2_0 = Yn_0 - Ym2_0;
            float Ynm2_1 = Yn_1 - Ym2_1;
            float Ynm3_0 = Yn_0 - Ym3_0;
            float Ynm3_1 = Yn_1 - Ym3_1;
            float Ynm4_0 = Yn_0 - Ym4_0;
            float Ynm4_1 = Yn_1 - Ym4_1;
            float Ynm5_0 = Yn_0 - Ym5_0;
            float Ynm5_1 = Yn_1 - Ym5_1;
            float Ynm6_0 = Yn_0 - Ym6_0;
            float Ynm6_1 = Yn_1 - Ym6_1;
            float Ynm7_0 = Yn_0 - Ym7_0;
            float Ynm7_1 = Yn_1 - Ym7_1;

            elem_0 += Ynm0_0 * Ynm0_0;
            elem_1 += Ynm1_0 * Ynm1_0;
            elem_2 += Ynm2_0 * Ynm2_0;
            elem_3 += Ynm3_0 * Ynm3_0;
            elem_4 += Ynm4_0 * Ynm4_0;
            elem_5 += Ynm5_0 * Ynm5_0;
            elem_6 += Ynm6_0 * Ynm6_0;
            elem_7 += Ynm7_0 * Ynm7_0;

            elem_0 += Ynm0_1 * Ynm0_1;
            elem_1 += Ynm1_1 * Ynm1_1;
            elem_2 += Ynm2_1 * Ynm2_1;
            elem_3 += Ynm3_1 * Ynm3_1;
            elem_4 += Ynm4_1 * Ynm4_1;
            elem_5 += Ynm5_1 * Ynm5_1;
            elem_6 += Ynm6_1 * Ynm6_1;
            elem_7 += Ynm7_1 * Ynm7_1;

            elem_0 += 1;
            elem_1 += 1;
            elem_2 += 1;
            elem_3 += 1;
            elem_4 += 1;
            elem_5 += 1;
            elem_6 += 1;
            elem_7 += 1;


            float elem_inv_0 = 1 / elem_0;
            float elem_inv_1 = 1 / elem_1;
            float elem_inv_2 = 1 / elem_2;
            float elem_inv_3 = 1 / elem_3;
            float elem_inv_4 = 1 / elem_4;
            float elem_inv_5 = 1 / elem_5;
            float elem_inv_6 = 1 / elem_6;
            float elem_inv_7 = 1 / elem_7;

            float sum_0 = elem_inv_0 + elem_inv_1;
            float sum_1 = elem_inv_2 + elem_inv_3;
            float sum_2 = elem_inv_4 + elem_inv_5;
            float sum_3 = elem_inv_6 + elem_inv_7;
            float sum_4 = sum_0 + sum_1;
            float sum_5 = sum_2 + sum_3;
            float sum_6 = sum_4 + sum_5;

            sum_Q += 2 * sum_6;

            Q[pos] = elem_inv_0;
            Q[pos+1] = elem_inv_1;
            Q[pos+2] = elem_inv_2;
            Q[pos+3] = elem_inv_3;
            Q[pos+4] = elem_inv_4;
            Q[pos+5] = elem_inv_5;
            Q[pos+6] = elem_inv_6;
            Q[pos+7] = elem_inv_7;

            Q[pos_sym] = elem_inv_0;
            Q[pos_sym+N] = elem_inv_1;
            Q[pos_sym+2*N] = elem_inv_2;
            Q[pos_sym+3*N] = elem_inv_3;
            Q[pos_sym+4*N] = elem_inv_4;
            Q[pos_sym+5*N] = elem_inv_5;
            Q[pos_sym+6*N] = elem_inv_6;
            Q[pos_sym+7*N] = elem_inv_7;
        }
        // Unroll with 4 accumulators
        for (; m + 4 <= N; m += 4, mD += 4*D, pos += 4, pos_sym += 4*N) {
            float elem_0 = 0;
            float elem_1 = 0;
            float elem_2 = 0;
            float elem_3 = 0;

            float Ym0_0 = Y[mD];
            float Ym0_1 = Y[mD+1];
            float Ym1_0 = Y[mD+2];
            float Ym1_1 = Y[mD+3];
            float Ym2_0 = Y[mD+4];
            float Ym2_1 = Y[mD+5];
            float Ym3_0 = Y[mD+6];
            float Ym3_1 = Y[mD+7];

            float Ynm0_0 = Yn_0 - Ym0_0;
            float Ynm0_1 = Yn_1 - Ym0_1;
            float Ynm1_0 = Yn_0 - Ym1_0;
            float Ynm1_1 = Yn_1 - Ym1_1;
            float Ynm2_0 = Yn_0 - Ym2_0;
            float Ynm2_1 = Yn_1 - Ym2_1;
            float Ynm3_0 = Yn_0 - Ym3_0;
            float Ynm3_1 = Yn_1 - Ym3_1;

            elem_0 += Ynm0_0 * Ynm0_0;
            elem_1 += Ynm1_0 * Ynm1_0;
            elem_2 += Ynm2_0 * Ynm2_0;
            elem_3 += Ynm3_0 * Ynm3_0;

            elem_0 += Ynm0_1 * Ynm0_1;
            elem_1 += Ynm1_1 * Ynm1_1;
            elem_2 += Ynm2_1 * Ynm2_1;
            elem_3 += Ynm3_1 * Ynm3_1;

            elem_0 += 1;
            elem_1 += 1;
            elem_2 += 1;
            elem_3 += 1;


            float elem_inv_0 = 1 / elem_0;
            float elem_inv_1 = 1 / elem_1;
            float elem_inv_2 = 1 / elem_2;
            float elem_inv_3 = 1 / elem_3;


            float sum_0 = elem_inv_0 + elem_inv_1;
            float sum_1 = elem_inv_2 + elem_inv_3;
            float sum_3 = sum_0 + sum_1;

            sum_Q += 2 * sum_3;

            Q[pos] = elem_inv_0;
            Q[pos+1] = elem_inv_1;
            Q[pos+2] = elem_inv_2;
            Q[pos+3] = elem_inv_3;

            Q[pos_sym] = elem_inv_0;
            Q[pos_sym+N] = elem_inv_1;
            Q[pos_sym+2*N] = elem_inv_2;
            Q[pos_sym+3*N] = elem_inv_3;
        }
        // Compute the remaining positions
        for (; m < N; m++, mD += D, pos++, pos_sym += N) {
            float elem = 0;
            float Ym_0 = Y[mD];
            float Ym_1 = Y[mD+1];
            float Ynm_0 = Yn_0 - Ym_0;
            float Ynm_1 = Yn_1 - Ym_1;
            elem += Ynm_0 * Ynm_0;
            elem += Ynm_1 * Ynm_1;
            elem += 1;
            float elem_inv = 1 / elem;
            Q[pos] = elem_inv;
            Q[pos_sym] = elem_inv;
            sum_Q += 2 * elem_inv;
        }
    }
    return sum_Q;
}

inline float fused(float* Y, int N, int D, float* Q) {
    const float* XnD = Y;
    float sum_Q = 0.;

    for(int n = 0; n < N; ++n, XnD += D) {
        const float* XmD = XnD + D;
        float* curr_elem = &Q[n*N + n];
        *curr_elem = 0.0;
        float* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
				float dist = (XnD[d] - XmD[d]);
                *curr_elem += dist * dist;
            }

            *curr_elem += 1;
            *curr_elem = 1 / *curr_elem;

            *curr_elem_sym = *curr_elem;
            sum_Q += 2 * (*curr_elem);
        }
    }
    return sum_Q;
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
