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

inline float compute_low_dimensional_affinities(float* Y, int N, int D,
        float* Q) {
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

                    __m256 Ym_0 = _mm256_loadu_ps(Y+mD);
                    __m256 Ym_1 = _mm256_loadu_ps(Y+mD+8);

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

                    _mm256_storeu_ps(Q+nN, q_0);
                    _mm256_storeu_ps(Q+nN+N, q_1);
                    _mm256_storeu_ps(Q+nN+2*N, q_2);
                    _mm256_storeu_ps(Q+nN+3*N, q_3);
                    _mm256_storeu_ps(Q+nN+4*N, q_4);
                    _mm256_storeu_ps(Q+nN+5*N, q_5);
                    _mm256_storeu_ps(Q+nN+6*N, q_6);
                    _mm256_storeu_ps(Q+nN+7*N, q_7);


                    // Check if we are in the diagonal or not
                    if (jK > iK) {

                        transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                        cum_sum = _mm256_add_ps(cum_sum, sum6);

                        _mm256_storeu_ps(Q+mN, q_0);
                        _mm256_storeu_ps(Q+mN+N, q_1);
                        _mm256_storeu_ps(Q+mN+2*N, q_2);
                        _mm256_storeu_ps(Q+mN+3*N, q_3);
                        _mm256_storeu_ps(Q+mN+4*N, q_4);
                        _mm256_storeu_ps(Q+mN+5*N, q_5);
                        _mm256_storeu_ps(Q+mN+6*N, q_6);
                        _mm256_storeu_ps(Q+mN+7*N, q_7);

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

                __m256 Ym_0 = _mm256_loadu_ps(Y+mD);
                __m256 Ym_1 = _mm256_loadu_ps(Y+mD+8);

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

                _mm256_storeu_ps(Q+nN, q_0);
                _mm256_storeu_ps(Q+nN+N, q_1);
                _mm256_storeu_ps(Q+nN+2*N, q_2);
                _mm256_storeu_ps(Q+nN+3*N, q_3);
                _mm256_storeu_ps(Q+nN+4*N, q_4);
                _mm256_storeu_ps(Q+nN+5*N, q_5);
                _mm256_storeu_ps(Q+nN+6*N, q_6);
                _mm256_storeu_ps(Q+nN+7*N, q_7);

                transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                cum_sum = _mm256_add_ps(cum_sum, sum6);

                _mm256_storeu_ps(Q+mN, q_0);
                _mm256_storeu_ps(Q+mN+N, q_1);
                _mm256_storeu_ps(Q+mN+2*N, q_2);
                _mm256_storeu_ps(Q+mN+3*N, q_3);
                _mm256_storeu_ps(Q+mN+4*N, q_4);
                _mm256_storeu_ps(Q+mN+5*N, q_5);
                _mm256_storeu_ps(Q+mN+6*N, q_6);
                _mm256_storeu_ps(Q+mN+7*N, q_7);


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

            __m256 Ym_0 = _mm256_loadu_ps(Y+mD);
            __m256 Ym_1 = _mm256_loadu_ps(Y+mD+8);

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

            _mm256_storeu_ps(Q+nN, q_0);
            _mm256_storeu_ps(Q+nN+N, q_1);
            _mm256_storeu_ps(Q+nN+2*N, q_2);
            _mm256_storeu_ps(Q+nN+3*N, q_3);
            _mm256_storeu_ps(Q+nN+4*N, q_4);
            _mm256_storeu_ps(Q+nN+5*N, q_5);
            _mm256_storeu_ps(Q+nN+6*N, q_6);
            _mm256_storeu_ps(Q+nN+7*N, q_7);


            // Check if we are in the diagonal or not
            if (jB > iB) {

                transpose8_ps(q_0, q_1, q_2, q_3, q_4, q_5, q_6, q_7);

                cum_sum = _mm256_add_ps(cum_sum, sum6);

                _mm256_storeu_ps(Q+mN, q_0);
                _mm256_storeu_ps(Q+mN+N, q_1);
                _mm256_storeu_ps(Q+mN+2*N, q_2);
                _mm256_storeu_ps(Q+mN+3*N, q_3);
                _mm256_storeu_ps(Q+mN+4*N, q_4);
                _mm256_storeu_ps(Q+mN+5*N, q_5);
                _mm256_storeu_ps(Q+mN+6*N, q_6);
                _mm256_storeu_ps(Q+mN+7*N, q_7);

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

#endif
