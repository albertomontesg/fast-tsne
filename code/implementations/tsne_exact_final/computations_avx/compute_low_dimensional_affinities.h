#ifndef COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H
#define COMPUTE_LOW_DIMENSIONAL_AFFINITIES_H

#include <stdio.h>
#include <immintrin.h>

inline float compute_low_dimensional_affinities(float* Y, int N, int D, float* Q) {
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


#endif
