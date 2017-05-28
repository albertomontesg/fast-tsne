#ifndef GRADIENT_COMPUTATION_UPDATE_NORMALIZE_H
#define GRADIENT_COMPUTATION_UPDATE_NORMALIZE_H

#include <stdio.h>
#include <immintrin.h>

inline void gradient_computation_update_normalize(float* Y, float* P, float* Q,
        float sum_Q, int N, int D, float* uY, float momentum, float eta) {
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


#endif
