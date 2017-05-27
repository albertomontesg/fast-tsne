#ifndef COMPUTE_SQUARED_EUCLEDIAN_DISTANCE_HIGH_NORMALIZE_H
#define COMPUTE_SQUARED_EUCLEDIAN_DISTANCE_HIGH_NORMALIZE_H

#include <x86intrin.h>

inline void compute_squared_eucledian_distance_high_normalize(float* X, int N, int D, float* DD, float* mean) {

    const int leftover_start = D-D%8;
    const int N_leftover_start = N-N%8;

    const __m256 sign_mask = _mm256_set1_ps(-0.f); // -0.f = 1 << 31

    int nD = 0;
    const float oneOverN = 1.0/N;
    const __m256 oneOverN_avx = _mm256_set1_ps(1.0/N);
    __m256 max = _mm256_setzero_ps();
    float max_leftover = 0;
    for(int d = 0; d < leftover_start; d+=8) {
         __m256 accum1 = _mm256_setzero_ps();
         __m256 accum2 = _mm256_setzero_ps();
         __m256 accum3 = _mm256_setzero_ps();
         __m256 accum4 = _mm256_setzero_ps();
         __m256 accum5 = _mm256_setzero_ps();
         __m256 accum6 = _mm256_setzero_ps();
         __m256 accum7 = _mm256_setzero_ps();
         __m256 accum8 = _mm256_setzero_ps();
        for(int n = 0; n < N_leftover_start; n+=8) {
            __m256 m1 = _mm256_load_ps(X + n*D + d);
            __m256 m2 = _mm256_load_ps(X + (n+1)*D + d);
            __m256 m3 = _mm256_load_ps(X + (n+2)*D + d);
            __m256 m4 = _mm256_load_ps(X + (n+3)*D + d);
            __m256 m5 = _mm256_load_ps(X + (n+4)*D + d);
            __m256 m6 = _mm256_load_ps(X + (n+5)*D + d);
            __m256 m7 = _mm256_load_ps(X + (n+6)*D + d);
            __m256 m8 = _mm256_load_ps(X + (n+7)*D + d);
            accum1 = _mm256_add_ps(accum1, m1);
            accum2 = _mm256_add_ps(accum2, m2);
            accum3 = _mm256_add_ps(accum3, m3);
            accum4 = _mm256_add_ps(accum4, m4);
            accum5 = _mm256_add_ps(accum5, m5);
            accum6 = _mm256_add_ps(accum6, m6);
            accum7 = _mm256_add_ps(accum7, m7);
            accum8 = _mm256_add_ps(accum8, m8);
        }
        for(int n = N_leftover_start; n < N; n++) {
            __m256 m1 = _mm256_load_ps(X + n*D + d);
            accum1 = _mm256_add_ps(accum1, m1);
        }
        __m256 accum12 = _mm256_add_ps(accum1, accum2);
        __m256 accum34 = _mm256_add_ps(accum3, accum4);
        __m256 accum56 = _mm256_add_ps(accum5, accum6);
        __m256 accum78 = _mm256_add_ps(accum7, accum8);
        __m256 accum1234 = _mm256_add_ps(accum12, accum34);
        __m256 accum5678 = _mm256_add_ps(accum56, accum78);
        __m256 accum = _mm256_add_ps(accum1234, accum5678);
        accum = _mm256_mul_ps(oneOverN_avx, accum);
        _mm256_store_ps(mean + d, accum);

        for(int n = 0; n < N; n++) {
            __m256 m1 = _mm256_load_ps(X + n*D + d);
            m1 = _mm256_sub_ps(m1, accum);
            _mm256_store_ps(X + n* D + d, m1);
            __m256 absx = _mm256_andnot_ps(sign_mask, m1);
            max = _mm256_max_ps(absx, max);
        }
    }
    for(int d = leftover_start; d < D; d++) {
         float accum1 = 0;
         float accum2 = 0;
         float accum3 = 0;
         float accum4 = 0;
         float accum5 = 0;
         float accum6 = 0;
         float accum7 = 0;
         float accum8 = 0;
        for(int n = 0; n < N_leftover_start; n+=8) {
            accum1 += X[n*D + d];
            accum2 += X[(n+1)*D + d];
            accum3 += X[(n+2)*D + d];
            accum4 += X[(n+3)*D + d];
            accum5 += X[(n+4)*D + d];
            accum6 += X[(n+5)*D + d];
            accum7 += X[(n+6)*D + d];
            accum8 += X[(n+7)*D + d];
        }
        for(int n = N_leftover_start; n < N; n++) {
            accum1 += X[n*D + d];
        }
        const float sum = accum1 + accum2 + accum3 + accum4 +
                          accum5 + accum6 + accum7 + accum8;
        const float res = oneOverN*sum;
        mean[d] = res;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= mean[d];
            max_leftover = fmaxf(max_leftover, fabsf(X[n*D + d]));
        }
    }
    __m128 max_lower = _mm256_extractf128_ps(max, 1);
    __m256 max_lower_256 = _mm256_castps128_ps256 (max_lower);
    __m256 max4 = _mm256_max_ps(max, max_lower_256);
    float* mm = (float*)&max4;
    float max_X = fmaxf(fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3])), max_leftover);
    const float factor = 1.0/(max_X*max_X);
    const __m256 factor_avx = _mm256_set1_ps(factor);


    int iN = 0;
    int iD = 0;
    for (int i = 0; i < N_leftover_start; i+=8)
    {
        int iiN = iN;
        int iiD = iD;
        for(int ii = i; ii < i+8; ii++)
        {
            DD[iiN + ii] = 0;
            int jjD = iiD+D;
            int jjN = iiN+N;
            for (int jj = ii+1; jj < i+8; jj++)
            {
                __m256 accum = _mm256_setzero_ps();
                int iiDk = iiD;
                int jjDk = jjD;
                for(int k = 0; k < leftover_start; k+=8)
                {
                    const __m256 x = _mm256_load_ps(X + iiDk);
                    const __m256 xj = _mm256_load_ps(X + jjDk);
                    const __m256 d = _mm256_sub_ps(x, xj);
                    accum = _mm256_fmadd_ps(d, d, accum);
                    iiDk += 8;
                    jjDk += 8;
                }
                float accum_leftover = 0.0f;
                for(int k = leftover_start; k < D; k++)
                {
                    const float d0 = X[iiD + k] - X[jjD + k];
                    accum_leftover += d0*d0;
                }
                __m256 hsum = _mm256_hadd_ps(accum, accum);
                float *hsumptr = (float *)&hsum;
                const float sum = hsumptr[0] + hsumptr[1] + hsumptr[4] + hsumptr[5] + accum_leftover;
                const float res = sum*factor;
                DD[iiN + jj] = res;
                DD[jjN + ii] = res;
                jjD += D;
                jjN += N;
            }
            iiN += N;
            iiD += D;
        }

        int jD = iD+8*D;
        int jN = iN+8*N;
        for (int j = i+8; j < N_leftover_start; j+=8)
        {
            iiN = iN;
            iiD = iD;
            int jNii = jN + i;
            for(int ii = i; ii < i+8; ii++)
            {
                __m256 b1_accum = _mm256_setzero_ps();
                __m256 b2_accum = _mm256_setzero_ps();
                __m256 b3_accum = _mm256_setzero_ps();
                __m256 b4_accum = _mm256_setzero_ps();
                __m256 b5_accum = _mm256_setzero_ps();
                __m256 b6_accum = _mm256_setzero_ps();
                __m256 b7_accum = _mm256_setzero_ps();
                __m256 b8_accum = _mm256_setzero_ps();
                int jDk = jD;
                for(int k = 0; k < leftover_start; k+=8)
                {
                    const __m256 x = _mm256_load_ps(X + iiD + k);
                    const __m256 b1_xj = _mm256_load_ps(X + jDk);
                    const __m256 b2_xj = _mm256_load_ps(X + jDk + D);
                    const __m256 b3_xj = _mm256_load_ps(X + jDk + 2*D);
                    const __m256 b4_xj = _mm256_load_ps(X + jDk + 3*D);
                    const __m256 b5_xj = _mm256_load_ps(X + jDk + 4*D);
                    const __m256 b6_xj = _mm256_load_ps(X + jDk + 5*D);
                    const __m256 b7_xj = _mm256_load_ps(X + jDk + 6*D);
                    const __m256 b8_xj = _mm256_load_ps(X + jDk + 7*D);
                    const __m256 b1_d = _mm256_sub_ps(x, b1_xj);
                    const __m256 b2_d = _mm256_sub_ps(x, b2_xj);
                    const __m256 b3_d = _mm256_sub_ps(x, b3_xj);
                    const __m256 b4_d = _mm256_sub_ps(x, b4_xj);
                    const __m256 b5_d = _mm256_sub_ps(x, b5_xj);
                    const __m256 b6_d = _mm256_sub_ps(x, b6_xj);
                    const __m256 b7_d = _mm256_sub_ps(x, b7_xj);
                    const __m256 b8_d = _mm256_sub_ps(x, b8_xj);
                    b1_accum = _mm256_fmadd_ps(b1_d, b1_d, b1_accum);
                    b2_accum = _mm256_fmadd_ps(b2_d, b2_d, b2_accum);
                    b3_accum = _mm256_fmadd_ps(b3_d, b3_d, b3_accum);
                    b4_accum = _mm256_fmadd_ps(b4_d, b4_d, b4_accum);
                    b5_accum = _mm256_fmadd_ps(b5_d, b5_d, b5_accum);
                    b6_accum = _mm256_fmadd_ps(b6_d, b6_d, b6_accum);
                    b7_accum = _mm256_fmadd_ps(b7_d, b7_d, b7_accum);
                    b8_accum = _mm256_fmadd_ps(b8_d, b8_d, b8_accum);
                    jDk += 8;
                }
                float b1_accum_leftover = 0.0f;
                float b2_accum_leftover = 0.0f;
                float b3_accum_leftover = 0.0f;
                float b4_accum_leftover = 0.0f;
                float b5_accum_leftover = 0.0f;
                float b6_accum_leftover = 0.0f;
                float b7_accum_leftover = 0.0f;
                float b8_accum_leftover = 0.0f;
                jDk = jD + leftover_start;
                for(int k = leftover_start; k < D; k++)
                {
                    const float x = X[iiD + k];
                    const float b1_xj = X[jDk];
                    const float b2_xj = X[jDk + D];
                    const float b3_xj = X[jDk + 2*D];
                    const float b4_xj = X[jDk + 3*D];
                    const float b5_xj = X[jDk + 4*D];
                    const float b6_xj = X[jDk + 5*D];
                    const float b7_xj = X[jDk + 6*D];
                    const float b8_xj = X[jDk + 7*D];
                    const float b1_d = x - b1_xj;
                    const float b2_d = x - b2_xj;
                    const float b3_d = x - b3_xj;
                    const float b4_d = x - b4_xj;
                    const float b5_d = x - b5_xj;
                    const float b6_d = x - b6_xj;
                    const float b7_d = x - b7_xj;
                    const float b8_d = x - b8_xj;
                    b1_accum_leftover += b1_d*b1_d;
                    b2_accum_leftover += b2_d*b2_d;
                    b3_accum_leftover += b3_d*b3_d;
                    b4_accum_leftover += b4_d*b4_d;
                    b5_accum_leftover += b5_d*b5_d;
                    b6_accum_leftover += b6_d*b6_d;
                    b7_accum_leftover += b7_d*b7_d;
                    b8_accum_leftover += b8_d*b8_d;
                    jDk++;
                }
                //12345678
                const __m256 leftover =  _mm256_set_ps (b8_accum_leftover, b7_accum_leftover, b6_accum_leftover, b5_accum_leftover, b4_accum_leftover, b3_accum_leftover, b2_accum_leftover, b1_accum_leftover);

                //11221122
                const __m256 hsum12 = _mm256_hadd_ps(b1_accum, b2_accum);
                //33443344
                const __m256 hsum34 = _mm256_hadd_ps(b3_accum, b4_accum);
                //55665566
                const __m256 hsum56 = _mm256_hadd_ps(b5_accum, b6_accum);
                //77887788
                const __m256 hsum78 = _mm256_hadd_ps(b7_accum, b8_accum);

                //12341234
                const __m256 hsum1234 = _mm256_hadd_ps(hsum12, hsum34);
                //56785678
                const __m256 hsum5678 = _mm256_hadd_ps(hsum56, hsum78);

                const __m256 hsum12345678a = _mm256_permute2f128_ps (hsum1234, hsum5678, 0x20);
                const __m256 hsum12345678b = _mm256_permute2f128_ps (hsum1234, hsum5678, 0x31);
                const __m256 hsum = _mm256_add_ps (hsum12345678a, hsum12345678b);
                const __m256 sum = _mm256_add_ps (leftover, hsum);
                const __m256 res = _mm256_mul_ps (factor_avx, sum);

                _mm256_store_ps (DD + iiN + j, res);
                float *res_ptr = (float *)&res;

                DD[jNii] = res_ptr[0];
                DD[jNii + N] = res_ptr[1];
                DD[jNii + 2*N] = res_ptr[2];
                DD[jNii + 3*N] = res_ptr[3];
                DD[jNii + 4*N] = res_ptr[4];
                DD[jNii + 5*N] = res_ptr[5];
                DD[jNii + 6*N] = res_ptr[6];
                DD[jNii + 7*N] = res_ptr[7];
                iiN += N;
                iiD += D;
                jNii++;
            }
            jD += 8*D;
            jN += 8*N;
        }
        for (int j = N_leftover_start; j < N; j++)
        {
            for(int ii = i; ii < i+8; ii++)
            {
                int jj = j;
                __m256 b1_accum = _mm256_setzero_ps();
                for(int k = 0; k < leftover_start; k+=8)
                {
                    const __m256 x = _mm256_load_ps(X + ii*D + k);
                    const __m256 b1_xj = _mm256_load_ps(X + jj*D + k);
                    const __m256 b1_d = _mm256_sub_ps(x, b1_xj);
                    b1_accum = _mm256_fmadd_ps(b1_d, b1_d, b1_accum);
                }
                float b1_accum_leftover = 0.0f;
                for(int k = leftover_start; k < D; k++)
                {
                    const float x = X[ii*D + k];
                    const float b1_xj = X[jj*D + k];
                    const float b1_d = x - b1_xj;
                    b1_accum_leftover += b1_d*b1_d;
                }
                b1_accum = _mm256_hadd_ps(b1_accum, b1_accum);

                const float sum0 = ((float *)&b1_accum)[0] + ((float *)&b1_accum)[1] +
                                   ((float *)&b1_accum)[4] + ((float *)&b1_accum)[5];
                const float sum = sum0 + b1_accum_leftover;
                const float res = sum*factor;
                DD[ii*N + jj] = res;
                DD[jj*N + ii] = res;
            }
        }
        iN += 8*N;
        iD += 8*D;
    }

}

#endif