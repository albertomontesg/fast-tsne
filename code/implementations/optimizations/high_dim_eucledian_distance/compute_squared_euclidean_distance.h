#ifndef COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H
#define COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H

#include "../../utils/data_type.h"
#include <iostream>
#include <x86intrin.h>
// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void base_version(float* X, int N, int D, float* DD, int max_value) {
    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) {
        mean[d] /= (float) N;
    }

    // Subtract data mean
    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
        }
        nD += D;
    }

    if (max_value > 0) {
        // Normalize to the maximum absolute value
        float max_X = .0;
        for(int i = 0; i < N * D; i++) {
            if(fabsf(X[i]) > max_X) max_X = fabsf(X[i]);
        }
        for(int i = 0; i < N * D; i++) X[i] /= max_X;
        // std::cout << max_X << std::endl;
    }

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



inline void fast_scalar(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;

    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) {
        mean[d] /= (float) N;
    }

    // Subtract data mean
    float max_X = .0;
    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
            if(fabsf(X[nD + d]) > max_X) max_X = fabsf(X[nD + d]);
        }
        nD += D;
    }
    // std::cout << max_X << std::endl;
    const float factor = 1.0/(max_X*max_X);

    int iN = 0; int iD = 0;
    for (int i = 0; i < N; ++i)
    {
        DD[iN + i] = 0;
        int jD = iD + D;
        int jN = iN + N;
        for (int j = i+1; j < N; ++j)
        {
            const int iNj = iN + j;
            // DD[iNj] = 0;
            float accum0 = 0;
            float accum1 = 0;
            float accum2 = 0;
            float accum3 = 0;
            float accum4 = 0;
            float accum5 = 0;
            float accum6 = 0;
            float accum7 = 0;

            for(int d = 0; d < leftover_start; d+=8)
            {
                const int iDd = iD + d;
                const int jDd = jD + d;
                const float dist0 = X[iDd    ] - X[jDd    ];
                const float dist1 = X[iDd+1] - X[jDd+1];
                const float dist2 = X[iDd+2] - X[jDd+2];
                const float dist3 = X[iDd+3] - X[jDd+3];
                const float dist4 = X[iDd+4] - X[jDd+4];
                const float dist5 = X[iDd+5] - X[jDd+5];
                const float dist6 = X[iDd+6] - X[jDd+6];
                const float dist7 = X[iDd+7] - X[jDd+7];
                const float prod0 = dist0 * dist0;
                const float prod1 = dist1 * dist1;
                const float prod2 = dist2 * dist2;
                const float prod3 = dist3 * dist3;
                const float prod4 = dist4 * dist4;
                const float prod5 = dist5 * dist5;
                const float prod6 = dist6 * dist6;
                const float prod7 = dist7 * dist7;
                accum0 += prod0;
                accum1 += prod1;
                accum2 += prod2;
                accum3 += prod3;
                accum4 += prod4;
                accum5 += prod5;
                accum6 += prod6;
                accum7 += prod7;
                // std::cout << d << std::endl;
            }
            
            const float sum0 = accum1 + accum2 + accum3 + accum4 + accum5 + accum6 + accum7; 
            for(int d = leftover_start; d < D; d++)
            {
                // std::cout << d << std::endl;
                const float dist0 = X[iD + d] - X[jD + d];
                const float prod0 = dist0 * dist0;
                accum0 += prod0;
            }
            const float sum1 = accum0 + sum0;
            const float dist = sum1 * factor;
            DD[iNj] = dist;
            DD[jN + i] = dist;
            jD += D;
            jN += N;
        }
        iN += N;
        iD += D;
    }
}

inline void fast_scalar_start_matter(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;
    const float oneOverN = 1.0/N;
    int nD = 0;
    float max_X = .0;
    for(int d = 0; d < leftover_start; d+=8) {
        float accum0 = 0.0;
        float accum1 = 0.0;
        float accum2 = 0.0;
        float accum3 = 0.0;
        float accum4 = 0.0;
        float accum5 = 0.0;
        float accum6 = 0.0;
        float accum7 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
            accum1 += X[n*D + d + 1];
            accum2 += X[n*D + d + 2];
            accum3 += X[n*D + d + 3];
            accum4 += X[n*D + d + 4];
            accum5 += X[n*D + d + 5];
            accum6 += X[n*D + d + 6];
            accum7 += X[n*D + d + 7];
        }
        accum0 *= oneOverN;
        accum1 *= oneOverN;
        accum2 *= oneOverN;
        accum3 *= oneOverN;
        accum4 *= oneOverN;
        accum5 *= oneOverN;
        accum6 *= oneOverN;
        accum7 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            X[n*D + d + 1] -= accum1;
            X[n*D + d + 2] -= accum2;
            X[n*D + d + 3] -= accum3;
            X[n*D + d + 4] -= accum4;
            X[n*D + d + 5] -= accum5;
            X[n*D + d + 6] -= accum6;
            X[n*D + d + 7] -= accum7;
            const float max0 = fmaxf(fabsf(X[n*D + d]), fabsf(X[n*D + d + 1]));
            const float max1 = fmaxf(fabsf(X[n*D + d + 2]), fabsf(X[n*D + d + 3]));
            const float max2 = fmaxf(fabsf(X[n*D + d + 4]), fabsf(X[n*D + d + 5]));
            const float max3 = fmaxf(fabsf(X[n*D + d + 6]), fabsf(X[n*D + d + 7]));
            const float max4 = fmaxf(max0, max1);
            const float max5 = fmaxf(max2, max3);
            const float max6 = fmaxf(max4, max5);
            max_X = fmaxf(max_X, max6);
        }
    }
    for(int d = leftover_start; d < D; d++) {
        float accum0 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
        }
        accum0 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            max_X = fmaxf(max_X, fabsf(X[n*D + d]));
        }
    }


    // Subtract data mean
    // std::cout << max_X << std::endl;
    const float factor = 1.0/(max_X*max_X);

    int iN = 0; int iD = 0;
    for (int i = 0; i < N; ++i)
    {
        DD[iN + i] = 0;
        int jD = iD + D;
        int jN = iN + N;
        for (int j = i+1; j < N; ++j)
        {
            const int iNj = iN + j;
            // DD[iNj] = 0;
            float accum0 = 0;
            float accum1 = 0;
            float accum2 = 0;
            float accum3 = 0;
            float accum4 = 0;
            float accum5 = 0;
            float accum6 = 0;
            float accum7 = 0;

            for(int d = 0; d < leftover_start; d+=8)
            {
                const int iDd = iD + d;
                const int jDd = jD + d;
                const float dist0 = X[iDd    ] - X[jDd    ];
                const float dist1 = X[iDd+1] - X[jDd+1];
                const float dist2 = X[iDd+2] - X[jDd+2];
                const float dist3 = X[iDd+3] - X[jDd+3];
                const float dist4 = X[iDd+4] - X[jDd+4];
                const float dist5 = X[iDd+5] - X[jDd+5];
                const float dist6 = X[iDd+6] - X[jDd+6];
                const float dist7 = X[iDd+7] - X[jDd+7];
                const float prod0 = dist0 * dist0;
                const float prod1 = dist1 * dist1;
                const float prod2 = dist2 * dist2;
                const float prod3 = dist3 * dist3;
                const float prod4 = dist4 * dist4;
                const float prod5 = dist5 * dist5;
                const float prod6 = dist6 * dist6;
                const float prod7 = dist7 * dist7;
                accum0 += prod0;
                accum1 += prod1;
                accum2 += prod2;
                accum3 += prod3;
                accum4 += prod4;
                accum5 += prod5;
                accum6 += prod6;
                accum7 += prod7;
                // std::cout << d << std::endl;
            }
            
            const float sum0 = accum1 + accum2 + accum3 + accum4 + accum5 + accum6 + accum7; 
            for(int d = leftover_start; d < D; d++)
            {
                // std::cout << d << std::endl;
                const float dist0 = X[iD + d] - X[jD + d];
                const float prod0 = dist0 * dist0;
                accum0 += prod0;
            }
            const float sum1 = accum0 + sum0;
            const float dist = sum1 * factor;
            DD[iNj] = dist;
            DD[jN + i] = dist;
            jD += D;
            jN += N;
        }
        iN += N;
        iD += D;
    }
}


inline void fast_scalar_avx(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;

    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }

    for(int d = 0; d < D; d++) {
        mean[d] /= (float) N;
    }

    // Subtract data mean
    float max_X = .0;
    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
            if(fabsf(X[nD + d]) > max_X) max_X = fabsf(X[nD + d]);
        }
        nD += D;
    }
    // std::cout << max_X << std::endl;
    const float factor = 1.0/(max_X*max_X);

    int iN = 0; int iD = 0;
    for (int i = 0; i < N; ++i)
    {
        DD[iN + i] = 0;
        int jD = iD + D;
        int jN = iN + N;
        for (int j = i+1; j < N; ++j)
        {
            const int iNj = iN + j;
            __m256 accum = _mm256_setzero_ps();
            for(int d = 0; d < leftover_start; d+=8)
            {
                const __m256 x1 = _mm256_loadu_ps(X + i*D+d);
                const __m256 x2 = _mm256_loadu_ps(X + j*D+d);
                const __m256 diff = _mm256_sub_ps(x1, x2);
                accum = _mm256_fmadd_ps (diff, diff, accum);
            }
            accum = _mm256_hadd_ps(accum, accum);

            const float sum0 = ((float *)&accum)[0] + ((float *)&accum)[1] +
                               ((float *)&accum)[4] + ((float *)&accum)[5]; 
            float accum0 = 0;
            for(int d = leftover_start; d < D; d++)
            {
                // std::cout << d << std::endl;
                const float dist0 = X[iD + d] - X[jD + d];
                const float prod0 = dist0 * dist0;
                accum0 += prod0;
            }
            const float sum1 = accum0 + sum0;
            const float dist = sum1 * factor;
            DD[iNj] = dist;
            DD[jN + i] = dist;
            jD += D;
            jN += N;
        }
        iN += N;
        iD += D;
    }
}

inline void fast_scalar_avx_start_matter(float* X, int N, int D, float* DD, float* mean, int max_value) {

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
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
            __m256 m2 = _mm256_loadu_ps(X + (n+1)*D + d);
            __m256 m3 = _mm256_loadu_ps(X + (n+2)*D + d);
            __m256 m4 = _mm256_loadu_ps(X + (n+3)*D + d);
            __m256 m5 = _mm256_loadu_ps(X + (n+4)*D + d);
            __m256 m6 = _mm256_loadu_ps(X + (n+5)*D + d);
            __m256 m7 = _mm256_loadu_ps(X + (n+6)*D + d);
            __m256 m8 = _mm256_loadu_ps(X + (n+7)*D + d);
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
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
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
        //_mm256_storeu_ps(mean + d, accum);

        for(int n = 0; n < N; n++) {
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
            m1 = _mm256_sub_ps(m1, accum);
            _mm256_storeu_ps(X + n* D + d, m1);
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

    int iN = 0; int iD = 0;
    for (int i = 0; i < N; ++i)
    {
        DD[iN + i] = 0;
        int jD = iD + D;
        int jN = iN + N;
        for (int j = i+1; j < N; ++j)
        {
            const int iNj = iN + j;
            __m256 accum = _mm256_setzero_ps();
            for(int d = 0; d < leftover_start; d+=8)
            {
                const __m256 x1 = _mm256_loadu_ps(X + i*D+d);
                const __m256 x2 = _mm256_loadu_ps(X + j*D+d);
                const __m256 diff = _mm256_sub_ps(x1, x2);
                accum = _mm256_fmadd_ps (diff, diff, accum);
            }
            accum = _mm256_hadd_ps(accum, accum);

            const float sum0 = ((float *)&accum)[0] + ((float *)&accum)[1] +
                               ((float *)&accum)[4] + ((float *)&accum)[5]; 
            float accum0 = 0;
            for(int d = leftover_start; d < D; d++)
            {
                // std::cout << d << std::endl;
                const float dist0 = X[iD + d] - X[jD + d];
                const float prod0 = dist0 * dist0;
                accum0 += prod0;
            }
            const float sum1 = accum0 + sum0;
            const float dist = sum1 * factor;
            DD[iNj] = dist;
            DD[jN + i] = dist;
            jD += D;
            jN += N;
        }
        iN += N;
        iD += D;
    }
}

inline void fast_scalar_4x4_base(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;
    const float oneOverN = 1.0/N;
    int nD = 0;
    float max_X = .0;
    for(int d = 0; d < leftover_start; d+=8) {
        float accum0 = 0.0;
        float accum1 = 0.0;
        float accum2 = 0.0;
        float accum3 = 0.0;
        float accum4 = 0.0;
        float accum5 = 0.0;
        float accum6 = 0.0;
        float accum7 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
            accum1 += X[n*D + d + 1];
            accum2 += X[n*D + d + 2];
            accum3 += X[n*D + d + 3];
            accum4 += X[n*D + d + 4];
            accum5 += X[n*D + d + 5];
            accum6 += X[n*D + d + 6];
            accum7 += X[n*D + d + 7];
        }
        accum0 *= oneOverN;
        accum1 *= oneOverN;
        accum2 *= oneOverN;
        accum3 *= oneOverN;
        accum4 *= oneOverN;
        accum5 *= oneOverN;
        accum6 *= oneOverN;
        accum7 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            X[n*D + d + 1] -= accum1;
            X[n*D + d + 2] -= accum2;
            X[n*D + d + 3] -= accum3;
            X[n*D + d + 4] -= accum4;
            X[n*D + d + 5] -= accum5;
            X[n*D + d + 6] -= accum6;
            X[n*D + d + 7] -= accum7;
            const float max0 = fmaxf(fabsf(X[n*D + d]), fabsf(X[n*D + d + 1]));
            const float max1 = fmaxf(fabsf(X[n*D + d + 2]), fabsf(X[n*D + d + 3]));
            const float max2 = fmaxf(fabsf(X[n*D + d + 4]), fabsf(X[n*D + d + 5]));
            const float max3 = fmaxf(fabsf(X[n*D + d + 6]), fabsf(X[n*D + d + 7]));
            const float max4 = fmaxf(max0, max1);
            const float max5 = fmaxf(max2, max3);
            const float max6 = fmaxf(max4, max5);
            max_X = fmaxf(max_X, max6);
        }
    }
    for(int d = leftover_start; d < D; d++) {
        float accum0 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
        }
        accum0 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            max_X = fmaxf(max_X, fabsf(X[n*D + d]));
        }
    }

    // std::cout << max_X << std::endl;
    const float factor = 1.0/(max_X*max_X);
    //const __m256 avx_factor = _mm256_set1_ps(factor);

    for (int i = 0; i < N; i+=4)
    {
        for(int ii = i; ii < i+4; ii++)
        {
            DD[ii*N + ii] = 0;
            for (int jj = ii+1; jj < i+4; jj++)
            {
                float accum0 = 0.0;
                float accum1 = 0.0;
                float accum2 = 0.0;
                float accum3 = 0.0;
                for(int k = 0; k < D; k+=4)
                {
                    const float d0 = X[ii*D + k] - X[jj*D + k];
                    const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                    const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                    const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                    accum0 += d0*d0;
                    accum1 += d1*d1;
                    accum2 += d2*d2;
                    accum3 += d3*d3;
                }
                const float sum = accum0 + accum1 + accum2 + accum3;
                const float res = sum*factor;
                DD[ii*N + jj] = res;
                DD[jj*N + ii] = res;
            }            
        }

        for (int j = i+4; j < N; j+=4)
        {
            // for(int k = 0; k < D; k+=4)
            {
                for(int ii = i; ii < i+4; ii++)
                {
                    for(int jj = j; jj < j+4; jj++)
                    {
                        float accum0 = 0.0;
                        float accum1 = 0.0;
                        float accum2 = 0.0;
                        float accum3 = 0.0;
                        for(int k = 0; k < D; k+=4)
                        {
                            const float d0 = X[ii*D + k] - X[jj*D + k];
                            const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                            const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                            const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                            accum0 += d0*d0;
                            accum1 += d1*d1;
                            accum2 += d2*d2;
                            accum3 += d3*d3;
                        }
                        const float sum = accum0 + accum1 + accum2 + accum3;
                        const float res = sum*factor;
                        DD[ii*N + jj] += res;
                        DD[jj*N + ii] += res;
                    }
                }
            }
        }
    }

}

inline void fast_scalar_8x8_base(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;
    const float oneOverN = 1.0/N;
    int nD = 0;
    float max_X = .0;
    for(int d = 0; d < leftover_start; d+=8) {
        float accum0 = 0.0;
        float accum1 = 0.0;
        float accum2 = 0.0;
        float accum3 = 0.0;
        float accum4 = 0.0;
        float accum5 = 0.0;
        float accum6 = 0.0;
        float accum7 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
            accum1 += X[n*D + d + 1];
            accum2 += X[n*D + d + 2];
            accum3 += X[n*D + d + 3];
            accum4 += X[n*D + d + 4];
            accum5 += X[n*D + d + 5];
            accum6 += X[n*D + d + 6];
            accum7 += X[n*D + d + 7];
        }
        accum0 *= oneOverN;
        accum1 *= oneOverN;
        accum2 *= oneOverN;
        accum3 *= oneOverN;
        accum4 *= oneOverN;
        accum5 *= oneOverN;
        accum6 *= oneOverN;
        accum7 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            X[n*D + d + 1] -= accum1;
            X[n*D + d + 2] -= accum2;
            X[n*D + d + 3] -= accum3;
            X[n*D + d + 4] -= accum4;
            X[n*D + d + 5] -= accum5;
            X[n*D + d + 6] -= accum6;
            X[n*D + d + 7] -= accum7;
            const float max0 = fmaxf(fabsf(X[n*D + d]), fabsf(X[n*D + d + 1]));
            const float max1 = fmaxf(fabsf(X[n*D + d + 2]), fabsf(X[n*D + d + 3]));
            const float max2 = fmaxf(fabsf(X[n*D + d + 4]), fabsf(X[n*D + d + 5]));
            const float max3 = fmaxf(fabsf(X[n*D + d + 6]), fabsf(X[n*D + d + 7]));
            const float max4 = fmaxf(max0, max1);
            const float max5 = fmaxf(max2, max3);
            const float max6 = fmaxf(max4, max5);
            max_X = fmaxf(max_X, max6);
        }
    }
    for(int d = leftover_start; d < D; d++) {
        float accum0 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
        }
        accum0 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            max_X = fmaxf(max_X, fabsf(X[n*D + d]));
        }
    }
    const float factor = 1.0/(max_X*max_X);

    for (int i = 0; i < N; i+=8)
    {
        for(int ii = i; ii < i+8; ii++)
        {
            DD[ii*N + ii] = 0;
            for (int jj = ii+1; jj < i+8; jj++)
            {
                float accum0 = 0.0;
                float accum1 = 0.0;
                float accum2 = 0.0;
                float accum3 = 0.0;
                for(int k = 0; k < D; k+=4)
                {
                    const float d0 = X[ii*D + k] - X[jj*D + k];
                    const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                    const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                    const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                    accum0 += d0*d0;
                    accum1 += d1*d1;
                    accum2 += d2*d2;
                    accum3 += d3*d3;
                }
                const float sum = accum0 + accum1 + accum2 + accum3;
                const float res = sum*factor;
                DD[ii*N + jj] = res;
                DD[jj*N + ii] = res;
            }            
        }

        for (int j = i+8; j < N; j+=8)
        {
            // for(int k = 0; k < D; k+=4)
            {
                for(int ii = i; ii < i+8; ii++)
                {
                    for(int jj = j; jj < j+8; jj++)
                    {
                        float accum0 = 0.0;
                        float accum1 = 0.0;
                        float accum2 = 0.0;
                        float accum3 = 0.0;
                        for(int k = 0; k < D; k+=4)
                        {
                            const float d0 = X[ii*D + k] - X[jj*D + k];
                            const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                            const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                            const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                            accum0 += d0*d0;
                            accum1 += d1*d1;
                            accum2 += d2*d2;
                            accum3 += d3*d3;
                        }
                        const float sum = accum0 + accum1 + accum2 + accum3;
                        const float res = sum*factor;
                        DD[ii*N + jj] += res;
                        DD[jj*N + ii] += res;
                    }
                }
            }
        }
    }

}


inline void fast_scalar_8x8_base_k(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;

    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) {
        mean[d] /= (float) N;
    }

    // Subtract data mean
    float max_X = .0;
    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
            if(fabsf(X[nD + d]) > max_X) max_X = fabsf(X[nD + d]);
        }
        nD += D;
    }
    // std::cout << max_X << std::endl;
    const float factor = 1.0/(max_X*max_X);

    for (int i = 0; i < N; i+=8)
    {
        for(int ii = i; ii < i+8; ii++)
        {
            DD[ii*N + ii] = 0;
            for (int jj = ii+1; jj < i+8; jj++)
            {
                float accum0 = 0.0;
                float accum1 = 0.0;
                float accum2 = 0.0;
                float accum3 = 0.0;
                for(int k = 0; k < D; k+=4)
                {
                    const float d0 = X[ii*D + k] - X[jj*D + k];
                    const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                    const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                    const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                    accum0 += d0*d0;
                    accum1 += d1*d1;
                    accum2 += d2*d2;
                    accum3 += d3*d3;
                }
                const float sum = accum0 + accum1 + accum2 + accum3;
                const float res = sum*factor;
                DD[ii*N + jj] = res;
                DD[jj*N + ii] = res;
            }            
        }

        for (int j = i+8; j < N; j+=8)
        {
            for(int k = 0; k < D; k+=4)
            {
                for(int ii = i; ii < i+8; ii++)
                {
                    for(int jj = j; jj < j+8; jj++)
                    {
                        float accum0 = 0.0;
                        float accum1 = 0.0;
                        float accum2 = 0.0;
                        float accum3 = 0.0;
                        const float d0 = X[ii*D + k] - X[jj*D + k];
                        const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                        const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                        const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                        accum0 += d0*d0;
                        accum1 += d1*d1;
                        accum2 += d2*d2;
                        accum3 += d3*d3;
                        const float sum = accum0 + accum1 + accum2 + accum3;
                        const float res = sum*factor;
                        DD[ii*N + jj] += res;
                        DD[jj*N + ii] += res;
                    }
                }
            }
        }
    }
}
// }


inline void fast_scalar_8x8x8_big_unroll(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;
    const float oneOverN = 1.0/N;
    int nD = 0;
    float max_X = .0;
    for(int d = 0; d < leftover_start; d+=8) {
        float accum0 = 0.0;
        float accum1 = 0.0;
        float accum2 = 0.0;
        float accum3 = 0.0;
        float accum4 = 0.0;
        float accum5 = 0.0;
        float accum6 = 0.0;
        float accum7 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
            accum1 += X[n*D + d + 1];
            accum2 += X[n*D + d + 2];
            accum3 += X[n*D + d + 3];
            accum4 += X[n*D + d + 4];
            accum5 += X[n*D + d + 5];
            accum6 += X[n*D + d + 6];
            accum7 += X[n*D + d + 7];
        }
        accum0 *= oneOverN;
        accum1 *= oneOverN;
        accum2 *= oneOverN;
        accum3 *= oneOverN;
        accum4 *= oneOverN;
        accum5 *= oneOverN;
        accum6 *= oneOverN;
        accum7 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            X[n*D + d + 1] -= accum1;
            X[n*D + d + 2] -= accum2;
            X[n*D + d + 3] -= accum3;
            X[n*D + d + 4] -= accum4;
            X[n*D + d + 5] -= accum5;
            X[n*D + d + 6] -= accum6;
            X[n*D + d + 7] -= accum7;
            const float max0 = fmaxf(fabsf(X[n*D + d]), fabsf(X[n*D + d + 1]));
            const float max1 = fmaxf(fabsf(X[n*D + d + 2]), fabsf(X[n*D + d + 3]));
            const float max2 = fmaxf(fabsf(X[n*D + d + 4]), fabsf(X[n*D + d + 5]));
            const float max3 = fmaxf(fabsf(X[n*D + d + 6]), fabsf(X[n*D + d + 7]));
            const float max4 = fmaxf(max0, max1);
            const float max5 = fmaxf(max2, max3);
            const float max6 = fmaxf(max4, max5);
            max_X = fmaxf(max_X, max6);
        }
    }
    for(int d = leftover_start; d < D; d++) {
        float accum0 = 0.0;
        for(int n = 0; n < N; n++) {
            accum0 += X[n*D + d];
        }
        accum0 *= oneOverN;
        for(int n = 0; n < N; n++) {
            X[n*D + d] -= accum0;
            max_X = fmaxf(max_X, fabsf(X[n*D + d]));
        }
    }
    const float factor = 1.0/(max_X*max_X);

    for (int i = 0; i < N; i+=8)
    {
        for(int ii = i; ii < i+8; ii++)
        {
            DD[ii*N + ii] = 0;
            for (int jj = ii+1; jj < i+8; jj++)
            {
                float accum0 = 0.0;
                float accum1 = 0.0;
                float accum2 = 0.0;
                float accum3 = 0.0;
                for(int k = 0; k < D; k+=4)
                {
                    const float d0 = X[ii*D + k] - X[jj*D + k];
                    const float d1 = X[ii*D + k + 1] - X[jj*D + k + 1];
                    const float d2 = X[ii*D + k + 2] - X[jj*D + k + 2];
                    const float d3 = X[ii*D + k + 3] - X[jj*D + k + 3];
                    accum0 += d0*d0;
                    accum1 += d1*d1;
                    accum2 += d2*d2;
                    accum3 += d3*d3;
                }
                const float sum = accum0 + accum1 + accum2 + accum3;
                const float res = sum*factor;
                DD[ii*N + jj] = res;
                DD[jj*N + ii] = res;
            }  
        }

        for (int j = i+8; j < N; j+=8)
        {
            // for(int k = 0; k < D; k+=4)
            {
                for(int ii = i; ii < i+8; ii++)
                {
            int jj = j;
            float b1_accum0 = 0.0;
            float b1_accum1 = 0.0;
            float b1_accum2 = 0.0;
            float b1_accum3 = 0.0;
            float b1_accum4 = 0.0;
            float b1_accum5 = 0.0;
            float b1_accum6 = 0.0;
            float b1_accum7 = 0.0;

            float b2_accum0 = 0.0;
            float b2_accum1 = 0.0;
            float b2_accum2 = 0.0;
            float b2_accum3 = 0.0;
            float b2_accum4 = 0.0;
            float b2_accum5 = 0.0;
            float b2_accum6 = 0.0;
            float b2_accum7 = 0.0;

            float b3_accum0 = 0.0;
            float b3_accum1 = 0.0;
            float b3_accum2 = 0.0;
            float b3_accum3 = 0.0;
            float b3_accum4 = 0.0;
            float b3_accum5 = 0.0;
            float b3_accum6 = 0.0;
            float b3_accum7 = 0.0;

            float b4_accum0 = 0.0;
            float b4_accum1 = 0.0;
            float b4_accum2 = 0.0;
            float b4_accum3 = 0.0;
            float b4_accum4 = 0.0;
            float b4_accum5 = 0.0;
            float b4_accum6 = 0.0;
            float b4_accum7 = 0.0;

            float b5_accum0 = 0.0;
            float b5_accum1 = 0.0;
            float b5_accum2 = 0.0;
            float b5_accum3 = 0.0;
            float b5_accum4 = 0.0;
            float b5_accum5 = 0.0;
            float b5_accum6 = 0.0;
            float b5_accum7 = 0.0;

            float b6_accum0 = 0.0;
            float b6_accum1 = 0.0;
            float b6_accum2 = 0.0;
            float b6_accum3 = 0.0;
            float b6_accum4 = 0.0;
            float b6_accum5 = 0.0;
            float b6_accum6 = 0.0;
            float b6_accum7 = 0.0;

            float b7_accum0 = 0.0;
            float b7_accum1 = 0.0;
            float b7_accum2 = 0.0;
            float b7_accum3 = 0.0;
            float b7_accum4 = 0.0;
            float b7_accum5 = 0.0;
            float b7_accum6 = 0.0;
            float b7_accum7 = 0.0;

            float b8_accum0 = 0.0;
            float b8_accum1 = 0.0;
            float b8_accum2 = 0.0;
            float b8_accum3 = 0.0;
            float b8_accum4 = 0.0;
            float b8_accum5 = 0.0;
            float b8_accum6 = 0.0;
            float b8_accum7 = 0.0;
            for(int k = 0; k < D; k+=8)
            {
                const float x0 = X[ii*D + k];
                const float x1 = X[ii*D + k + 1];
                const float x2 = X[ii*D + k + 2];
                const float x3 = X[ii*D + k + 3];
                const float x4 = X[ii*D + k + 4];
                const float x5 = X[ii*D + k + 5];
                const float x6 = X[ii*D + k + 6];
                const float x7 = X[ii*D + k + 7];
                const float b1_jx0 = X[jj*D + k];
                const float b1_jx1 = X[jj*D + k + 1];
                const float b1_jx2 = X[jj*D + k + 2];
                const float b1_jx3 = X[jj*D + k + 3];
                const float b1_jx4 = X[jj*D + k + 4];
                const float b1_jx5 = X[jj*D + k + 5];
                const float b1_jx6 = X[jj*D + k + 6];
                const float b1_jx7 = X[jj*D + k + 7];
                const float b2_jx0 = X[(jj+1)*D + k];
                const float b2_jx1 = X[(jj+1)*D + k + 1];
                const float b2_jx2 = X[(jj+1)*D + k + 2];
                const float b2_jx3 = X[(jj+1)*D + k + 3];
                const float b2_jx4 = X[(jj+1)*D + k + 4];
                const float b2_jx5 = X[(jj+1)*D + k + 5];
                const float b2_jx6 = X[(jj+1)*D + k + 6];
                const float b2_jx7 = X[(jj+1)*D + k + 7];
                const float b3_jx0 = X[(jj+2)*D + k];
                const float b3_jx1 = X[(jj+2)*D + k + 1];
                const float b3_jx2 = X[(jj+2)*D + k + 2];
                const float b3_jx3 = X[(jj+2)*D + k + 3];
                const float b3_jx4 = X[(jj+2)*D + k + 4];
                const float b3_jx5 = X[(jj+2)*D + k + 5];
                const float b3_jx6 = X[(jj+2)*D + k + 6];
                const float b3_jx7 = X[(jj+2)*D + k + 7];
                const float b4_jx0 = X[(jj+3)*D + k];
                const float b4_jx1 = X[(jj+3)*D + k + 1];
                const float b4_jx2 = X[(jj+3)*D + k + 2];
                const float b4_jx3 = X[(jj+3)*D + k + 3];
                const float b4_jx4 = X[(jj+3)*D + k + 4];
                const float b4_jx5 = X[(jj+3)*D + k + 5];
                const float b4_jx6 = X[(jj+3)*D + k + 6];
                const float b4_jx7 = X[(jj+3)*D + k + 7];
                const float b5_jx0 = X[(jj+4)*D + k];
                const float b5_jx1 = X[(jj+4)*D + k + 1];
                const float b5_jx2 = X[(jj+4)*D + k + 2];
                const float b5_jx3 = X[(jj+4)*D + k + 3];
                const float b5_jx4 = X[(jj+4)*D + k + 4];
                const float b5_jx5 = X[(jj+4)*D + k + 5];
                const float b5_jx6 = X[(jj+4)*D + k + 6];
                const float b5_jx7 = X[(jj+4)*D + k + 7];
                const float b6_jx0 = X[(jj+5)*D + k];
                const float b6_jx1 = X[(jj+5)*D + k + 1];
                const float b6_jx2 = X[(jj+5)*D + k + 2];
                const float b6_jx3 = X[(jj+5)*D + k + 3];
                const float b6_jx4 = X[(jj+5)*D + k + 4];
                const float b6_jx5 = X[(jj+5)*D + k + 5];
                const float b6_jx6 = X[(jj+5)*D + k + 6];
                const float b6_jx7 = X[(jj+5)*D + k + 7];
                const float b7_jx0 = X[(jj+6)*D + k];
                const float b7_jx1 = X[(jj+6)*D + k + 1];
                const float b7_jx2 = X[(jj+6)*D + k + 2];
                const float b7_jx3 = X[(jj+6)*D + k + 3];
                const float b7_jx4 = X[(jj+6)*D + k + 4];
                const float b7_jx5 = X[(jj+6)*D + k + 5];
                const float b7_jx6 = X[(jj+6)*D + k + 6];
                const float b7_jx7 = X[(jj+6)*D + k + 7];
                const float b8_jx0 = X[(jj+7)*D + k];
                const float b8_jx1 = X[(jj+7)*D + k + 1];
                const float b8_jx2 = X[(jj+7)*D + k + 2];
                const float b8_jx3 = X[(jj+7)*D + k + 3];
                const float b8_jx4 = X[(jj+7)*D + k + 4];
                const float b8_jx5 = X[(jj+7)*D + k + 5];
                const float b8_jx6 = X[(jj+7)*D + k + 6];
                const float b8_jx7 = X[(jj+7)*D + k + 7];
                const float b1_d0 = x0 - b1_jx0;
                const float b1_d1 = x1 - b1_jx1;
                const float b1_d2 = x2 - b1_jx2;
                const float b1_d3 = x3 - b1_jx3;
                const float b1_d4 = x4 - b1_jx4;
                const float b1_d5 = x5 - b1_jx5;
                const float b1_d6 = x6 - b1_jx6;
                const float b1_d7 = x7 - b1_jx7;
                const float b2_d0 = x0 - b2_jx0;
                const float b2_d1 = x1 - b2_jx1;
                const float b2_d2 = x2 - b2_jx2;
                const float b2_d3 = x3 - b2_jx3;
                const float b2_d4 = x4 - b2_jx4;
                const float b2_d5 = x5 - b2_jx5;
                const float b2_d6 = x6 - b2_jx6;
                const float b2_d7 = x7 - b2_jx7;
                const float b3_d0 = x0 - b3_jx0;
                const float b3_d1 = x1 - b3_jx1;
                const float b3_d2 = x2 - b3_jx2;
                const float b3_d3 = x3 - b3_jx3;
                const float b3_d4 = x4 - b3_jx4;
                const float b3_d5 = x5 - b3_jx5;
                const float b3_d6 = x6 - b3_jx6;
                const float b3_d7 = x7 - b3_jx7;
                const float b4_d0 = x0 - b4_jx0;
                const float b4_d1 = x1 - b4_jx1;
                const float b4_d2 = x2 - b4_jx2;
                const float b4_d3 = x3 - b4_jx3;
                const float b4_d4 = x4 - b4_jx4;
                const float b4_d5 = x5 - b4_jx5;
                const float b4_d6 = x6 - b4_jx6;
                const float b4_d7 = x7 - b4_jx7;
                const float b5_d0 = x0 - b5_jx0;
                const float b5_d1 = x1 - b5_jx1;
                const float b5_d2 = x2 - b5_jx2;
                const float b5_d3 = x3 - b5_jx3;
                const float b5_d4 = x4 - b5_jx4;
                const float b5_d5 = x5 - b5_jx5;
                const float b5_d6 = x6 - b5_jx6;
                const float b5_d7 = x7 - b5_jx7;
                const float b6_d0 = x0 - b6_jx0;
                const float b6_d1 = x1 - b6_jx1;
                const float b6_d2 = x2 - b6_jx2;
                const float b6_d3 = x3 - b6_jx3;
                const float b6_d4 = x4 - b6_jx4;
                const float b6_d5 = x5 - b6_jx5;
                const float b6_d6 = x6 - b6_jx6;
                const float b6_d7 = x7 - b6_jx7;
                const float b7_d0 = x0 - b7_jx0;
                const float b7_d1 = x1 - b7_jx1;
                const float b7_d2 = x2 - b7_jx2;
                const float b7_d3 = x3 - b7_jx3;
                const float b7_d4 = x4 - b7_jx4;
                const float b7_d5 = x5 - b7_jx5;
                const float b7_d6 = x6 - b7_jx6;
                const float b7_d7 = x7 - b7_jx7;
                const float b8_d0 = x0 - b8_jx0;
                const float b8_d1 = x1 - b8_jx1;
                const float b8_d2 = x2 - b8_jx2;
                const float b8_d3 = x3 - b8_jx3;
                const float b8_d4 = x4 - b8_jx4;
                const float b8_d5 = x5 - b8_jx5;
                const float b8_d6 = x6 - b8_jx6;
                const float b8_d7 = x7 - b8_jx7;
                b1_accum0 += b1_d0*b1_d0;
                b1_accum1 += b1_d1*b1_d1;
                b1_accum2 += b1_d2*b1_d2;
                b1_accum3 += b1_d3*b1_d3;
                b1_accum4 += b1_d4*b1_d4;
                b1_accum5 += b1_d5*b1_d5;
                b1_accum6 += b1_d6*b1_d6;
                b1_accum7 += b1_d7*b1_d7;
                b2_accum0 += b2_d0*b2_d0;
                b2_accum1 += b2_d1*b2_d1;
                b2_accum2 += b2_d2*b2_d2;
                b2_accum3 += b2_d3*b2_d3;
                b2_accum4 += b2_d4*b2_d4;
                b2_accum5 += b2_d5*b2_d5;
                b2_accum6 += b2_d6*b2_d6;
                b2_accum7 += b2_d7*b2_d7;
                b3_accum0 += b3_d0*b3_d0;
                b3_accum1 += b3_d1*b3_d1;
                b3_accum2 += b3_d2*b3_d2;
                b3_accum3 += b3_d3*b3_d3;
                b3_accum4 += b3_d4*b3_d4;
                b3_accum5 += b3_d5*b3_d5;
                b3_accum6 += b3_d6*b3_d6;
                b3_accum7 += b3_d7*b3_d7;
                b4_accum0 += b4_d0*b4_d0;
                b4_accum1 += b4_d1*b4_d1;
                b4_accum2 += b4_d2*b4_d2;
                b4_accum3 += b4_d3*b4_d3;
                b4_accum4 += b4_d4*b4_d4;
                b4_accum5 += b4_d5*b4_d5;
                b4_accum6 += b4_d6*b4_d6;
                b4_accum7 += b4_d7*b4_d7;
                b5_accum0 += b5_d0*b5_d0;
                b5_accum1 += b5_d1*b5_d1;
                b5_accum2 += b5_d2*b5_d2;
                b5_accum3 += b5_d3*b5_d3;
                b5_accum4 += b5_d4*b5_d4;
                b5_accum5 += b5_d5*b5_d5;
                b5_accum6 += b5_d6*b5_d6;
                b5_accum7 += b5_d7*b5_d7;
                b6_accum0 += b6_d0*b6_d0;
                b6_accum1 += b6_d1*b6_d1;
                b6_accum2 += b6_d2*b6_d2;
                b6_accum3 += b6_d3*b6_d3;
                b6_accum4 += b6_d4*b6_d4;
                b6_accum5 += b6_d5*b6_d5;
                b6_accum6 += b6_d6*b6_d6;
                b6_accum7 += b6_d7*b6_d7;
                b7_accum0 += b7_d0*b7_d0;
                b7_accum1 += b7_d1*b7_d1;
                b7_accum2 += b7_d2*b7_d2;
                b7_accum3 += b7_d3*b7_d3;
                b7_accum4 += b7_d4*b7_d4;
                b7_accum5 += b7_d5*b7_d5;
                b7_accum6 += b7_d6*b7_d6;
                b7_accum7 += b7_d7*b7_d7;
                b8_accum0 += b8_d0*b8_d0;
                b8_accum1 += b8_d1*b8_d1;
                b8_accum2 += b8_d2*b8_d2;
                b8_accum3 += b8_d3*b8_d3;
                b8_accum4 += b8_d4*b8_d4;
                b8_accum5 += b8_d5*b8_d5;
                b8_accum6 += b8_d6*b8_d6;
                b8_accum7 += b8_d7*b8_d7;
            }

            const float b1_sum = b1_accum0 + b1_accum1 + b1_accum2 + b1_accum3 +
                                 b1_accum4 + b1_accum5 + b1_accum6 + b1_accum7;
            const float b1_res = b1_sum*factor;
            const float b2_sum = b2_accum0 + b2_accum1 + b2_accum2 + b2_accum3 +
                                 b2_accum4 + b2_accum5 + b2_accum6 + b2_accum7;
            const float b2_res = b2_sum*factor;
            const float b3_sum = b3_accum0 + b3_accum1 + b3_accum2 + b3_accum3 +
                                 b3_accum4 + b3_accum5 + b3_accum6 + b3_accum7;
            const float b3_res = b3_sum*factor;
            const float b4_sum = b4_accum0 + b4_accum1 + b4_accum2 + b4_accum3 +
                                 b4_accum4 + b4_accum5 + b4_accum6 + b4_accum7;
            const float b4_res = b4_sum*factor;
            const float b5_sum = b5_accum0 + b5_accum1 + b5_accum2 + b5_accum3 +
                                 b5_accum4 + b5_accum5 + b5_accum6 + b5_accum7;
            const float b5_res = b5_sum*factor;
            const float b6_sum = b6_accum0 + b6_accum1 + b6_accum2 + b6_accum3 +
                                 b6_accum4 + b6_accum5 + b6_accum6 + b6_accum7;
            const float b6_res = b6_sum*factor;
            const float b7_sum = b7_accum0 + b7_accum1 + b7_accum2 + b7_accum3 +
                                 b7_accum4 + b7_accum5 + b7_accum6 + b7_accum7;
            const float b7_res = b7_sum*factor;
            const float b8_sum = b8_accum0 + b8_accum1 + b8_accum2 + b8_accum3 +
                                 b8_accum4 + b8_accum5 + b8_accum6 + b8_accum7;
            const float b8_res = b8_sum*factor;
            DD[ii*N + jj] += b1_res;
            DD[ii*N + (jj + 1)] += b2_res;
            DD[ii*N + (jj + 2)] += b3_res;
            DD[ii*N + (jj + 3)] += b4_res;
            DD[ii*N + (jj + 4)] += b5_res;
            DD[ii*N + (jj + 5)] += b6_res;
            DD[ii*N + (jj + 6)] += b7_res;
            DD[ii*N + (jj + 7)] += b8_res;
            DD[jj*N + ii] += b1_res;
            DD[(jj + 1)*N + ii] += b2_res;
            DD[(jj + 2)*N + ii] += b3_res;
            DD[(jj + 3)*N + ii] += b4_res;
            DD[(jj + 4)*N + ii] += b5_res;
            DD[(jj + 5)*N + ii] += b6_res;
            DD[(jj + 6)*N + ii] += b7_res;
            DD[(jj + 7)*N + ii] += b8_res;
                }
            }
        }
    }

}

inline void fast_scalar_8x8x8_avx(float* X, int N, int D, float* DD, float* mean, int max_value) {

    const int leftover_start = D-D%8;
    const int leftover_start_4 = D-D%4;
    const int N_leftover_start = N-N%8;

    int nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for(int d = 0; d < D; d++) {
        mean[d] /= (float) N;
    }

    // Subtract data mean
    float max_X = .0;
    nD = 0;
    for(int n = 0; n < N; n++) {
        for(int d = 0; d < D; d++) {
            X[nD + d] -= mean[d];
            if(fabsf(X[nD + d]) > max_X) max_X = fabsf(X[nD + d]);
        }
        nD += D;
    }

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
                    const __m256 x = _mm256_loadu_ps(X + iiDk);
                    const __m256 xj = _mm256_loadu_ps(X + jjDk);
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
                    const __m256 x = _mm256_loadu_ps(X + iiD + k);
                    const __m256 b1_xj = _mm256_loadu_ps(X + jDk);
                    const __m256 b2_xj = _mm256_loadu_ps(X + jDk + D);
                    const __m256 b3_xj = _mm256_loadu_ps(X + jDk + 2*D);
                    const __m256 b4_xj = _mm256_loadu_ps(X + jDk + 3*D);
                    const __m256 b5_xj = _mm256_loadu_ps(X + jDk + 4*D);
                    const __m256 b6_xj = _mm256_loadu_ps(X + jDk + 5*D);
                    const __m256 b7_xj = _mm256_loadu_ps(X + jDk + 6*D);
                    const __m256 b8_xj = _mm256_loadu_ps(X + jDk + 7*D);
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

                _mm256_storeu_ps (DD + iiN + j, res);
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
                    const __m256 x = _mm256_loadu_ps(X + ii*D + k);
                    const __m256 b1_xj = _mm256_loadu_ps(X + jj*D + k);
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

inline void fast_scalar_8x8x8_avx_with_start(float* X, int N, int D, float* DD, float* mean, int max_value) {

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
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
            __m256 m2 = _mm256_loadu_ps(X + (n+1)*D + d);
            __m256 m3 = _mm256_loadu_ps(X + (n+2)*D + d);
            __m256 m4 = _mm256_loadu_ps(X + (n+3)*D + d);
            __m256 m5 = _mm256_loadu_ps(X + (n+4)*D + d);
            __m256 m6 = _mm256_loadu_ps(X + (n+5)*D + d);
            __m256 m7 = _mm256_loadu_ps(X + (n+6)*D + d);
            __m256 m8 = _mm256_loadu_ps(X + (n+7)*D + d);
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
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
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
        //_mm256_storeu_ps(mean + d, accum);

        for(int n = 0; n < N; n++) {
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
            m1 = _mm256_sub_ps(m1, accum);
            _mm256_storeu_ps(X + n* D + d, m1);
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
                    const __m256 x = _mm256_loadu_ps(X + iiDk);
                    const __m256 xj = _mm256_loadu_ps(X + jjDk);
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
                    const __m256 x = _mm256_loadu_ps(X + iiD + k);
                    const __m256 b1_xj = _mm256_loadu_ps(X + jDk);
                    const __m256 b2_xj = _mm256_loadu_ps(X + jDk + D);
                    const __m256 b3_xj = _mm256_loadu_ps(X + jDk + 2*D);
                    const __m256 b4_xj = _mm256_loadu_ps(X + jDk + 3*D);
                    const __m256 b5_xj = _mm256_loadu_ps(X + jDk + 4*D);
                    const __m256 b6_xj = _mm256_loadu_ps(X + jDk + 5*D);
                    const __m256 b7_xj = _mm256_loadu_ps(X + jDk + 6*D);
                    const __m256 b8_xj = _mm256_loadu_ps(X + jDk + 7*D);
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

                _mm256_storeu_ps (DD + iiN + j, res);
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
                    const __m256 x = _mm256_loadu_ps(X + ii*D + k);
                    const __m256 b1_xj = _mm256_loadu_ps(X + jj*D + k);
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

inline void fast_scalar_8x8x8_avx_with_start_more_block(float* X, int N, int D, float* DD, float* mean, int max_value) {

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
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
            __m256 m2 = _mm256_loadu_ps(X + (n+1)*D + d);
            __m256 m3 = _mm256_loadu_ps(X + (n+2)*D + d);
            __m256 m4 = _mm256_loadu_ps(X + (n+3)*D + d);
            __m256 m5 = _mm256_loadu_ps(X + (n+4)*D + d);
            __m256 m6 = _mm256_loadu_ps(X + (n+5)*D + d);
            __m256 m7 = _mm256_loadu_ps(X + (n+6)*D + d);
            __m256 m8 = _mm256_loadu_ps(X + (n+7)*D + d);
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
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
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
        //_mm256_storeu_ps(mean + d, accum);

        for(int n = 0; n < N; n++) {
            __m256 m1 = _mm256_loadu_ps(X + n*D + d);
            m1 = _mm256_sub_ps(m1, accum);
            _mm256_storeu_ps(X + n* D + d, m1);
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
                    const __m256 x = _mm256_loadu_ps(X + iiDk);
                    const __m256 xj = _mm256_loadu_ps(X + jjDk);
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
            __m256 r1;
            __m256 r2;
            __m256 r3;
            __m256 r4;
            __m256 r5;
            __m256 r6;
            __m256 r7;
            __m256 r8;
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
                    const __m256 x = _mm256_loadu_ps(X + iiD + k);
                    const __m256 b1_xj = _mm256_loadu_ps(X + jDk);
                    const __m256 b2_xj = _mm256_loadu_ps(X + jDk + D);
                    const __m256 b3_xj = _mm256_loadu_ps(X + jDk + 2*D);
                    const __m256 b4_xj = _mm256_loadu_ps(X + jDk + 3*D);
                    const __m256 b5_xj = _mm256_loadu_ps(X + jDk + 4*D);
                    const __m256 b6_xj = _mm256_loadu_ps(X + jDk + 5*D);
                    const __m256 b7_xj = _mm256_loadu_ps(X + jDk + 6*D);
                    const __m256 b8_xj = _mm256_loadu_ps(X + jDk + 7*D);
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

                switch(ii%8)
                {
                    case 0: r1 = res; break;
                    case 1: r2 = res; break;
                    case 2: r3 = res; break;
                    case 3: r4 = res; break;
                    case 4: r5 = res; break;
                    case 5: r6 = res; break;
                    case 6: r7 = res; break;
                    case 7: r8 = res; break;
                }

                _mm256_storeu_ps (DD + iiN + j, res);
                // float *res_ptr = (float *)&res;

                // DD[jNii] = res_ptr[0];
                // DD[jNii + N] = res_ptr[1];
                // DD[jNii + 2*N] = res_ptr[2];
                // DD[jNii + 3*N] = res_ptr[3];
                // DD[jNii + 4*N] = res_ptr[4];
                // DD[jNii + 5*N] = res_ptr[5];
                // DD[jNii + 6*N] = res_ptr[6];
                // DD[jNii + 7*N] = res_ptr[7];
                iiN += N;
                iiD += D;
                jNii++;
            }

            __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
            __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
            __t0 = _mm256_unpacklo_ps(r1, r2);
            __t1 = _mm256_unpackhi_ps(r1, r2);
            __t2 = _mm256_unpacklo_ps(r3, r4);
            __t3 = _mm256_unpackhi_ps(r3, r4);
            __t4 = _mm256_unpacklo_ps(r5, r6);
            __t5 = _mm256_unpackhi_ps(r5, r6);
            __t6 = _mm256_unpacklo_ps(r7, r8);
            __t7 = _mm256_unpackhi_ps(r7, r8);
            __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
            __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
            __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
            __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
            __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
            __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
            __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
            __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
            r1 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
            r2 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
            r3 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
            r4 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
            r5 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
            r6 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
            r7 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
            r8 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
            _mm256_storeu_ps (DD + jN + i, r1);
            _mm256_storeu_ps (DD + jN + N+ i, r2);
            _mm256_storeu_ps (DD + jN + 2*N+ i, r3);
            _mm256_storeu_ps (DD + jN + 3*N+ i, r4);
            _mm256_storeu_ps (DD + jN + 4*N+ i, r5);
            _mm256_storeu_ps (DD + jN + 5*N+ i, r6);
            _mm256_storeu_ps (DD + jN + 6*N+ i, r7);
            _mm256_storeu_ps (DD + jN + 7*N+ i, r8);

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
                    const __m256 x = _mm256_loadu_ps(X + ii*D + k);
                    const __m256 b1_xj = _mm256_loadu_ps(X + jj*D + k);
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


// inline void fast_scalar_8x8x8_avx_with_start_more_exact(float* X, int N, int D, float* DD, float* mean, int max_value) {

//     const int leftover_start = D-D%8;
//     const int N_leftover_start = N-N%8;

//     const __m256 sign_mask = _mm256_set1_ps(-0.f); // -0.f = 1 << 31

//     int nD = 0;
//     const float oneOverN = 1.0/N;
//     const __m256 oneOverN_avx = _mm256_set1_ps(1.0/N);
//     __m256 max = _mm256_setzero_ps();
//     float max_leftover = 0;
//     for(int d = 0; d < leftover_start; d+=8) {
//          __m256 accum1 = _mm256_setzero_ps();
//          __m256 accum2 = _mm256_setzero_ps();
//          __m256 accum3 = _mm256_setzero_ps();
//          __m256 accum4 = _mm256_setzero_ps();
//          __m256 accum5 = _mm256_setzero_ps();
//          __m256 accum6 = _mm256_setzero_ps();
//          __m256 accum7 = _mm256_setzero_ps();
//          __m256 accum8 = _mm256_setzero_ps();
//         for(int n = 0; n < N_leftover_start; n+=8) {
//             __m256 m1 = _mm256_loadu_ps(X + n*D + d);
//             __m256 m2 = _mm256_loadu_ps(X + (n+1)*D + d);
//             __m256 m3 = _mm256_loadu_ps(X + (n+2)*D + d);
//             __m256 m4 = _mm256_loadu_ps(X + (n+3)*D + d);
//             __m256 m5 = _mm256_loadu_ps(X + (n+4)*D + d);
//             __m256 m6 = _mm256_loadu_ps(X + (n+5)*D + d);
//             __m256 m7 = _mm256_loadu_ps(X + (n+6)*D + d);
//             __m256 m8 = _mm256_loadu_ps(X + (n+7)*D + d);
//             accum1 = _mm256_add_ps(accum1, m1);
//             accum2 = _mm256_add_ps(accum2, m2);
//             accum3 = _mm256_add_ps(accum3, m3);
//             accum4 = _mm256_add_ps(accum4, m4);
//             accum5 = _mm256_add_ps(accum5, m5);
//             accum6 = _mm256_add_ps(accum6, m6);
//             accum7 = _mm256_add_ps(accum7, m7);
//             accum8 = _mm256_add_ps(accum8, m8);
//         }
//         for(int n = N_leftover_start; n < N; n++) {
//             __m256 m1 = _mm256_loadu_ps(X + n*D + d);
//             accum1 = _mm256_add_ps(accum1, m1);
//         }
//         __m256 accum12 = _mm256_add_ps(accum1, accum2);
//         __m256 accum34 = _mm256_add_ps(accum3, accum4);
//         __m256 accum56 = _mm256_add_ps(accum5, accum6);
//         __m256 accum78 = _mm256_add_ps(accum7, accum8);
//         __m256 accum1234 = _mm256_add_ps(accum12, accum34);
//         __m256 accum5678 = _mm256_add_ps(accum56, accum78);
//         __m256 accum = _mm256_add_ps(accum1234, accum5678);
//         accum = _mm256_mul_ps(oneOverN_avx, accum);
//         _mm256_storeu_ps(mean + d, accum);

//         for(int n = 0; n < N; n++) {
//             __m256 m1 = _mm256_loadu_ps(X + n*D + d);
//             m1 = _mm256_sub_ps(m1, accum);
//             _mm256_storeu_ps(X + n* D + d, m1);
//             __m256 absx = _mm256_andnot_ps(sign_mask, m1);
//             max = _mm256_max_ps(absx, max);
//         }
//     }
//     for(int d = leftover_start; d < D; d++) {
//          float accum1 = 0;
//          float accum2 = 0;
//          float accum3 = 0;
//          float accum4 = 0;
//          float accum5 = 0;
//          float accum6 = 0;
//          float accum7 = 0;
//          float accum8 = 0;
//         for(int n = 0; n < N_leftover_start; n+=8) {
//             accum1 += X[n*D + d];
//             accum2 += X[(n+1)*D + d];
//             accum3 += X[(n+2)*D + d];
//             accum4 += X[(n+3)*D + d];
//             accum5 += X[(n+4)*D + d];
//             accum6 += X[(n+5)*D + d];
//             accum7 += X[(n+6)*D + d];
//             accum8 += X[(n+7)*D + d];
//         }
//         for(int n = N_leftover_start; n < N; n++) {
//             accum1 += X[n*D + d];
//         }
//         const float sum = accum1 + accum2 + accum3 + accum4 +
//                           accum5 + accum6 + accum7 + accum8;
//         const float res = oneOverN*sum;
//         mean[d] = res;
//         for(int n = 0; n < N; n++) {
//             X[n*D + d] -= mean[d];
//             max_leftover = fmaxf(max_leftover, fabsf(X[n*D + d]));
//         }
//     }
//     __m128 max_lower = _mm256_extractf128_ps(max, 1);
//     __m256 max_lower_256 = _mm256_castps128_ps256 (max_lower);
//     __m256 max4 = _mm256_max_ps(max, max_lower_256);
//     float* mm = (float*)&max4;
//     float max_X = fmaxf(fmaxf(fmaxf(mm[0], mm[1]), fmaxf(mm[2], mm[3])), max_leftover);
//     const float factor = 1.0/max_X;
//     const __m256 factor_avx = _mm256_set1_ps(factor);


//     int iN = 0;
//     int iD = 0;
//     for (int i = 0; i < N_leftover_start; i+=8)
//     {
//         int iiN = iN;
//         int iiD = iD;
//         for(int ii = i; ii < i+8; ii++)
//         {
//             DD[iiN + ii] = 0;
//             int jjD = iiD+D;
//             int jjN = iiN+N;
//             for (int jj = ii+1; jj < i+8; jj++)
//             {
//                 __m256 accum = _mm256_setzero_ps();
//                 int iiDk = iiD;
//                 int jjDk = jjD;
//                 for(int k = 0; k < leftover_start; k+=8)
//                 {
//                     const __m256 x = _mm256_loadu_ps(X + iiDk);
//                     const __m256 xj = _mm256_loadu_ps(X + jjDk);                    
//                     const __m256 sx = _mm256_mul_ps(x, factor_avx);
//                     const __m256 sxj = _mm256_mul_ps(xj, factor_avx);
//                     const __m256 d = _mm256_sub_ps(sx, sxj);
//                     accum = _mm256_fmadd_ps(d, d, accum);
//                     iiDk += 8;
//                     jjDk += 8;
//                 }
//                 float accum_leftover = 0.0f;
//                 for(int k = leftover_start; k < D; k++)
//                 {
//                     const float d0 = factor*X[iiD + k] - factor*X[jjD + k];
//                     accum_leftover += d0*d0;
//                 }
//                 __m256 hsum = _mm256_hadd_ps(accum, accum);
//                 float *hsumptr = (float *)&hsum;
//                 const float res = hsumptr[0] + hsumptr[1] + hsumptr[4] + hsumptr[5] + accum_leftover;
//                 DD[iiN + jj] = res;
//                 DD[jjN + ii] = res;
//                 jjD += D;
//                 jjN += N;
//             }
//             iiN += N;
//             iiD += D;
//         }

//         int jD = iD+8*D;
//         int jN = iN+8*N;
//         for (int j = i+8; j < N_leftover_start; j+=8)
//         {            
//             iiN = iN;
//             iiD = iD;
//             int jNii = jN + i;
//             for(int ii = i; ii < i+8; ii++)
//             {
//                 __m256 b1_accum = _mm256_setzero_ps();
//                 __m256 b2_accum = _mm256_setzero_ps();
//                 __m256 b3_accum = _mm256_setzero_ps();
//                 __m256 b4_accum = _mm256_setzero_ps();
//                 __m256 b5_accum = _mm256_setzero_ps();
//                 __m256 b6_accum = _mm256_setzero_ps();
//                 __m256 b7_accum = _mm256_setzero_ps();
//                 __m256 b8_accum = _mm256_setzero_ps();
//                 int jDk = jD;
//                 for(int k = 0; k < leftover_start; k+=8)
//                 {
//                     const __m256 x = _mm256_loadu_ps(X + iiD + k);
//                     const __m256 sx = _mm256_mul_ps(x, factor_avx);
//                     const __m256 b1_xj = _mm256_loadu_ps(X + jDk);
//                     const __m256 b2_xj = _mm256_loadu_ps(X + jDk + D);
//                     const __m256 b3_xj = _mm256_loadu_ps(X + jDk + 2*D);
//                     const __m256 b4_xj = _mm256_loadu_ps(X + jDk + 3*D);
//                     const __m256 b5_xj = _mm256_loadu_ps(X + jDk + 4*D);
//                     const __m256 b6_xj = _mm256_loadu_ps(X + jDk + 5*D);
//                     const __m256 b7_xj = _mm256_loadu_ps(X + jDk + 6*D);
//                     const __m256 b8_xj = _mm256_loadu_ps(X + jDk + 7*D);
//                     const __m256 b1_sxj = _mm256_mul_ps(b1_xj, factor_avx);
//                     const __m256 b2_sxj = _mm256_mul_ps(b2_xj, factor_avx);
//                     const __m256 b3_sxj = _mm256_mul_ps(b3_xj, factor_avx);
//                     const __m256 b4_sxj = _mm256_mul_ps(b4_xj, factor_avx);
//                     const __m256 b5_sxj = _mm256_mul_ps(b5_xj, factor_avx);
//                     const __m256 b6_sxj = _mm256_mul_ps(b6_xj, factor_avx);
//                     const __m256 b7_sxj = _mm256_mul_ps(b7_xj, factor_avx);
//                     const __m256 b8_sxj = _mm256_mul_ps(b8_xj, factor_avx);
//                     const __m256 b1_d = _mm256_sub_ps(sx, b1_sxj);
//                     const __m256 b2_d = _mm256_sub_ps(sx, b2_sxj);
//                     const __m256 b3_d = _mm256_sub_ps(sx, b3_sxj);
//                     const __m256 b4_d = _mm256_sub_ps(sx, b4_sxj);
//                     const __m256 b5_d = _mm256_sub_ps(sx, b5_sxj);
//                     const __m256 b6_d = _mm256_sub_ps(sx, b6_sxj);
//                     const __m256 b7_d = _mm256_sub_ps(sx, b7_sxj);
//                     const __m256 b8_d = _mm256_sub_ps(sx, b8_sxj);
//                     b1_accum = _mm256_fmadd_ps(b1_d, b1_d, b1_accum);
//                     b2_accum = _mm256_fmadd_ps(b2_d, b2_d, b2_accum);
//                     b3_accum = _mm256_fmadd_ps(b3_d, b3_d, b3_accum);
//                     b4_accum = _mm256_fmadd_ps(b4_d, b4_d, b4_accum);
//                     b5_accum = _mm256_fmadd_ps(b5_d, b5_d, b5_accum);
//                     b6_accum = _mm256_fmadd_ps(b6_d, b6_d, b6_accum);
//                     b7_accum = _mm256_fmadd_ps(b7_d, b7_d, b7_accum);
//                     b8_accum = _mm256_fmadd_ps(b8_d, b8_d, b8_accum);
//                     jDk += 8;
//                 }
//                 float b1_accum_leftover = 0.0f;
//                 float b2_accum_leftover = 0.0f;
//                 float b3_accum_leftover = 0.0f;
//                 float b4_accum_leftover = 0.0f;
//                 float b5_accum_leftover = 0.0f;
//                 float b6_accum_leftover = 0.0f;
//                 float b7_accum_leftover = 0.0f;
//                 float b8_accum_leftover = 0.0f;
//                 jDk = jD + leftover_start;
//                 for(int k = leftover_start; k < D; k++)
//                 {
//                     const float x = factor*X[iiD + k];
//                     const float b1_xj = factor*X[jDk];
//                     const float b2_xj = factor*X[jDk + D];
//                     const float b3_xj = factor*X[jDk + 2*D];
//                     const float b4_xj = factor*X[jDk + 3*D];
//                     const float b5_xj = factor*X[jDk + 4*D];
//                     const float b6_xj = factor*X[jDk + 5*D];
//                     const float b7_xj = factor*X[jDk + 6*D];
//                     const float b8_xj = factor*X[jDk + 7*D];
//                     const float b1_d = x - b1_xj;
//                     const float b2_d = x - b2_xj;
//                     const float b3_d = x - b3_xj;
//                     const float b4_d = x - b4_xj;
//                     const float b5_d = x - b5_xj;
//                     const float b6_d = x - b6_xj;
//                     const float b7_d = x - b7_xj;
//                     const float b8_d = x - b8_xj;
//                     b1_accum_leftover += b1_d*b1_d;
//                     b2_accum_leftover += b2_d*b2_d;
//                     b3_accum_leftover += b3_d*b3_d;
//                     b4_accum_leftover += b4_d*b4_d;
//                     b5_accum_leftover += b5_d*b5_d;
//                     b6_accum_leftover += b6_d*b6_d;
//                     b7_accum_leftover += b7_d*b7_d;
//                     b8_accum_leftover += b8_d*b8_d;
//                     jDk++;
//                 }
//                 //12345678
//                 const __m256 leftover =  _mm256_set_ps (b8_accum_leftover, b7_accum_leftover, b6_accum_leftover, b5_accum_leftover, b4_accum_leftover, b3_accum_leftover, b2_accum_leftover, b1_accum_leftover);

//                 //11221122
//                 const __m256 hsum12 = _mm256_hadd_ps(b1_accum, b2_accum);
//                 //33443344
//                 const __m256 hsum34 = _mm256_hadd_ps(b3_accum, b4_accum);
//                 //55665566
//                 const __m256 hsum56 = _mm256_hadd_ps(b5_accum, b6_accum);
//                 //77887788
//                 const __m256 hsum78 = _mm256_hadd_ps(b7_accum, b8_accum);

//                 //12341234
//                 const __m256 hsum1234 = _mm256_hadd_ps(hsum12, hsum34);
//                 //56785678
//                 const __m256 hsum5678 = _mm256_hadd_ps(hsum56, hsum78);

//                 const __m256 hsum12345678a = _mm256_permute2f128_ps (hsum1234, hsum5678, 0x20);
//                 const __m256 hsum12345678b = _mm256_permute2f128_ps (hsum1234, hsum5678, 0x31);
//                 const __m256 hsum = _mm256_add_ps (hsum12345678a, hsum12345678b);
//                 const __m256 res = _mm256_add_ps (leftover, hsum);
//                 // const __m256 res = _mm256_mul_ps (factor_avx, sum);

//                 _mm256_storeu_ps (DD + iiN + j, res);
//                 float *res_ptr = (float *)&res;

//                 DD[jNii] = res_ptr[0];
//                 DD[jNii + N] = res_ptr[1];
//                 DD[jNii + 2*N] = res_ptr[2];
//                 DD[jNii + 3*N] = res_ptr[3];
//                 DD[jNii + 4*N] = res_ptr[4];
//                 DD[jNii + 5*N] = res_ptr[5];
//                 DD[jNii + 6*N] = res_ptr[6];
//                 DD[jNii + 7*N] = res_ptr[7];
//                 iiN += N;
//                 iiD += D;
//                 jNii++;
//             }
//             jD += 8*D;
//             jN += 8*N;
//         }
//         for (int j = N_leftover_start; j < N; j++)
//         {
//             for(int ii = i; ii < i+8; ii++)
//             {
//                 int jj = j;
//                 __m256 b1_accum = _mm256_setzero_ps();
//                 for(int k = 0; k < leftover_start; k+=8)
//                 {
//                     const __m256 x = _mm256_loadu_ps(X + ii*D + k);
//                     const __m256 sx = _mm256_mul_ps(x, factor_avx);
//                     const __m256 b1_xj = _mm256_loadu_ps(X + jj*D + k);
//                     const __m256 b1_sxj = _mm256_mul_ps(b1_xj, factor_avx);
//                     const __m256 b1_d = _mm256_sub_ps(sx, b1_sxj);
//                     b1_accum = _mm256_fmadd_ps(b1_d, b1_d, b1_accum);
//                 }
//                 float b1_accum_leftover = 0.0f;
//                 for(int k = leftover_start; k < D; k++)
//                 {
//                     const float x = factor*X[ii*D + k];
//                     const float b1_xj = factor*X[jj*D + k];
//                     const float b1_d = x - b1_xj;
//                     b1_accum_leftover += b1_d*b1_d;
//                 }
//                 b1_accum = _mm256_hadd_ps(b1_accum, b1_accum);

//                 const float sum0 = ((float *)&b1_accum)[0] + ((float *)&b1_accum)[1] +
//                                    ((float *)&b1_accum)[4] + ((float *)&b1_accum)[5]; 
//                 const float res = sum0 + b1_accum_leftover;
//                 // const float res = sum*factor;
//                 DD[ii*N + jj] = res;
//                 DD[jj*N + ii] = res;
//             }
//         }
//         iN += 8*N;
//         iD += 8*D;
//     }

// }

// inline void fast_scalar_unroll_first(float* X, int N, int D, float* DD, float* mean, int max_value) {

//     const int leftover_start = D-D%8;
//     const int leftover_start_N = N-N%8;

//     for(int n = 0; n < leftover_start_N; n+=8) {
//         const int nD0 = n*D;
//         const int nD1 = nD0 + D;
//         const int nD2 = nD0 + 2*D;
//         const int nD3 = nD0 + 3*D;
//         const int nD4 = nD0 + 4*D;
//         const int nD5 = nD0 + 5*D;
//         const int nD6 = nD0 + 6*D;
//         const int nD7 = nD0 + 7*D;
//         for(int d = 0; d < leftover_start; d+=8) {
//             const int d1 = d+1;
//             const int d2 = d+2;
//             const int d3 = d+3;
//             const int d4 = d+4;
//             const int d5 = d+5;
//             const int d6 = d+6;
//             const int d7 = d+7;
//             mean[d] += X[nD0 + d] + X[nD1 + d] + X[nD2 + d] + X[nD3 + d] +
//                     X[nD4 + d] + X[nD5 + d] + X[nD6 + d] + X[nD7 + d];
//             mean[d1] += X[nD0 + d1] + X[nD1 + d1] + X[nD2 + d1] + X[nD3 + d1] +
//                     X[nD4 + d1] + X[nD5 + d1] + X[nD6 + d1] + X[nD7 + d1];        
//             mean[d2] += X[nD0 + d2] + X[nD1 + d2] + X[nD2 + d2] + X[nD3 + d2] +
//                     X[nD4 + d2] + X[nD5 + d2] + X[nD6 + d2] + X[nD7 + d2];
//             mean[d3] += X[nD0 + d3] + X[nD1 + d3] + X[nD2 + d3] + X[nD3 + d3] +
//                     X[nD4 + d3] + X[nD5 + d3] + X[nD6 + d3] + X[nD7 + d3];        
//             mean[d4] += X[nD0 + d4] + X[nD1 + d4] + X[nD2 + d4] + X[nD3 + d4] +
//                     X[nD4 + d4] + X[nD5 + d4] + X[nD6 + d4] + X[nD7 + d4];
//             mean[d5] += X[nD0 + d5] + X[nD1 + d5] + X[nD2 + d5] + X[nD3 + d5] +
//                     X[nD4 + d5] + X[nD5 + d5] + X[nD6 + d5] + X[nD7 + d5];        
//             mean[d6] += X[nD0 + d6] + X[nD1 + d6] + X[nD2 + d6] + X[nD3 + d6] +
//                     X[nD4 + d6] + X[nD5 + d6] + X[nD6 + d6] + X[nD7 + d6];
//             mean[d7] += X[nD0 + d7] + X[nD1 + d7] + X[nD2 + d7] + X[nD3 + d7] +
//                     X[nD4 + d7] + X[nD5 + d7] + X[nD6 + d7] + X[nD7 + d7];        
//         }
//         for(int d = leftover_start; d < D; d++) {
//             mean[d] += X[nD0 + d] + X[nD1 + d] + X[nD2 + d] + X[nD3 + d] +
//                     X[nD4 + d] + X[nD5 + d] + X[nD6 + d] + X[nD7 + d];
//         }
//     }
//     for(int n = leftover_start_N; n < N; n++) {
//         const int nD = n*D;
//         for(int d = 0; d < D; d++) {
//             mean[d] += X[nD + d] ;
//         }
//     }


//     for(int d = 0; d < D; d++) {
//         mean[d] /= (float) N;
//     }

//     // Subtract data mean
//     float max_X = .0;
//     int nD = 0;
//     for(int n = 0; n < N; n++) {
//         for(int d = 0; d < D; d++) {
//             X[nD + d] -= mean[d];
//             if(fabsf(X[nD + d]) > max_X) max_X = fabsf(X[nD + d]);
//         }
//         nD += D;
//     }
//     // std::cout << max_X << std::endl;
//     const float factor = 1.0/(max_X*max_X);

//     int iN = 0; int iD = 0;
//     for (int i = 0; i < N; ++i)
//     {
//         DD[iN + i] = 0;
//         int jD = iD + D;
//         int jN = iN + N;
//         for (int j = i+1; j < N; ++j)
//         {
//             const int iNj = iN + j;
//             // DD[iNj] = 0;
//             float accum0 = 0;
//             float accum1 = 0;
//             float accum2 = 0;
//             float accum3 = 0;
//             float accum4 = 0;
//             float accum5 = 0;
//             float accum6 = 0;
//             float accum7 = 0;

//             for(int d = 0; d < leftover_start; d+=8)
//             {
//                 const int iDd = iD + d;
//                 const int jDd = jD + d;
//                 const float dist0 = X[iDd    ] - X[jDd    ];
//                 const float dist1 = X[iDd+1] - X[jDd+1];
//                 const float dist2 = X[iDd+2] - X[jDd+2];
//                 const float dist3 = X[iDd+3] - X[jDd+3];
//                 const float dist4 = X[iDd+4] - X[jDd+4];
//                 const float dist5 = X[iDd+5] - X[jDd+5];
//                 const float dist6 = X[iDd+6] - X[jDd+6];
//                 const float dist7 = X[iDd+7] - X[jDd+7];
//                 const float prod0 = dist0 * dist0;
//                 const float prod1 = dist1 * dist1;
//                 const float prod2 = dist2 * dist2;
//                 const float prod3 = dist3 * dist3;
//                 const float prod4 = dist4 * dist4;
//                 const float prod5 = dist5 * dist5;
//                 const float prod6 = dist6 * dist6;
//                 const float prod7 = dist7 * dist7;
//                 accum0 += prod0;
//                 accum1 += prod1;
//                 accum2 += prod2;
//                 accum3 += prod3;
//                 accum4 += prod4;
//                 accum5 += prod5;
//                 accum6 += prod6;
//                 accum7 += prod7;
//                 // std::cout << d << std::endl;
//             }
            
//             const float sum0 = accum1 + accum2 + accum3 + accum4 + accum5 + accum6 + accum7; 
//             for(int d = leftover_start; d < D; d++)
//             {
//                 // std::cout << d << std::endl;
//                 const float dist0 = X[iD + d] - X[jD + d];
//                 const float prod0 = dist0 * dist0;
//                 accum0 += prod0;
//             }
//             const float sum1 = accum0 + sum0;
//             const float dist = sum1 * factor;
//             DD[iNj] = dist;
//             DD[jN + i] = dist;
//             jD += D;
//             jN += N;
//         }
//         iN += N;
//         iD += D;
//     }
// }
#endif
