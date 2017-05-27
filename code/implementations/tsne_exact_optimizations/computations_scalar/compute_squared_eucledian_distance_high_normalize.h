#ifndef COMPUTE_SQUARED_EUCLEDIAN_DISTANCE_HIGH_NORMALIZE_H
#define COMPUTE_SQUARED_EUCLEDIAN_DISTANCE_HIGH_NORMALIZE_H

#include <x86intrin.h>

inline void compute_squared_eucledian_distance_high_normalize(float* X, int N, int D, float* DD, float* mean, int max_value) {

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

#endif
