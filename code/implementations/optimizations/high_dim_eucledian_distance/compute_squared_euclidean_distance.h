#ifndef COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H
#define COMPUTE_SQUARED_EUCLIDEAN_DISTANCE_H

#include "../../utils/data_type.h"
#include <iostream>
#include <x86intrin.h>
// Compute squared euclidean disctance for all pairs of vectors X_i X_j
inline void base_version(float* X, int N, int D, float* DD, float* mean, int max_value) {

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


// inline void fast_scalar(float* X, int N, int D, float* DD, float* mean, int max_value) {

//     int nD = 0;
//     float *mins = (float *) calloc(D, sizeof(float));
//     float *maxs = (float *) calloc(D, sizeof(float));
//     int *minsI = (int *) calloc(D, sizeof(int));
//     int *maxsI = (int *) calloc(D, sizeof(int));

//     for(int n = 0; n < N; n++) {
//         for(int d = 0; d < D; d++) {
//             const float x = X[nD + d];
//             mean[d] += x;
//             if(x < mins[d])
//             {
//                 mins[d] = x;
//                 minsI[d] = n;
//             }
//             if(x > maxs[d])
//             {
//                 maxs[d] = x;
//                 maxsI[d] = n;
//             }
//         }
//         nD += D;
//     }
//     float max_X = 0;
//     int maxd = 0;
//     for(int d = 0; d < D; d++) {
//         mean[d] /= (float) N;
//         const float m = fmaxf(maxs[d] - mean[d], fabsf(mins[d]- mean[d]));
//         // std::cout << m << " " << maxs[d] - mean[d] << " " << -mins[d] + mean[d] << std::endl;
//         if (m > max_X)
//         {
//             max_X = m;
//             maxd = d;
//         }

//     }
//     std::cout << maxd << " " << maxsI[maxd] << " " << X[maxsI[maxd]*D + maxd]  << std::endl;

//     nD = 0;
//     for(int n = 0; n < N; n++) {
//         for(int d = 0; d < D; d++) {
//             X[nD + d] -= mean[d];
//         }
//         nD += D;
//     }
//     float max2 = .0;
//     int maxi = 0;
//     for(int i = 0; i < N * D; i++) {
//         if(fabsf(X[i]) > max2){max2 = fabsf(X[i]); maxi = i;}
//     }
//     std::cout << maxd << " " << maxsI[maxd] << " " << X[maxsI[maxd]*D + maxd] << " " << X[maxi] << std::endl;
//     std::cout << max_X << " " << max2 << " "<< maxi << " " << maxi%N << " " << maxi-(maxi%N) << " " << maxd << std::endl;

//     free(mins); free(maxs);
//     // std::cout << max_X << std::endl;
//     for(int d = 0; d < D; d++)
//         X[d] = (X[d] - mean[d])/max_X;
//     int iN = 0; int iD = 0;
//     for (int i = 0; i < N; ++i)
//     {
//         DD[iN + i] = 0;
//         int jD = iD + D;
//         int jN = iN + N;
//         for (int j = i+1; j < N; ++j)
//         {
//             const int iNj = iN + j;
//             DD[iNj] = 0;
//             for(int d = 0; d < D; d++)
//             {
//                 if(i == 0)
//                     X[jD + d] = (X[jD + d] - mean[d])/max_X;
//                 const float dist = X[iD + d] - X[jD + d];
//                 DD[iNj] += dist*dist;
//             }
//             DD[jN + i] = DD[iNj];
//             jD += D;
//             jN += N;
//         }
//         iN += N;
//         iD += D;
//     }
// }

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


inline void fast_scalar_4x4_base(float* X, int N, int D, float* DD, float* mean, int max_value) {

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
