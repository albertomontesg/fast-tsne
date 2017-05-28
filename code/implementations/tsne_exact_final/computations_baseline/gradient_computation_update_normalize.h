#ifndef GRADIENT_COMPUTATION_UPDATE_NORMALIZE_H
#define GRADIENT_COMPUTATION_UPDATE_NORMALIZE_H

#include <stdio.h>
#include <immintrin.h>

inline void gradient_computation_update_normalize(float* Y, float* P, float* Q,
                                                  float sum_Q, int N,
			                                      int D, float* uY,
                                                  float momentum, float eta) {
    // Perform the computation of the gradient
    int nN = 0;
    int nD = 0;
    for(int n = 0; n < N; n++) {
    	int mD = 0;
        float dC_0 = 0;
        float dC_1 = 0;

    	for(int m = 0; m < N; m++) {
			float mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
            dC_0 += (Y[nD]-Y[mD]) * mult;
            dC_1 += (Y[nD+1]-Y[mD+1]) * mult;
			// for (int d = 0; d < D; d++) {
			// 	dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
			// }
    		mD += D;
    	}

        uY[nD] = momentum * uY[nD] - eta * dC_0;
        uY[nD+1] = momentum * uY[nD+1] - eta * dC_1;

    	nN += N;
    	nD += D;
    }
    float mean0 = 0;
	float mean1 = 0;
    nD = 0;
    for (int n = 0; n < N; n++, nD += D) {
        Y[nD] = Y[nD] +  uY[nD];
        Y[nD+1] = Y[nD+1] +  uY[nD+1];
        mean0 += Y[nD];
        mean1 += Y[nD+1];
    }

    mean0 /= (double) N;
    mean1 /= (double) N;

    // Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++, nD += D) {
		Y[nD] -= mean0;
		Y[nD+1] -= mean1;
	}
 }

#endif
