#ifndef NORMALIZE_H
#define NORMALIZE_H

#include <stdio.h>
#include <immintrin.h>

typedef void(*comp_func)(float *, int, int);

inline void unroll_accum8_vec(float* X, int N, int D) {
	int nD, n;
	const int V = 4; // samples that fit in a AVX register
	const int K = 8;

	__m256 _mean = _mm256_setzero_ps();
	__m256 _mean_0 = _mm256_setzero_ps();
	__m256 _mean_1 = _mm256_setzero_ps();
	__m256 _mean_2 = _mm256_setzero_ps();
	__m256 _mean_3 = _mm256_setzero_ps();
	__m256 _mean_4 = _mm256_setzero_ps();
	__m256 _mean_5 = _mm256_setzero_ps();
	__m256 _mean_6 = _mm256_setzero_ps();
	__m256 _mean_7 = _mm256_setzero_ps();

	__m256 inv_N = _mm256_set1_ps(1 / (double) N);


	for (n = 0, nD = 0; n + V * K <= N; n += V*K, nD += V*K*D) {
		__m256 _X_0 = _mm256_loadu_ps(X+nD);
		__m256 _X_1 = _mm256_loadu_ps(X+nD+V*D);
		__m256 _X_2 = _mm256_loadu_ps(X+nD+2*V*D);
		__m256 _X_3 = _mm256_loadu_ps(X+nD+3*V*D);
		__m256 _X_4 = _mm256_loadu_ps(X+nD+4*V*D);
		__m256 _X_5 = _mm256_loadu_ps(X+nD+5*V*D);
		__m256 _X_6 = _mm256_loadu_ps(X+nD+6*V*D);
		__m256 _X_7 = _mm256_loadu_ps(X+nD+7*V*D);

		_mean_0 = _mm256_add_ps(_mean_0, _X_0);
		_mean_1 = _mm256_add_ps(_mean_1, _X_1);
		_mean_2 = _mm256_add_ps(_mean_2, _X_2);
		_mean_3 = _mm256_add_ps(_mean_3, _X_3);
		_mean_4 = _mm256_add_ps(_mean_4, _X_4);
		_mean_5 = _mm256_add_ps(_mean_5, _X_5);
		_mean_6 = _mm256_add_ps(_mean_6, _X_6);
		_mean_7 = _mm256_add_ps(_mean_7, _X_7);

	}
	for (; n + V <= N; n += V, nD += V*D) {
		__m256 _X = _mm256_loadu_ps(X+nD);
		_mean = _mm256_add_ps(_mean, _X);
	}
	for(; n < N; n++, nD += D) {
		_mean[0] += X[nD];
		_mean[1] += X[nD+1];
	}

	__m256 _mean_8 = _mm256_add_ps(_mean_0, _mean_1);
	__m256 _mean_9 = _mm256_add_ps(_mean_2, _mean_3);
	__m256 _mean_10 = _mm256_add_ps(_mean_4, _mean_5);
	__m256 _mean_11 = _mm256_add_ps(_mean_6, _mean_7);
	__m256 _mean_12 = _mm256_add_ps(_mean_8, _mean_9);
	__m256 _mean_13 = _mm256_add_ps(_mean_10, _mean_11);

	__m256 _mean_ = _mm256_add_ps(_mean_12, _mean_13);
	_mean_ = _mm256_add_ps(_mean, _mean_); // Sum the remaining samples

	__m256 _mean_permute = _mm256_permute2f128_ps(_mean_, _mean_, 1);
	__m256 _mean__ = _mm256_add_ps(_mean_, _mean_permute);

	__m256 _mean__shuffle = _mm256_shuffle_ps(_mean__, _mean__, 78);
	_mean = _mm256_add_ps(_mean__, _mean__shuffle);

	_mean = _mm256_mul_ps(_mean, inv_N);

	// Subtract data mean
	for (n = 0, nD = 0; n + V * K <= N; n += V*K, nD += V*K*D) {
		__m256 _X_0 = _mm256_loadu_ps(X+nD);
		__m256 _X_1 = _mm256_loadu_ps(X+nD+V*D);
		__m256 _X_2 = _mm256_loadu_ps(X+nD+2*V*D);
		__m256 _X_3 = _mm256_loadu_ps(X+nD+3*V*D);
		__m256 _X_4 = _mm256_loadu_ps(X+nD+4*V*D);
		__m256 _X_5 = _mm256_loadu_ps(X+nD+5*V*D);
		__m256 _X_6 = _mm256_loadu_ps(X+nD+6*V*D);
		__m256 _X_7 = _mm256_loadu_ps(X+nD+7*V*D);

		_X_0 = _mm256_sub_ps(_X_0, _mean);
		_X_1 = _mm256_sub_ps(_X_1, _mean);
		_X_2 = _mm256_sub_ps(_X_2, _mean);
		_X_3 = _mm256_sub_ps(_X_3, _mean);
		_X_4 = _mm256_sub_ps(_X_4, _mean);
		_X_5 = _mm256_sub_ps(_X_5, _mean);
		_X_6 = _mm256_sub_ps(_X_6, _mean);
		_X_7 = _mm256_sub_ps(_X_7, _mean);

		_mm256_storeu_ps(X+nD, _X_0);
		_mm256_storeu_ps(X+nD+V*D, _X_1);
		_mm256_storeu_ps(X+nD+2*V*D, _X_2);
		_mm256_storeu_ps(X+nD+3*V*D, _X_3);
		_mm256_storeu_ps(X+nD+4*V*D, _X_4);
		_mm256_storeu_ps(X+nD+5*V*D, _X_5);
		_mm256_storeu_ps(X+nD+6*V*D, _X_6);
		_mm256_storeu_ps(X+nD+7*V*D, _X_7);

	}
	for (; n + V <= N; n += V, nD += V*D) {
		__m256 _X = _mm256_loadu_ps(X+nD);
		_X = _mm256_sub_ps(_X, _mean);
		_mm256_storeu_ps(X+nD, _X);
	}
	for(; n < N; n++, nD += D) {
		X[nD] -= _mean[0];
		X[nD+1] -= _mean[1];
	}
}


inline void unroll_accum8(float* X, int N, int D) {
	int nD, n;
	const int K = 8;

	float mean0_0 = 0;
	float mean0_1 = 0;
	float mean0_2 = 0;
	float mean0_3 = 0;
	float mean0_4 = 0;
	float mean0_5 = 0;
	float mean0_6 = 0;
	float mean0_7 = 0;
	float mean1_0 = 0;
	float mean1_1 = 0;
	float mean1_2 = 0;
	float mean1_3 = 0;
	float mean1_4 = 0;
	float mean1_5 = 0;
	float mean1_6 = 0;
	float mean1_7 = 0;

	float mean0 = 0;
	float mean1 = 0;

	// With 8 accumulators
	for (n = 0, nD = 0; n + K <= N; n += K, nD += K*D) {
		float X0_0 = X[nD];
		float X1_0 = X[nD+1];
		float X0_1 = X[nD+D];
		float X1_1 = X[nD+D+1];
		float X0_2 = X[nD+2*D];
		float X1_2 = X[nD+2*D+1];
		float X0_3 = X[nD+3*D];
		float X1_3 = X[nD+3*D+1];
		float X0_4 = X[nD+4*D];
		float X1_4 = X[nD+4*D+1];
		float X0_5 = X[nD+5*D];
		float X1_5 = X[nD+5*D+1];
		float X0_6 = X[nD+6*D];
		float X1_6 = X[nD+6*D+1];
		float X0_7 = X[nD+7*D];
		float X1_7 = X[nD+7*D+1];

		mean0_0 += X0_0;
		mean1_0 += X1_0;
		mean0_1 += X0_1;
		mean1_1 += X1_1;
		mean0_2 += X0_2;
		mean1_2 += X1_2;
		mean0_3 += X0_3;
		mean1_3 += X1_3;
		mean0_4 += X0_4;
		mean1_4 += X1_4;
		mean0_5 += X0_5;
		mean1_5 += X1_5;
		mean0_6 += X0_6;
		mean1_6 += X1_6;
		mean0_7 += X0_7;
		mean1_7 += X1_7;
	}
	// With 4 accumulators
	for (; n + K/2 <= N; n += K/2, nD += K/2*D) {
		float X0_0 = X[nD];
		float X1_0 = X[nD+1];
		float X0_1 = X[nD+D];
		float X1_1 = X[nD+D+1];
		float X0_2 = X[nD+2*D];
		float X1_2 = X[nD+2*D+1];
		float X0_3 = X[nD+3*D];
		float X1_3 = X[nD+3*D+1];

		mean0_0 += X0_0;
		mean1_0 += X1_0;
		mean0_1 += X0_1;
		mean1_1 += X1_1;
		mean0_2 += X0_2;
		mean1_2 += X1_2;
		mean0_3 += X0_3;
		mean1_3 += X1_3;
	}
	// The remaining
	for(; n < N; n++, nD += D) {
		mean0 += X[nD];
		mean1 += X[nD+1];
	}

	mean0 += mean0_0 + mean0_1 + mean0_2 + mean0_3;
	mean0 += mean0_4 + mean0_5 + mean0_6 + mean0_7;
	mean1 += mean1_0 + mean1_1 + mean1_2 + mean1_3;
	mean1 += mean1_4 + mean1_5 + mean1_6 + mean1_7;
	mean0 /= (double) N;
	mean1 /= (double) N;

	// Subtract data mean
	for(n = 0, nD = 0; n + K <= N; n += K, nD += K*D) {
		X[nD] 		-= mean0;
		X[nD+1] 	-= mean1;
		X[nD+D] 	-= mean0;
		X[nD+D+1] 	-= mean1;
		X[nD+2*D] 	-= mean0;
		X[nD+2*D+1] -= mean1;
		X[nD+3*D] 	-= mean0;
		X[nD+3*D+1] -= mean1;
		X[nD+4*D] 	-= mean0;
		X[nD+4*D+1] -= mean1;
		X[nD+5*D] 	-= mean0;
		X[nD+5*D+1] -= mean1;
		X[nD+6*D] 	-= mean0;
		X[nD+6*D+1] -= mean1;
		X[nD+7*D] 	-= mean0;
		X[nD+7*D+1] -= mean1;
	}
	// With 4 accumulators
	for(; n + K/2 <= N; n += K/2, nD += K/2*D) {
		X[nD] 		-= mean0;
		X[nD+1] 	-= mean1;
		X[nD+D] 	-= mean0;
		X[nD+D+1] 	-= mean1;
		X[nD+2*D] 	-= mean0;
		X[nD+2*D+1] -= mean1;
		X[nD+3*D] 	-= mean0;
		X[nD+3*D+1] -= mean1;
	}
	// The remaining
	for(; n < N; n++, nD += D) {
		X[nD] -= mean0;
		X[nD+1] -= mean1;
	}
}

inline void unroll_accum4(float* X, int N, int D) {
	int nD, n;
	const int K = 4;

	float mean0_0 = 0;
	float mean0_1 = 0;
	float mean0_2 = 0;
	float mean0_3 = 0;
	float mean1_0 = 0;
	float mean1_1 = 0;
	float mean1_2 = 0;
	float mean1_3 = 0;

	float mean0 = 0;
	float mean1 = 0;

	for (n = 0, nD = 0; n + K <= N; n += K, nD += K*D) {
		float X0_0 = X[nD];
		float X1_0 = X[nD+1];
		float X0_1 = X[nD+D];
		float X1_1 = X[nD+D+1];
		float X0_2 = X[nD+2*D];
		float X1_2 = X[nD+2*D+1];
		float X0_3 = X[nD+3*D];
		float X1_3 = X[nD+3*D+1];

		mean0_0 += X0_0;
		mean1_0 += X1_0;
		mean0_1 += X0_1;
		mean1_1 += X1_1;
		mean0_2 += X0_2;
		mean1_2 += X1_2;
		mean0_3 += X0_3;
		mean1_3 += X1_3;
	}
	for(; n < N; n++, nD += D) {
		mean0 += X[nD];
		mean1 += X[nD+1];
	}

	mean0 += mean0_0 + mean0_1 + mean0_2 + mean0_3;
	mean1 += mean1_0 + mean1_1 + mean1_2 + mean1_3;
	mean0 /= (double) N;
	mean1 /= (double) N;

	// Subtract data mean
	for(n = 0, nD = 0; n + K <= N; n += K, nD += K*D) {
		X[nD] 		-= mean0;
		X[nD+1] 	-= mean1;
		X[nD+D] 	-= mean0;
		X[nD+D+1] 	-= mean1;
		X[nD+2*D] 	-= mean0;
		X[nD+2*D+1] -= mean1;
		X[nD+3*D] 	-= mean0;
		X[nD+3*D+1] -= mean1;
	}
	for(; n < N; n++, nD += D) {
		X[nD] -= mean0;
		X[nD+1] -= mean1;
	}
}

// Normalize X substracting mean and
inline void base_version(float* X, int N, int D) {
	int nD = 0;

	float mean0 = 0;
	float mean1 = 0;

	for(int n = 0; n < N; n++) {
		mean0 += X[nD];
		mean1 += X[nD+1];
        nD += D;
	}
	mean0 /= (double) N;
	mean1 /= (double) N;
	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		X[nD] -= mean0;
		X[nD+1] -= mean1;
        nD += D;
	}
}

#endif
