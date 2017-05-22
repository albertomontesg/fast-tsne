#ifndef SYMMETRIZE_AFFINITIES_H
#define SYMMETRIZE_AFFINITIES_H

#include <stdio.h>
#include <immintrin.h>

// Compute low dimensional affinities
inline void unrolling(float* P, int N, float scale) {
	int nN = 0;
	float sum_P = .0;
	int m = 0;

    for(int n = 0; n < N; n++) {
		int mN = (n + 1) * N;
		sum_P += P[nN + n];
		for(m = n + 1; m < N; m+=8) {
			int mNn = mN + n;
			int i1 = mNn + N;
			int i2 = mNn + 2*N;
			int i3 = mNn + 3*N;
			int i4 = mNn + 4*N;
			int i5 = mNn + 5*N;
			int i6 = mNn + 6*N;
			int i7 = mNn + 7*N;

			__m256 P_row = _mm256_loadu_ps(P + nN + m);

			//here is the bad spatial locality -> we always access elements with N stride

			__m256 P_col = _mm256_set_ps(P[i7],P[i6],P[i5],P[i4],P[i3],P[i2],P[i1],P[mNn]);

			__m256 P_sum_row_col = _mm256_add_ps(P_row,P_col);
			_mm256_storeu_ps(P+nN + m,P_sum_row_col);

			P[i7]= P_sum_row_col[7];
			P[i6]= P_sum_row_col[6];
			P[i5]= P_sum_row_col[5];
			P[i4]= P_sum_row_col[4];
			P[i3] = P_sum_row_col[3];
			P[i2] = P_sum_row_col[2];
			P[i1] = P_sum_row_col[1];
			P[mNn] = P_sum_row_col[0];

			__m256 P_row_update = _mm256_loadu_ps(P+nN + m);
			//__m256 P_col_update = _mm256_set_ps(P[mN + 7*N + n],P[mN + 6*N + n],P[mN + 5*N + n],P[mN + 4*N + n],P[mN + 2*N + n],P[mN + 2*N + n],P[mN + N + n],P[mN + n]);
			__m256 s = _mm256_hadd_ps(P_row_update,P_row_update);
			sum_P +=  s[0] + s[1] + s[4] + s[5];

			mN += 8*N;

		}
		//if N is not multiplicative factor of 8 do the rest sequentially
		for (int i = m; i < N; ++i){
			P[nN + i] += P[mN + n];
			P[mN + n]  = P[nN + i];
			mN += N;
		}

		nN += N;
	}

	__m256 scale_vec = _mm256_set1_ps(scale);
	__m256 sum_vec = _mm256_set1_ps(sum_P);
	__m256 scale_sum_vec = _mm256_div_ps(scale_vec,sum_vec);
	int j = 0;
	for(j = 0; j < N * N; j+=8){
		__m256 P_vec = _mm256_loadu_ps(P+j);
		__m256 P_vec_scaled_norm = _mm256_mul_ps(P_vec,scale_sum_vec);
		_mm256_storeu_ps(P+j,P_vec_scaled_norm);
	}

	//if N*N is not multiplicative factor of 8 do the rest sequentially
	for (int i = j; i < N*N; ++i)
	{
		P[i] *= scale;
		P[i] /= sum_P;
	}
}

inline void blocking(float* P, int N, float scale) {
	int nN = 0;
	float sum_P = .0;
	int m = 0;
	//build 2*2 blocks an compute with them
	for(int n = 0; n < N; n+=8) {
		int mN = (n + 1) * N;
		for(m = n; m < N; m+=8) {
			int nNm = nN + m;
			int k1 = nNm + N;
			int k2 = nNm + 2*N;
			int k3 = nNm + 3*N;
			int k4 = nNm + 4*N;
			int k5 = nNm + 5*N;
			int k6 = nNm + 6*N;
			int k7 = nNm + 7*N;

			__m256 rowb1_1 = _mm256_loadu_ps(P+nNm);
			__m256 rowb1_2 = _mm256_loadu_ps(P+k1);
			__m256 rowb1_3 = _mm256_loadu_ps(P+k2);
			__m256 rowb1_4 = _mm256_loadu_ps(P+k3);
			__m256 rowb1_5 = _mm256_loadu_ps(P+k4);
			__m256 rowb1_6 = _mm256_loadu_ps(P+k5);
			__m256 rowb1_7 = _mm256_loadu_ps(P+k6);
			__m256 rowb1_8 = _mm256_loadu_ps(P+k7);

			int mNn = mN + n;
			int i1 = mNn + N;
			int i2 = mNn + 2*N;
			int i3 = mNn + 3*N;
			int i4 = mNn + 4*N;
			int i5 = mNn + 5*N;
			int i6 = mNn + 6*N;
			int i7 = mNn + 7*N;

			__m256 rowb2_1 = _mm256_loadu_ps(P+mNn);
			__m256 rowb2_2 = _mm256_loadu_ps(P+i1);
			__m256 rowb2_3 = _mm256_loadu_ps(P+i2);
			__m256 rowb2_4 = _mm256_loadu_ps(P+i3);
			__m256 rowb2_5 = _mm256_loadu_ps(P+i4);
			__m256 rowb2_6 = _mm256_loadu_ps(P+i5);
			__m256 rowb2_7 = _mm256_loadu_ps(P+i6);
			__m256 rowb2_8 = _mm256_loadu_ps(P+i7);

			__m256 rc1 = _mm256_add_ps(rowb2_1,rowb1_1);
			__m256 rc2 = _mm256_add_ps(rowb2_2,rowb1_2);
			__m256 rc3 = _mm256_add_ps(rowb2_3,rowb1_3);
			__m256 rc4 = _mm256_add_ps(rowb2_4,rowb1_4);
			__m256 rc5 = _mm256_add_ps(rowb2_5,rowb1_5);
			__m256 rc6 = _mm256_add_ps(rowb2_6,rowb1_6);
			__m256 rc7 = _mm256_add_ps(rowb2_7,rowb1_7);
			__m256 rc8 = _mm256_add_ps(rowb2_8,rowb1_8);

			__m256 rowb1_1_new = _mm256_set_ps(rc8[0],rc7[0],rc6[0],rc5[0],rc4[0],rc3[0],rc2[0],rc1[0]);
			__m256 rowb1_2_new = _mm256_set_ps(rc8[1],rc7[1],rc6[1],rc5[1],rc4[1],rc3[1],rc2[1],rc1[1]);
			__m256 rowb1_3_new = _mm256_set_ps(rc8[2],rc7[2],rc6[2],rc5[2],rc4[2],rc3[2],rc2[2],rc1[2]);
			__m256 rowb1_4_new = _mm256_set_ps(rc8[3],rc7[3],rc6[3],rc5[3],rc4[3],rc3[3],rc2[3],rc1[3]);
			__m256 rowb1_5_new = _mm256_set_ps(rc8[4],rc7[4],rc6[4],rc5[4],rc4[4],rc3[4],rc2[4],rc1[4]);
			__m256 rowb1_6_new = _mm256_set_ps(rc8[5],rc7[5],rc6[5],rc5[5],rc4[5],rc3[5],rc2[5],rc1[5]);
			__m256 rowb1_7_new = _mm256_set_ps(rc8[6],rc7[6],rc6[6],rc5[6],rc4[6],rc3[6],rc2[6],rc1[6]);
			__m256 rowb1_8_new = _mm256_set_ps(rc8[7],rc7[7],rc6[7],rc5[7],rc4[7],rc3[7],rc2[7],rc1[7]);

			//if(m == n){ //our block has a diagonal crossing
				//in this case we have to divide the diagonal elements by 2
				//but the numbers are so small anyway so its fine? MIN_VAL for the diag

			//}else{
				_mm256_storeu_ps(P+nNm,rowb1_1_new);
				_mm256_storeu_ps(P+k1,rowb1_1_new);
				_mm256_storeu_ps(P+k2,rowb1_2_new);
				_mm256_storeu_ps(P+k3,rowb1_3_new);
				_mm256_storeu_ps(P+k4,rowb1_4_new);
				_mm256_storeu_ps(P+k5,rowb1_5_new);
				_mm256_storeu_ps(P+k6,rowb1_6_new);
				_mm256_storeu_ps(P+k7,rowb1_7_new);

				_mm256_storeu_ps(P+mNn,rc1);
				_mm256_storeu_ps(P+i1,rc2);
				_mm256_storeu_ps(P+i2,rc3);
				_mm256_storeu_ps(P+i3,rc4);
				_mm256_storeu_ps(P+i4,rc5);
				_mm256_storeu_ps(P+i5,rc6);
				_mm256_storeu_ps(P+i6,rc7);
				_mm256_storeu_ps(P+i7,rc8);

			//}
			mN += 8*N;

		}

		//if N is not multiplicative factor of 8 do the rest sequentially
		for (int i = m; i < N; ++i)
		{
			P[nN + i] += P[mN + n];
			P[mN + n]  = P[nN + i];
			mN += N;
		}

		nN += N;
	}


	//compute the sum_S

	__m256 scale_vec = _mm256_set1_ps(scale);
	__m256 sum_vec = _mm256_set1_ps(sum_P);
	__m256 scale_sum_vec = _mm256_div_ps(scale_vec,sum_vec);
	int j = 0;
	for(j = 0; j < N * N; j+=8){
		__m256 P_vec = _mm256_loadu_ps(P+j);
		__m256 P_vec_scaled_norm = _mm256_mul_ps(P_vec,scale_sum_vec);
		_mm256_storeu_ps(P+j,P_vec_scaled_norm);
	}

	//if N*N is not multiplicative factor of 8 do the rest sequentially
	for (int i = j; i < N*N; ++i)
	{
		P[i] *= scale;
		P[i] /= sum_P;
	}
}

inline void base_version(float* P, int N, float scale) {
	int nN = 0;
	for(int n = 0; n < N; n++) {
		int mN = (n + 1) * N;
		for(int m = n + 1; m < N; m++) {
			P[nN + m] += P[mN + n];
			P[mN + n]  = P[nN + m];
			mN += N;
		}
		nN += N;
	}
	dt sum_P = .0;
	for(int i = 0; i < N * N; i++) sum_P += P[i];
	for(int i = 0; i < N * N; i++) P[i] /= sum_P;

    for(int i = 0; i < N * N; i++) P[i] *= scale;
}

#endif
