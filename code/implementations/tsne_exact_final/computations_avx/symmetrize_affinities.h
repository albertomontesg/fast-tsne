#ifndef SYMMETRIZE_AFFINITIES_H
#define SYMMETRIZE_AFFINITIES_H

#include <immintrin.h>

// Symmetrize pairwise affinities P_ij
inline void symmetrize_affinities(float* P, int N, float scale) {
	int nN = 0;
	float sum_P = .0;
    __m256 sum_P_vector = _mm256_setzero_ps();
	int m = 0; int n=0;
	//build 8*8 blocks an compute with them
	for(n = 0; n + 8 <= N; n+=8) {
		int mN = nN; // mN: start of block
		for(m = n; m + 8 <= N; m+=8) {
            // Load first blocks (correspond to rows)
            int row0_index = mN;
			int row1_index = mN + N;
			int row2_index = mN + 2*N;
			int row3_index = mN + 3*N;
			int row4_index = mN + 4*N;
			int row5_index = mN + 5*N;
			int row6_index = mN + 6*N;
			int row7_index = mN + 7*N;

			__m256 row0 = _mm256_loadu_ps(P+row0_index);
			__m256 row1 = _mm256_loadu_ps(P+row1_index);
			__m256 row2 = _mm256_loadu_ps(P+row2_index);
			__m256 row3 = _mm256_loadu_ps(P+row3_index);
			__m256 row4 = _mm256_loadu_ps(P+row4_index);
			__m256 row5 = _mm256_loadu_ps(P+row5_index);
			__m256 row6 = _mm256_loadu_ps(P+row6_index);
			__m256 row7 = _mm256_loadu_ps(P+row7_index);

			int mN_transpose = m*N + n; // start of transposed block
			int col_index0 = mN_transpose;
			int col_index1 = mN_transpose + N;
			int col_index2 = mN_transpose + 2*N;
			int col_index3 = mN_transpose + 3*N;
			int col_index4 = mN_transpose + 4*N;
			int col_index5 = mN_transpose + 5*N;
			int col_index6 = mN_transpose + 6*N;
			int col_index7 = mN_transpose + 7*N;

			__m256 col0 = _mm256_loadu_ps(P+col_index0);
			__m256 col1 = _mm256_loadu_ps(P+col_index1);
			__m256 col2 = _mm256_loadu_ps(P+col_index2);
			__m256 col3 = _mm256_loadu_ps(P+col_index3);
			__m256 col4 = _mm256_loadu_ps(P+col_index4);
			__m256 col5 = _mm256_loadu_ps(P+col_index5);
			__m256 col6 = _mm256_loadu_ps(P+col_index6);
			__m256 col7 = _mm256_loadu_ps(P+col_index7);


            // instead of transposing one block, adding together, storing back one, storing back 2nd transposed, we do:
            // transposing both blocks, adding together each, then can do store both aligned
			__m256 row_transposed0 = col0;
			__m256 row_transposed1 = col1;
			__m256 row_transposed2 = col2;
			__m256 row_transposed3 = col3;
			__m256 row_transposed4 = col4;
			__m256 row_transposed5 = col5;
			__m256 row_transposed6 = col6;
			__m256 row_transposed7 = col7;

			transpose8_ps(row_transposed0, row_transposed1, row_transposed2, row_transposed3, row_transposed4, row_transposed5, row_transposed6, row_transposed7);

			__m256 col_transposed0 = row0;
			__m256 col_transposed1 = row1;
			__m256 col_transposed2 = row2;
			__m256 col_transposed3 = row3;
			__m256 col_transposed4 = row4;
			__m256 col_transposed5 = row5;
			__m256 col_transposed6 = row6;
			__m256 col_transposed7 = row7;

			transpose8_ps(col_transposed0, col_transposed1, col_transposed2, col_transposed3, col_transposed4, col_transposed5, col_transposed6, col_transposed7);

            __m256 row0_sum = _mm256_add_ps(row0,row_transposed0);
			__m256 row1_sum = _mm256_add_ps(row1,row_transposed1);
			__m256 row2_sum = _mm256_add_ps(row2,row_transposed2);
			__m256 row3_sum = _mm256_add_ps(row3,row_transposed3);
			__m256 row4_sum = _mm256_add_ps(row4,row_transposed4);
			__m256 row5_sum = _mm256_add_ps(row5,row_transposed5);
			__m256 row6_sum = _mm256_add_ps(row6,row_transposed6);
			__m256 row7_sum = _mm256_add_ps(row7,row_transposed7);

            __m256 col0_sum = _mm256_add_ps(col0,col_transposed0);
			__m256 col1_sum = _mm256_add_ps(col1,col_transposed1);
			__m256 col2_sum = _mm256_add_ps(col2,col_transposed2);
			__m256 col3_sum = _mm256_add_ps(col3,col_transposed3);
			__m256 col4_sum = _mm256_add_ps(col4,col_transposed4);
			__m256 col5_sum = _mm256_add_ps(col5,col_transposed5);
			__m256 col6_sum = _mm256_add_ps(col6,col_transposed6);
			__m256 col7_sum = _mm256_add_ps(col7,col_transposed7);


			_mm256_storeu_ps(P+row0_index,row0_sum);
			_mm256_storeu_ps(P+row1_index,row1_sum);
			_mm256_storeu_ps(P+row2_index,row2_sum);
			_mm256_storeu_ps(P+row3_index,row3_sum);
			_mm256_storeu_ps(P+row4_index,row4_sum);
			_mm256_storeu_ps(P+row5_index,row5_sum);
			_mm256_storeu_ps(P+row6_index,row6_sum);
			_mm256_storeu_ps(P+row7_index,row7_sum);

			_mm256_storeu_ps(P+col_index0,col0_sum);
			_mm256_storeu_ps(P+col_index1,col1_sum);
			_mm256_storeu_ps(P+col_index2,col2_sum);
			_mm256_storeu_ps(P+col_index3,col3_sum);
			_mm256_storeu_ps(P+col_index4,col4_sum);
			_mm256_storeu_ps(P+col_index5,col5_sum);
			_mm256_storeu_ps(P+col_index6,col6_sum);
			_mm256_storeu_ps(P+col_index7,col7_sum);


            // additionally sum the elements
            __m256 sum_P_helper1 = _mm256_add_ps(row0_sum, row1_sum);
            __m256 sum_P_helper2 = _mm256_add_ps(row2_sum, row3_sum);
            __m256 sum_P_helper3 = _mm256_add_ps(row4_sum, row5_sum);
            __m256 sum_P_helper4 = _mm256_add_ps(row6_sum, row7_sum);
            sum_P_helper1 = _mm256_add_ps(sum_P_helper1, sum_P_helper2);
            sum_P_helper3 = _mm256_add_ps(sum_P_helper3, sum_P_helper4);
            sum_P_helper1 = _mm256_add_ps(sum_P_helper1, sum_P_helper3);
            sum_P_vector = _mm256_add_ps(sum_P_vector, sum_P_helper1);

            // all elements that are in the diagonal are counted two times, so half them afterwards
            // also remove 2nd sum term
            if(n==m){
                P[mN] *= 0.5;
                sum_P -= P[mN];
                P[mN + N + 1] *= 0.5;
                sum_P -= P[mN + 1*N + 1];
                P[mN + 2*N + 2] *= 0.5; // all elements of diagonal in current block
                sum_P -= P[mN + 2*N + 2];
                P[mN + 3*N + 3] *= 0.5;
                sum_P -= P[mN + 3*N + 3];
                P[mN + 4*N + 4] *= 0.5;
                sum_P -= P[mN + 4*N + 4];
                P[mN + 5*N + 5] *= 0.5;
                sum_P -= P[mN + 5*N + 5];
                P[mN + 6*N + 6] *= 0.5;
                sum_P -= P[mN + 6*N + 6];
                P[mN + 7*N + 7] *= 0.5;
                sum_P -= P[mN + 7*N + 7];
                }

			mN += 8; // next block
            mN_transpose += N; // next transposed block

		}
		nN += N; // next row
	}

    //TODO: N not dividable by 8 case. Further optimizable?
    for (; n < N; ++n){// where left off
	    int mN = 0;
        for(int m = 0; m < N; m++) {
            if(m==n){
                mN += N;
                continue;
            }
		P[nN + m] += P[mN + n];
		P[mN + n]  = P[nN + m];
            sum_P += P[nN +m];
		    mN += N;
	    }
	    nN += N;
    }


	//compute the sum_S
    __m256 s = _mm256_hadd_ps(sum_P_vector,sum_P_vector);
    sum_P +=  s[0] + s[1] + s[4] + s[5];

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


#endif
