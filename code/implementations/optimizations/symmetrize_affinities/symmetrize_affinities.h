#ifndef SYMMETRIZE_AFFINITIES_H
#define SYMMETRIZE_AFFINITIES_H

#include <stdio.h>
#include <x86intrin.h>

inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2,
        __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
    __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
    __t0 = _mm256_unpacklo_ps(row0, row1);
    __t1 = _mm256_unpackhi_ps(row0, row1);
    __t2 = _mm256_unpacklo_ps(row2, row3);
    __t3 = _mm256_unpackhi_ps(row2, row3);
    __t4 = _mm256_unpacklo_ps(row4, row5);
    __t5 = _mm256_unpackhi_ps(row4, row5);
    __t6 = _mm256_unpacklo_ps(row6, row7);
    __t7 = _mm256_unpackhi_ps(row6, row7);
    __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
    __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
    __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
    __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
    __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
    __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
    __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
    __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
    row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
    row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
    row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
    row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
    row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
    row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
    row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
    row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

// Compute low dimensional affinities
inline void unrolling(float* P, int N, float scale) {
	int nN = 0;
	float sum_P = .0;
	int m = 0;

    for(int n = 0; n < N; n++) {
		int mN = (n + 1) * N;
		sum_P += P[nN + n];
		for(m = n + 1; m + 8 <= N; m+=8) {
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
            __m256 row_transposed0 = _mm256_set_ps(col0[0],col1[0],col2[0],col3[0],
                    col4[0],col5[0],col6[0],col7[0]);
            __m256 row_transposed1 = _mm256_set_ps(col0[1],col1[1],col2[1],col3[1],
                    col4[1],col5[1],col6[1],col7[0]);
            __m256 row_transposed2 = _mm256_set_ps(col0[2],col1[2],col2[2],col3[2],
                    col4[2],col5[2],col6[2],col7[2]);
            __m256 row_transposed3 = _mm256_set_ps(col0[3],col1[3],col2[3],col3[3],
                    col4[3],col5[3],col6[3],col7[3]);
            __m256 row_transposed4 = _mm256_set_ps(col0[4],col1[4],col2[4],col3[4],
                    col4[4],col5[4],col6[4],col7[4]);
            __m256 row_transposed5 = _mm256_set_ps(col0[5],col1[5],col2[5],col3[5],
                    col4[5],col5[5],col6[5],col7[5]);
            __m256 row_transposed6 = _mm256_set_ps(col0[6],col1[6],col2[6],col3[6],
                    col4[6],col5[6],col6[6],col7[6]);
            __m256 row_transposed7 = _mm256_set_ps(col0[7],col1[7],col2[7],col3[7],
                    col4[7],col5[7],col6[7],col7[7]);

            __m256 col_transposed0 = _mm256_set_ps(row0[0],row1[0],row2[0],row3[0],
                    row4[0],row5[0],row6[0],row7[0]);
            __m256 col_transposed1 = _mm256_set_ps(row0[1],row1[1],row2[1],row3[1],
                    row4[1],row5[1],row6[1],row7[0]);
            __m256 col_transposed2 = _mm256_set_ps(row0[2],row1[2],row2[2],row3[2],
                    row4[2],row5[2],row6[2],row7[2]);
            __m256 col_transposed3 = _mm256_set_ps(row0[3],row1[3],row2[3],row3[3],
                    row4[3],row5[3],row6[3],row7[3]);
            __m256 col_transposed4 = _mm256_set_ps(row0[4],row1[4],row2[4],row3[4],
                    row4[4],row5[4],row6[4],row7[4]);
            __m256 col_transposed5 = _mm256_set_ps(row0[5],row1[5],row2[5],row3[5],
                    row4[5],row5[5],row6[5],row7[5]);
            __m256 col_transposed6 = _mm256_set_ps(row0[6],row1[6],row2[6],row3[6],
                    row4[6],row5[6],row6[6],row7[6]);
            __m256 col_transposed7 = _mm256_set_ps(row0[7],row1[7],row2[7],row3[7],
                    row4[7],row5[7],row6[7],row7[7]);



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
    if(N%8 != 0){
        // int nN = N * ((n-8)+1);
        for (; n < N; ++n) { // where left off
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
    }

	//compute the sum_S
    __m256 s = _mm256_hadd_ps(sum_P_vector,sum_P_vector);
    sum_P +=  s[0] + s[1] + s[4] + s[5];

	__m256 scale_vec = _mm256_set1_ps(scale);
	__m256 sum_vec = _mm256_set1_ps(sum_P);
	__m256 scale_sum_vec = _mm256_div_ps(scale_vec,sum_vec);
	int j = 0;
	for(j = 0; j + 8 <= N * N; j+=8){
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

inline void blocking2(float* P, int N, float scale) {
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
            __m256 row_transposed0 = _mm256_set_ps(col0[0],col1[0],col2[0],col3[0],
                    col4[0],col5[0],col6[0],col7[0]);
            __m256 row_transposed1 = _mm256_set_ps(col0[1],col1[1],col2[1],col3[1],
                    col4[1],col5[1],col6[1],col7[0]);
            __m256 row_transposed2 = _mm256_set_ps(col0[2],col1[2],col2[2],col3[2],
                    col4[2],col5[2],col6[2],col7[2]);
            __m256 row_transposed3 = _mm256_set_ps(col0[3],col1[3],col2[3],col3[3],
                    col4[3],col5[3],col6[3],col7[3]);
            __m256 row_transposed4 = _mm256_set_ps(col0[4],col1[4],col2[4],col3[4],
                    col4[4],col5[4],col6[4],col7[4]);
            __m256 row_transposed5 = _mm256_set_ps(col0[5],col1[5],col2[5],col3[5],
                    col4[5],col5[5],col6[5],col7[5]);
            __m256 row_transposed6 = _mm256_set_ps(col0[6],col1[6],col2[6],col3[6],
                    col4[6],col5[6],col6[6],col7[6]);
            __m256 row_transposed7 = _mm256_set_ps(col0[7],col1[7],col2[7],col3[7],
                    col4[7],col5[7],col6[7],col7[7]);

            __m256 col_transposed0 = _mm256_set_ps(row0[0],row1[0],row2[0],row3[0],
                    row4[0],row5[0],row6[0],row7[0]);
            __m256 col_transposed1 = _mm256_set_ps(row0[1],row1[1],row2[1],row3[1],
                    row4[1],row5[1],row6[1],row7[0]);
            __m256 col_transposed2 = _mm256_set_ps(row0[2],row1[2],row2[2],row3[2],
                    row4[2],row5[2],row6[2],row7[2]);
            __m256 col_transposed3 = _mm256_set_ps(row0[3],row1[3],row2[3],row3[3],
                    row4[3],row5[3],row6[3],row7[3]);
            __m256 col_transposed4 = _mm256_set_ps(row0[4],row1[4],row2[4],row3[4],
                    row4[4],row5[4],row6[4],row7[4]);
            __m256 col_transposed5 = _mm256_set_ps(row0[5],row1[5],row2[5],row3[5],
                    row4[5],row5[5],row6[5],row7[5]);
            __m256 col_transposed6 = _mm256_set_ps(row0[6],row1[6],row2[6],row3[6],
                    row4[6],row5[6],row6[6],row7[6]);
            __m256 col_transposed7 = _mm256_set_ps(row0[7],row1[7],row2[7],row3[7],
                    row4[7],row5[7],row6[7],row7[7]);



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


inline void blocking3(float* P, int N, float scale) {
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
	float sum_P = .0;
	for(int i = 0; i < N * N; i++) sum_P += P[i];
	for(int i = 0; i < N * N; i++) P[i] /= sum_P;

    for(int i = 0; i < N * N; i++) P[i] *= scale;
}

#endif
