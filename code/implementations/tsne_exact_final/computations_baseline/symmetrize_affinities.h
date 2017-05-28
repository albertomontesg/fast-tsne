#ifndef SYMMETRIZE_AFFINITIES_H
#define SYMMETRIZE_AFFINITIES_H


// Symmetrize pairwise affinities P_ij
inline void symmetrize_affinities(float* P, int N, float scale) {
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
