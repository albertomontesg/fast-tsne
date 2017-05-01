#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "../utils/io.h"
#include "../utils/tsc_x86.h"


#define DEBUG 1
#define COUNTING 0
#define NUMERIC_CHECK 1
#define BENCHMARK 0
#define NUM_RUNS 1

// Avoid any other task while benchmarking
#if BENCHMARK == 1
#undef DEBUG
#define DEBUG 0
#undef COUNTING
#define COUNTING 0
#undef NUMERIC_CHECK
#define NUMERIC_CHECK 0
#else
#undef NUM_RUNS
#define NUM_RUNS 1
#endif

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// Generates a Gaussian random number
double randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	return x;
}


// Normalize X substracting mean and
void normalize(double* X, int N, int D, double* mean, bool max_value) {
	int nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			mean[d] += X[nD + d];
		}
        nD += D;
	}
	for(int d = 0; d < D; d++) {
		mean[d] /= (double) N;
	}

	// Subtract data mean
    nD = 0;
	for(int n = 0; n < N; n++) {
		for(int d = 0; d < D; d++) {
			X[nD + d] -= mean[d];
		}
        nD += D;
	}

	if (max_value) {
		// Normalize to the maximum absolute value
		double max_X = .0;
		for(int i = 0; i < N * D; i++) {
	        if(fabs(X[i]) > max_X) max_X = fabs(X[i]);
	    }
	    for(int i = 0; i < N * D; i++) X[i] /= max_X;
	}
}


// Compute squared euclidean disctance for all pairs of vectors X_i X_j
void compute_squared_euclidean_distance(double* X, int N, int D, double* DD) {
	const double* XnD = X;
    for(int n = 0; n < N; ++n, XnD += D) {
        const double* XmD = XnD + D;
        double* curr_elem = &DD[n*N + n];
        *curr_elem = 0.0;
        double* curr_elem_sym = curr_elem + N;
        for(int m = n + 1; m < N; ++m, XmD+=D, curr_elem_sym+=N) {
            *(++curr_elem) = 0.0;
            for(int d = 0; d < D; ++d) {
                *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
            }
            *curr_elem_sym = *curr_elem;
        }
    }
}


// Compute pairwise affinity perplexity
void compute_pairwise_affinity_perplexity(double* X, int N, int D, double* P,
										  double perplexity, double* DD){

	compute_squared_euclidean_distance(X, N, D, DD);

	// Compute the Gaussian kernel row by row
	int nN = 0;
	for (int n = 0; n < N; n++) {
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MIN;
		double tol = 1e-5;
		double sum_P;

		int iter = 0;
		while (!found && iter < 200) {
			// Compute Gaussian kernel row
			for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
			P[nN + n] = DBL_MIN;

			// Compute entropy of current row
			sum_P = DBL_MIN;
			for (int m = 0; m < N; m++) sum_P += P[nN + m];
			double H = 0.0;
			for(int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
			H = (H / sum_P) + log(sum_P);

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity);
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}
			// Update iteration counter
			iter++;
		}
		// Row normalize P
		for(int m = 0; m < N; m++) P[nN + m] /= sum_P;
		nN += N;
	}

}


// Symmetrize pairwise affinities P_ij
void symmetrize_affinities(double* P, int N) {
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
	double sum_P = .0;
	for(int i = 0; i < N * N; i++) sum_P += P[i];
	for(int i = 0; i < N * N; i++) P[i] /= sum_P;
}


// Compute low dimensional affinities
double compute_low_dimensional_affinities(double* Y, int N, int no_dims,
										  double* Q, double* DD) {

	compute_squared_euclidean_distance(Y, N, no_dims, DD);

	double sum_Q = .0;
    int nN = 0;
    for(int n = 0; n < N; n++) {
    	for(int m = 0; m < N; m++) {
            if(n != m) {
                Q[nN + m] = 1 / (1 + DD[nN + m]);
                sum_Q += Q[nN + m];
            }
        }
        nN += N;
    }

	return sum_Q;
}


// Gradient computation dC_dy
void gradient_computation(double* Y, double* P, double* Q, double sum_Q, int N,
						  int D, double* dC) {
	// Perform the computation of the gradient
	int nN = 0;
	int nD = 0;
	for(int n = 0; n < N; n++) {
		int mD = 0;
		for(int m = 0; m < N; m++) {
			if(n != m) {
				double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
				for(int d = 0; d < D; d++) {
					dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
				}
			}
			mD += D;
		}
		nN += N;
		nD += D;
	}
}


// Update gains and update Y with the computed gradient
void gradient_update(double* Y, double* dC, double* uY, double* gains, int N,
					 int no_dims, double momentum, double eta){
	// Update gains
	for(int i = 0; i < N * no_dims; i++)
		gains[i] = (sign(dC[i]) != sign(uY[i])) ? (gains[i] + .2) :
												  (gains[i] * .8);
	for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

	// Perform gradient update (with momentum and gains)
	for(int i = 0; i < N * no_dims; i++)
		uY[i] = momentum * uY[i] - eta * gains[i] * dC[i];
	for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
}


// Run function
double run(double* X, int N, int D, double* Y, int no_dims, double perplexity,
	 	 int max_iter) {
	double cycles = 0;

	#if BENCHMARK
	myInt64 start_normalize, cycles_normalize;
	myInt64 start_perplexity, cycles_perplexity;
	myInt64 start_symmetrize, cycles_symmetrize;
	myInt64 start_ld_affinity, cycles_ld_affinity = 0;
	myInt64 start_gradient, cycles_gradient = 0;
	myInt64 start_update, cycles_update = 0;
	#endif

	// Normalize input X (substract mean and normalize to the maximum value
	// to avoid numerical inestabilities)
	double* mean = (double*) calloc(D, sizeof(double));
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }
	#if BENCHMARK
	start_normalize = start_tsc();
	#endif
	// Compute
	normalize(X, N, D, mean, true);
	// End compute
	#if BENCHMARK
	cycles_normalize = stop_tsc(start_normalize);
	cycles += (double) cycles_normalize;
	#endif
	#if NUMERIC_CHECK
	save_data(X, N, D, "./datum/X_normalized");
	#endif
	free(mean); mean = NULL;


	// Compute pairsiwe affinity with perplexity which include binary search
	// for the best perplexity value
	double* P = (double*) calloc(N * N, sizeof(double));
	double* DD = (double*) malloc(N * N * sizeof(double));
	if(P == NULL) { printf("[P] Memory allocation failed!\n"); exit(1); }
	if(DD == NULL) { printf("[DD] Memory allocation failed!\n"); exit(1); }
	#if BENCHMARK
	start_perplexity = start_tsc();
	#endif
	// Compute
	compute_pairwise_affinity_perplexity(X, N, D, P, perplexity, DD);
	// End compute
	#if BENCHMARK
	cycles_perplexity = stop_tsc(start_perplexity);
	cycles += cycles_perplexity;
	#endif
	// P and DD will not be remove because future use
	#if NUMERIC_CHECK
	save_data(P, N, N, "./datum/P");
	#endif


	// Symmetrize affinities
	#if BENCHMARK
	start_symmetrize = start_tsc();
	#endif
	// Compute
	symmetrize_affinities(P, N);
	// End compute
	#if BENCHMARK
	cycles_symmetrize = stop_tsc(start_symmetrize);
	cycles += cycles_symmetrize;
	#endif
	#if NUMERIC_CHECK
	save_data(P, N, N, "./datum/P_sym");
	#endif


	// Early exageration
	for(int i = 0; i < N * N; i++) P[i] *= 12.0;

	// Initialize Q low dimensionality affinity matrix and gradient dC
	double* Q = (double*) malloc(N * N * sizeof(double));
	double* dC = (double*) malloc(N * no_dims * sizeof(double));
	double* uY = (double*) calloc(N * no_dims, sizeof(double));
	double* gains = (double*) malloc(N * no_dims * sizeof(double));
	mean = (double*) malloc(no_dims * sizeof(double));
	if(Q == NULL) { printf("[Q] Memory allocation failed!\n"); exit(1); }
	if(dC == NULL) { printf("[dC] Memory allocation failed!\n"); exit(1); }
	if(uY == NULL) { printf("[uY] Memory allocation failed!\n"); exit(1); }
	if(gains == NULL) {printf("[gains] Memory allocation failed!\n"); exit(1);}
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

	// Training parameters
	double eta = 200.0;
	double momentum = .5;
	double final_momentum = .8;
	// Perform the main training loop
	for (int iter = 0; iter < max_iter; iter++) {

		// Compute low dimensional affinities
		// Reset DD to all 0s
		for (int i = 0; i < N * N; i++) DD[i] = 0.;
		#if BENCHMARK
		start_ld_affinity = start_tsc();
		#endif
		// Compute
		double sum_Q;
		sum_Q = compute_low_dimensional_affinities(Y, N, no_dims, Q, DD);
		// End compute
		#if BENCHMARK
		cycles_ld_affinity += stop_tsc(start_ld_affinity);
		#endif
		#if NUMERIC_CHECK
		if (iter == 0) printf("%lf\n", sum_Q);
		if (iter == 0) save_data(Q, N, N, "./datum/Q_0");
		if (iter == 300) save_data(Q, N, N, "./datum/Q_300");
		#endif


		// Gradient Computation
		// Make sure the gradient contains all zeros
		for (int i = 0; i < N * no_dims; i++) dC[i] = 0.;
		#if BENCHMARK
		start_gradient = start_tsc();
		#endif
		// Compute
		gradient_computation(Y, P, Q, sum_Q, N, no_dims, dC);
		// End compute
		#if BENCHMARK
		cycles_gradient += stop_tsc(start_gradient);
		#endif
		#if NUMERIC_CHECK
		if (iter == 0) save_data(dC, N, no_dims, "./datum/dC_0");
		if (iter == 300) save_data(dC, N, no_dims, "./datum/dC_300");
		#endif


		// Perform gradient update
		#if BENCHMARK
		start_update = start_tsc();
		#endif
		// Compute
		gradient_update(Y, dC, uY, gains, N, no_dims, momentum, eta);
		// End compute
		#if BENCHMARK
		cycles_update += stop_tsc(start_update);
		#endif
		#if NUMERIC_CHECK
		if (iter == 0) save_data(Y, N, no_dims, "./datum/Y_0");
		if (iter == 300) save_data(Y, N, no_dims, "./datum/Y_300");
		#endif

		// Zero mean to solution Y
		for (int i = 0; i < no_dims; i++) mean[i] = 0.;
		// Compute
		normalize(Y, N, no_dims, mean, false);
		// End compute
		#if NUMERIC_CHECK
		if (iter == 0) save_data(Y, N, no_dims, "./datum/Y_0_normalized");
		if (iter == 300) save_data(Y, N, no_dims, "./datum/Y_300_normalized");
		#endif

		// Stop early exageration
		if (iter == 250) for (int i = 0; i < N * N; i++) { P[i] /= 12.0; }
		// Switch momentum
		if (iter == 250) momentum = final_momentum;
	}

	#if BENCHMARK
	cycles += cycles_ld_affinity + cycles_gradient + cycles_update;
	#endif


	free(P); 		P = NULL;
	free(DD); 		DD = NULL;
	free(Q);		Q = NULL;
	free(dC);		dC = NULL;
	free(gains);	gains = NULL;
	free(mean);		mean = NULL;

	return cycles;
}

int main(int argc, char **argv) {
    /* Usage:
    data:       data file
    result:     result file
    N:          number of samples
    perp:       perplexity              (best value: 50)
    o_dim:      output dimensionality   (best value: 2)
    max_iter:   max iterations          (best value: 1000)
    */

    // Parse arguments
    char *data_file = argv[1];
    char *result_file = argv[2];
    int N = atoi(argv[3]);
    double perplexity = atof(argv[4]);
    int no_dims = atoi(argv[5]);
    int max_iter = atoi(argv[6]);

	// Set random seed
    int rand_seed = 23;
    srand((unsigned int) rand_seed);

	#if DEBUG == 1
	printf("Data file: %s\n", data_file);
	printf("Result file: %s\n", result_file);
	printf("---------\nN = %d\n", N);
	printf("perplexity = %.2f\n", perplexity);
	printf("no_dims = %d\n", no_dims);
	printf("max_iter = %d\n", max_iter);
	printf("Using random seed: %d\n", rand_seed);
	#endif

	// Define some variables
	int D;
	double *data = (double*) malloc(N * 784 * sizeof(double));
	double *X = (double*) malloc(N * 784 * sizeof(double));
    double* Y = (double*) malloc(N * no_dims * sizeof(double));
    if(data == NULL) { printf("[data] Memory allocation failed!\n"); exit(1); }
    if(X == NULL) { printf("[X] Memory allocation failed!\n"); exit(1); }
    if(Y == NULL ) { printf("[Y] Memory allocation failed!\n"); exit(1); }

	// Randomly intialize Y array
    for (int i = 0; i < N * no_dims; i++) {
        Y[i] = randn() * .0001;
    }

	#if NUMERIC_CHECK
	save_data(Y, N, no_dims, "./datum/Y");
	#endif

	// Read the parameters and the dataset
    bool success = load_data(data, N, &D, data_file);
    if (!success) exit(1);

    double cycles = 0;

	int num_runs = (DEBUG == 1) ? 1 : NUM_RUNS;

	for (int i = 0; i < num_runs; i++) {
		// Randomly intialize arrays
		for (int i = 0; i < N * D; i++) 		X[i] = data[i];
	    for (int i = 0; i < N * no_dims; i++) 	Y[i] = randn() * .0001;

		cycles += run(data, N, D, Y, no_dims, perplexity, max_iter);
	}
	cycles /= (double) num_runs;

	printf("%lf cycles\n", cycles);

	// Save the results
	save_data(Y, N, no_dims, result_file);

	// Clean up the memory
	free(data);   	data = NULL;
	free(X);   		X = NULL;
	free(Y);      	Y = NULL;
}
