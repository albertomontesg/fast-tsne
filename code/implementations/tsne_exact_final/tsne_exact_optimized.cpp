#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#define SINGLE_PRECISION

#include "../utils/io.h"
#include "../utils/tsc_x86.h"
#include "../utils/random.h"

#ifdef BASELINE
#include "computations_baseline/compute_low_dimensional_affinities.h"
#include "computations_baseline/gradient_computation_update_normalize.h"
#include "computations_baseline/compute_squared_euclidean_distance_high_normalize.h"
#include "computations_baseline/symmetrize_affinities.h"
#include "computations_baseline/compute_pairwise_affinity_perplexity.h"

#elif SCALAR
#include "computations_scalar/compute_low_dimensional_affinities.h"
#include "computations_scalar/gradient_computation_update_normalize.h"
#include "computations_scalar/compute_squared_euclidean_distance_high_normalize.h"
#include "computations_scalar/symmetrize_affinities.h"
#include "computations_scalar/compute_pairwise_affinity_perplexity.h"

#elif AVX
#include "computations_avx/compute_low_dimensional_affinities.h"
#include "computations_avx/gradient_computation_update_normalize.h"
#include "computations_avx/compute_squared_euclidean_distance_high_normalize.h"
#include "computations_avx/symmetrize_affinities.h"
#include "computations_avx/compute_pairwise_affinity_perplexity.h"

#else
#define EXIT
#endif


#define NUM_RUNS 11


#ifdef BENCHMARK
double cycles = 0;
double cycles_distance = 0, cycles_perplexity = 0, cycles_symmetrize = 0;
double cycles_ld_affinity = 0, cycles_gradient = 0;
myInt64 start_distance, start_perplexity, start_symmetrize;
myInt64 start_ld_affinity, start_gradient;
#endif




// Run function
void run(float* X, int N, int D, float* Y, int no_dims, float perplexity,
	 	 int max_iter) {

    // Normalize input and compute pairwise euclidean distances for X
    float* DD = (float*) _mm_malloc(N * N * sizeof(float), 32);
    float* mean = (float*) _mm_malloc(D * sizeof(float), 32);
    if(DD == NULL) { printf("[DD] Memory allocation failed!\n"); exit(1); }
    if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }
    #ifdef BENCHMARK
    start_distance = start_tsc();
    #endif
    compute_squared_euclidean_distance_high_normalize(X, N, D, DD, mean);
    #ifdef BENCHMARK
    cycles_distance += (double) stop_tsc(start_distance);
    #endif


	// Compute pairsiwe affinity with perplexity which include binary search
	// for the best perplexity value
	float* P = (float*) _mm_malloc(N * N * sizeof(float), 32);
	if(P == NULL) { printf("[P] Memory allocation failed!\n"); exit(1); }
    for (int i = 00; i < N*N; i++) P[i] = 0;
	#ifdef BENCHMARK
	start_perplexity = start_tsc();
	#endif
	// Compute
	compute_pairwise_affinity_perplexity(X, N, D, P, perplexity, DD);
	// End compute
	#ifdef BENCHMARK
	cycles_perplexity += (double) stop_tsc(start_perplexity);
	#endif
	// P and DD will not be remove because future use
	#ifdef NUMERIC_CHECK
	save_data(P, N, N, "./datum/P");
	#endif


	// Symmetrize affinities and early exageration
	#ifdef BENCHMARK
	start_symmetrize = start_tsc();
	#endif
	// Compute
	symmetrize_affinities(P, N, 12.0);
	// End compute
	#ifdef BENCHMARK
	cycles_symmetrize += (double) stop_tsc(start_symmetrize);
	#endif
	#ifdef NUMERIC_CHECK
	save_data(P, N, N, "./datum/P_sym");
	#endif

	// Initialize Q low dimensionality affinity matrix and gradient dC
	float* Q = (float*) _mm_malloc(N * N * sizeof(float), 32);
	float* uY = (float*) _mm_malloc(N * no_dims * sizeof(float), 32);
    for (int i = 0; i < N*no_dims; i++) uY[i] = 0;
	if(Q == NULL) { printf("[Q] Memory allocation failed!\n"); exit(1); }
	if(uY == NULL) { printf("[uY] Memory allocation failed!\n"); exit(1); }

	// As the counting is perform before entering the main training loop
	// it is not necessary to run it, so time is saved to count the
	// iterations
	#ifdef COUNTING
	max_iter = 0;
	#endif

	// Training parameters
	float eta = 200.0;
	float momentum = .5;
	float final_momentum = .8;
	// Perform the main training loop
	for (int iter = 0; iter < max_iter; iter++) {
        // Compute the main training loop
        #ifdef BENCHMARK
        start_ld_affinity = start_tsc();
        #endif
        float sum_Q = compute_low_dimensional_affinities(Y, N, no_dims, Q);
        #ifdef BENCHMARK
        cycles_ld_affinity += (double) stop_tsc(start_ld_affinity);
        #endif

		// Compute the main training loop
		#ifdef BENCHMARK
		start_gradient = start_tsc();
		#endif
		gradient_computation_update_normalize(Y, P, Q, sum_Q, N, no_dims, uY, momentum, eta);
		#ifdef BENCHMARK
		cycles_gradient += (double) stop_tsc(start_gradient);
		#endif

		// Switch momentum
		if (iter == 250) momentum = final_momentum;
        if (iter == 250) {
            // Stop early exageration (do not need to be benchmarked)
            float factor = 1.0/12.0;
            for (int i = 0; i < N * N; i++) P[i] *= factor;
        }
	}

	#ifdef BENCHMARK
	cycles += cycles_distance + cycles_perplexity + cycles_symmetrize +
        cycles_ld_affinity + cycles_gradient;
	#endif

	_mm_free(P); 		P = NULL;
	_mm_free(DD); 		DD = NULL;
	_mm_free(Q);		Q = NULL;
	_mm_free(mean);		mean = NULL;
    _mm_free(uY);       uY = NULL;
}

int main(int argc, char **argv) {
    /* Usage:
    data:       data file
    result:     result file
    N:          number of samples
    perp:       perplexity              (best value: 50)
    o_dim:      output dimensionality   (best value: 2)
    max_iter:   max iterations          (best value: 1000)
    inputDim:   input dimensionality of data; optional defaults to 784
    */

    #ifdef EXIT
    printf("Exiting. Specify some type of optimization: BASELINE, SCALAR, AVX\n");
    exit(1);
    #endif

    // Parse arguments
    char *data_file = argv[1];
    char *result_file = argv[2];
    int N = atoi(argv[3]);
    float perplexity = (float) atof(argv[4]);
    int no_dims = 2;
    int max_iter = atoi(argv[6]);
    int inputDim = 784;
    if (argc > 7)
    	inputDim = atoi(argv[7]);

	// Set random seed
    int rand_seed = 23;
    srand((unsigned int) rand_seed);

	#ifdef DEBUG
	printf("Data file: %s\n", data_file);
	printf("Result file: %s\n", result_file);
	printf("---------\nN = %d\n", N);
	printf("perplexity = %.2f\n", perplexity);
	printf("no_dims = %d\n", no_dims);
	printf("max_iter = %d\n", max_iter);
	printf("Using random seed: %d\n", rand_seed);
	printf("Input Dim (given): %d\n", inputDim);
	#endif

	// Define some variables
	int D;
	float* data = (float*) _mm_malloc(N * inputDim * sizeof(double),32);
	float* X = (float*) _mm_malloc(N * inputDim * sizeof(float),32);
    float* Y = (float*) _mm_malloc(N * no_dims * sizeof(float),32);
    if(data == NULL) { printf("[data] Memory allocation failed!\n"); exit(1); }
    if(X == NULL) { printf("[X] Memory allocation failed!\n"); exit(1); }
    if(Y == NULL ) { printf("[Y] Memory allocation failed!\n"); exit(1); }

	// Randomly intialize Y array
    for (int i = 0; i < N * no_dims; i++) {
        Y[i] = randn() * .0001;
    }

	#ifdef NUMERIC_CHECK
	save_data(Y, N, no_dims, "./datum/Y");
	#endif

	// Read the parameters and the dataset
    bool success = load_data(data, N, &D, data_file);
    if (!success) exit(1);

	#ifdef BENCHMARK
	int num_runs = NUM_RUNS;
	#else
	int num_runs = 1;
	#endif

	for (int i = 0; i < num_runs; i++) {
		// Randomly intialize arrays
		for (int i = 0; i < N * D; i++) 		X[i] = (float) data[i];
	    for (int i = 0; i < N * no_dims; i++) 	Y[i] = randn() * .0001;

		run(X, N, D, Y, no_dims, perplexity, max_iter);
	}

    #ifdef BENCHMARK
	cycles_distance /= (double) num_runs;
	cycles_perplexity /= (double) num_runs;
	cycles_symmetrize /= (double) num_runs;
	cycles_ld_affinity /= (double) num_runs;
	cycles_gradient /= (double) num_runs;

	cycles /= (double) num_runs;
	printf("%d,%lf,%lf,%lf,%lf,%lf,%lf\n", N,
		cycles_distance, cycles_perplexity, cycles_symmetrize,
        cycles_ld_affinity, cycles_gradient, cycles);
    #endif

	// Save the results
	save_data(Y, N, no_dims, result_file);

	// Clean up the memory
	_mm_free(data);   	data = NULL;
	_mm_free(X);   		X = NULL;
	_mm_free(Y);      	Y = NULL;
}
