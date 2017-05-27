#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>

#include "../utils/io.h"
#include "../utils/tsc_x86.h"
#include "../utils/random.h"
// #include "../utils/data_type.h"
// #include "computations/compute_low_dimensional_affinities.h"
// #include "computations/compute_pairwise_affinity_perplexity.h"
// #include "computations/compute_squared_euclidean_distance_normalize_high_dim_optimized.h"
// #include "computations/early_exageration.h"
// #include "computations/gradient_computation.h"
// #include "computations/gradient_update.h"
// #include "computations/normalize.h"
// #include "computations/symmetrize_affinities.h"
// #include "computations/training_step.h"

#ifdef BASELINE
#include "computation_baseline/compute_low_dimensional_affinities.h"
#include "computation_baseline/gradient_computation_update_normalize.h"

#elseif SCALAR
#include "computation_scalar/compute_low_dimensional_affinities.h"
#include "computation_scalar/gradient_computation_update_normalize.h"

#elseif AVX
#include "computation_avx/compute_low_dimensional_affinities.h"
#include "computation_avx/gradient_computation_update_normalize.h"

#else
#define EXIT
#endif


#define NUM_RUNS 1
#define SINGLE_PRECISION

#ifdef BENCHMARK
double cycles = 0;
double cycles_perplexity = 0, cycles_symmetrize = 0;
double cycles_ld_affinity = 0, cycles_gradient = 0;
myInt64 start_perplexity, start_symmetrize;
myInt64 start_ld_affinity, start_gradient;
#endif


bool load_data(float* data, int n, int* d, char* data_file) {

	// Open file, read first 2 integers, allocate memory, and read the data
    FILE *h;
	if((h = fopen(data_file, "r+b")) == NULL) {
		printf("Error: could not open data file.\n");
		return false;
	}

    int magic_number, origN, rows, cols;
    fread(&magic_number, sizeof(int), 1, h);
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        printf("Invalid magic number, it should be 2051, instead %d found\n", magic_number);
        return false;
    }
    fread(&origN, sizeof(uint32_t), 1, h);
    origN = reverse_int(origN);
    #ifdef DEBUG
    printf("Number of samples available: %d\n", origN);
    #endif
    if (n > origN) {
        printf("N can not be larger than the number of samples in the data file (%d)\n", origN);
        return false;
    }

    fread(&rows, sizeof(int), 1, h);
    fread(&cols, sizeof(int), 1, h);
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    *d = rows * cols;

    // Read the data
    unsigned char* raw_data = (unsigned char*) malloc(*d * n * sizeof(uint8_t));
    if(data == NULL) { printf("[data] Memory allocation failed!\n"); exit(1); }
    if(raw_data == NULL) { printf("[raw_data] Memory allocation failed!\n"); exit(1); }

    // Read raw data into unsigned char vector
    fread(raw_data, sizeof(uint8_t), n * *d, h);

    for (int i = 0; i < *d * n; i++) {
        data[i] = ((float) raw_data[i]) / 255.;
    }

    // Showing image
    if (false) {
        int offset = 10;
        printf("Sample %d\n\n", offset+1);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double v = data[offset * *d + i*cols+j];
                if (v > 0.)
                    printf("\x1B[31m %.2f \x1B[0m\t", v);
                else printf("%.2f\t", v);
            }
            printf("\n");
        }
    }
    free(raw_data); raw_data = NULL;
    return true;
}


// Run function
void run(float* X, int N, int D, float* Y, int no_dims, float perplexity,
	 	 int max_iter) {

	// Compute pairsiwe affinity with perplexity which include binary search
	// for the best perplexity value
	float* P = (float*) calloc(N * N, sizeof(float));
	float* DD = (float*) _mm_malloc(N * N * sizeof(float),32);
	if(P == NULL) { printf("[P] Memory allocation failed!\n"); exit(1); }
	if(DD == NULL) { printf("[DD] Memory allocation failed!\n"); exit(1); }
	#ifdef BENCHMARK
	start_perplexity = start_tsc();
	#endif
	// Compute
	compute_pairwise_affinity_perplexity_opt_eucledian(X, N, D, P, perplexity, DD);
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
	symmetrize_affinities(P, N);
	// End compute
	#ifdef BENCHMARK
	cycles_symmetrize += (double) stop_tsc(start_symmetrize);
	#endif
	#ifdef NUMERIC_CHECK
	save_data(P, N, N, "./datum/P_sym");
	#endif


	// Early exageration
    // #ifdef BENCHMARK
    // start_early_exageration = start_tsc();
    // #endif
    // // Compute
    // early_exageration(P, N, 12.0);
    // // End compute
    // #ifdef BENCHMARK
    // cycles_early_exageration += (double) stop_tsc(start_early_exageration);
    // #endif


	// Initialize Q low dimensionality affinity matrix and gradient dC
	float* Q = (float*) _mm_malloc(N * N * sizeof(float),32);
	float* uY = (float*) calloc(N * no_dims, sizeof(float));
	float* mean = (float*) _mm_malloc(no_dims * sizeof(float), 32);
	if(Q == NULL) { printf("[Q] Memory allocation failed!\n"); exit(1); }
	if(uY == NULL) { printf("[uY] Memory allocation failed!\n"); exit(1); }
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }

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
        start_training = start_tsc();
        #endif
        float sum_Q = compute_low_dimensional_affinities(Y, N, no_dims, Q);
        #ifdef BENCHMARK
        cycles_training += (double) stop_tsc(start_training);
        #endif

		// Compute the main training loop
		#ifdef BENCHMARK
		start_training = start_tsc();
		#endif
		gradient_computation_update_normalize(Y, P, Q, sum_Q, N, no_dims, uY, momentum, eta);
		#ifdef BENCHMARK
		cycles_training += (double) stop_tsc(start_training);
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
	cycles += cycles_normalize + cycles_perplexity + cycles_symmetrize + cycles_early_exageration + cycles_ld_affinity + cycles_gradient;
	#endif

	free(P); 		P = NULL;
	free(DD); 		DD = NULL;
	free(Q);		Q = NULL;
	free(mean);		mean = NULL;
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

	#ifdef DEBUG
	int num_runs = 1;
	#else
	int num_runs = NUM_RUNS;
	#endif

	for (int i = 0; i < num_runs; i++) {
		// Randomly intialize arrays
		for (int i = 0; i < N * D; i++) 		X[i] = (float) data[i];
	    for (int i = 0; i < N * no_dims; i++) 	Y[i] = randn() * .0001;

		run(X, N, D, Y, no_dims, perplexity, max_iter);
	}

    #ifdef BENCHMARK
	cycles_normalize /= (double) num_runs;
	cycles_perplexity /= (double) num_runs;
	cycles_symmetrize /= (double) num_runs;
    cycles_early_exageration /= (double) num_runs;
	cycles_ld_affinity /= (double) num_runs;
	cycles_gradient /= (double) num_runs;

	cycles /= (double) num_runs;
	printf("%lf %lf %lf %lf %lf %lf %lf\n", cycles_normalize,
		cycles_perplexity, cycles_symmetrize, cycles_early_exageration,
        cycles_training, cycles);
    #endif

	// Save the results
	save_data(Y, N, no_dims, result_file);

	// Clean up the memory
	free(data);   	data = NULL;
	free(X);   		X = NULL;
	free(Y);      	Y = NULL;
}
