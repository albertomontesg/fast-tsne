#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "../utils/io.h"
#include "../utils/tsc_x86.h"
#include "computations/comp.h"


#define NUM_RUNS 1

#ifdef BENCHMARK
double cycles = 0;
double cycles_normalize = 0, cycles_perplexity = 0, cycles_symmetrize = 0;
double cycles_early_exageration = 0;
double cycles_ld_affinity = 0, cycles_gradient = 0, cycles_update = 0;
double cycles_normalize_2 = 0;
myInt64 start_normalize, start_perplexity, start_symmetrize;
myInt64 start_early_exageration, start_ld_affinity;
myInt64 start_gradient, start_update, start_normalize_2;
#endif

// Generates a Gaussian random number
dt randn() {
	dt x, y, radius;
	do {
		x = 2 * (rand() / ((dt) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((dt) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt_c(-2 * log_c(radius) / radius);
	x *= radius;
	return x;
}


// Run function
void run(dt* X, int N, int D, dt* Y, int no_dims, dt perplexity,
	 	 int max_iter) {

    #ifdef NUMERIC_CHECK
    save_data(X, N, D, "./datum/X");
    #endif

	// Normalize input X (substract mean and normalize to the maximum value
	// to avoid numerical inestabilities)
	dt* mean = (dt*) calloc(D, sizeof(dt));
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }
	#ifdef BENCHMARK
	start_normalize = start_tsc();
	#endif
	// Compute
	normalize(X, N, D, mean, 1);
	// End compute
	#ifdef BENCHMARK
	cycles_normalize += (double) stop_tsc(start_normalize);
	#endif

	free(mean); mean = NULL;


	// Compute pairsiwe affinity with perplexity which include binary search
	// for the best perplexity value
	dt* P = (dt*) calloc(N * N, sizeof(dt));
	dt* DD = (dt*) malloc(N * N * sizeof(dt));
	if(P == NULL) { printf("[P] Memory allocation failed!\n"); exit(1); }
	if(DD == NULL) { printf("[DD] Memory allocation failed!\n"); exit(1); }
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


	// Symmetrize affinities
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
    #ifdef BENCHMARK
    start_early_exageration = start_tsc();
    #endif
    // Compute
    early_exageration(P, N, 12.0);
    // End compute
    #ifdef BENCHMARK
    cycles_early_exageration += (double) stop_tsc(start_early_exageration);
    #endif


	// Initialize Q low dimensionality affinity matrix and gradient dC
	dt* Q = (dt*) malloc(N * N * sizeof(dt));
	dt* dC = (dt*) malloc(N * no_dims * sizeof(dt));
	dt* uY = (dt*) calloc(N * no_dims, sizeof(dt));
	dt* gains = (dt*) malloc(N * no_dims * sizeof(dt));
	mean = (dt*) malloc(no_dims * sizeof(dt));
	if(Q == NULL) { printf("[Q] Memory allocation failed!\n"); exit(1); }
	if(dC == NULL) { printf("[dC] Memory allocation failed!\n"); exit(1); }
	if(uY == NULL) { printf("[uY] Memory allocation failed!\n"); exit(1); }
	if(gains == NULL) {printf("[gains] Memory allocation failed!\n"); exit(1);}
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

	// As the counting is perform before entering the main training loop
	// it is not necessary to run it, so time is saved to count the
	// iterations
	#ifdef COUNTING
	max_iter = 0;
	#endif

	// Training parameters
	dt eta = 200.0;
	dt momentum = .5;
	dt final_momentum = .8;
	// Perform the main training loop
	for (int iter = 0; iter < max_iter; iter++) {

		// Compute low dimensional affinities
		// Reset DD to all 0s
		for (int i = 0; i < N * N; i++) DD[i] = 0.;
		#ifdef BENCHMARK
		start_ld_affinity = start_tsc();
		#endif
		// Compute
		dt sum_Q;
		sum_Q = compute_low_dimensional_affinities(Y, N, no_dims, Q, DD);
		// End compute
		#ifdef BENCHMARK
		cycles_ld_affinity += (double) stop_tsc(start_ld_affinity);
		#endif
		#ifdef NUMERIC_CHECK
		if (iter == 0) printf("%lf\n", sum_Q);
		if (iter == 0) save_data(Q, N, N, "./datum/Q_0");
		if (iter == 300) save_data(Q, N, N, "./datum/Q_300");
		#endif


		// Gradient Computation
		// Make sure the gradient contains all zeros
		for (int i = 0; i < N * no_dims; i++) dC[i] = 0.;
		#ifdef BENCHMARK
		start_gradient = start_tsc();
		#endif
		// Compute
		gradient_computation(Y, P, Q, sum_Q, N, no_dims, dC);
		// End compute
		#ifdef BENCHMARK
		cycles_gradient += (double) stop_tsc(start_gradient);
		#endif
		#ifdef NUMERIC_CHECK
		if (iter == 0) save_data(dC, N, no_dims, "./datum/dC_0");
		if (iter == 300) save_data(dC, N, no_dims, "./datum/dC_300");
		#endif


		// Perform gradient update
		#ifdef BENCHMARK
		start_update = start_tsc();
		#endif
		// Compute
		gradient_update(Y, dC, uY, gains, N, no_dims, momentum, eta);
		// End compute
		#ifdef BENCHMARK
		cycles_update += (double) stop_tsc(start_update);
		#endif
		#ifdef NUMERIC_CHECK
		if (iter == 0) save_data(Y, N, no_dims, "./datum/Y_0");
		if (iter == 300) save_data(Y, N, no_dims, "./datum/Y_300");
		#endif

		// Zero mean to solution Y
		for (int i = 0; i < no_dims; i++) mean[i] = 0.;
        #ifdef BENCHMARK
        start_normalize_2 = start_tsc();
        #endif
		// Compute
		normalize(Y, N, no_dims, mean, 0);
		// End compute
        #ifdef BENCHMARK
        cycles_normalize_2 += (double) stop_tsc(start_normalize_2);
        #endif
		#ifdef NUMERIC_CHECK
		if (iter == 0) save_data(Y, N, no_dims, "./datum/Y_0_normalized");
		if (iter == 300) save_data(Y, N, no_dims, "./datum/Y_300_normalized");
		#endif

        if (iter == 250) {
            // Stop early exageration
            #ifdef BENCHMARK
            start_early_exageration = start_tsc();
            #endif
            // Compute
            early_exageration(P, N, 1/12.0);
            // End compute
            #ifdef BENCHMARK
            cycles_early_exageration += (double) stop_tsc(start_early_exageration);
            #endif
        }
		// Switch momentum
		if (iter == 250) momentum = final_momentum;
	}

	#ifdef BENCHMARK
	cycles += cycles_normalize + cycles_perplexity + cycles_symmetrize + cycles_early_exageration + cycles_ld_affinity + cycles_gradient + cycles_update + cycles_normalize_2;
	#endif

	free(P); 		P = NULL;
	free(DD); 		DD = NULL;
	free(Q);		Q = NULL;
	free(dC);		dC = NULL;
	free(gains);	gains = NULL;
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
    */

    // Parse arguments
    char *data_file = argv[1];
    char *result_file = argv[2];
    int N = atoi(argv[3]);
    dt perplexity = (dt) atof(argv[4]);
    int no_dims = atoi(argv[5]);
    int max_iter = atoi(argv[6]);

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
	#endif

	// Define some variables
	int D;
	double *data = (double*) malloc(N * 784 * sizeof(double));
	dt* X = (dt*) malloc(N * 784 * sizeof(dt));
    dt* Y = (dt*) malloc(N * no_dims * sizeof(dt));
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
		for (int i = 0; i < N * D; i++) 		X[i] = (dt) data[i];
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
	cycles_update /= (double) num_runs;
	cycles_normalize_2 /= (double) num_runs;
	cycles /= (double) num_runs;
	printf("%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", cycles_normalize,
		cycles_perplexity, cycles_symmetrize, cycles_early_exageration,
        cycles_ld_affinity, cycles_gradient, cycles_update,
        cycles_normalize_2, cycles);
    #endif

	// Save the results
	save_data(Y, N, no_dims, result_file);

	// Clean up the memory
	free(data);   	data = NULL;
	free(X);   		X = NULL;
	free(Y);      	Y = NULL;
}
