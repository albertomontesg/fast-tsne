#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>

#ifdef COUNTING
size_t ITERS_insert = 0;
size_t ITERS_subdivide = 0;
#endif


#include "../utils/io.h"
#include "../utils/tsc_x86.h"
#include "../utils/random.h"
#include "../utils/data_type.h"
#include "../tsne_exact_optimizations/computations/compute_low_dimensional_affinities.h"
#include "computations/compute_pairwise_affinity_perplexity_nlogn.h"
#include "computations/early_exageration_nlogn.h"
#include "computations/gradient_computation_nlogn.h"
#include "../tsne_exact_optimizations/computations/gradient_update.h"
#include "../tsne_exact_optimizations/computations/normalize.h"
#include "computations/symmetrize_affinities_nlogn.h"
#include "trees/sptree.h"

#define NUM_RUNS 1

#ifdef BENCHMARK
double cycles = 0;
double cycles_normalize = 0, cycles_perplexity = 0, cycles_symmetrize = 0;
double cycles_early_exageration = 0, cycles_tree = 0;
double cycles_ld_affinity = 0, cycles_gradient = 0, cycles_update = 0;
double cycles_normalize_2 = 0;
myInt64 start_normalize, start_perplexity, start_symmetrize;
myInt64 start_early_exageration, start_ld_affinity, start_tree;
myInt64 start_gradient, start_update, start_normalize_2;
#endif


// Run function
void run(double* X, int N, int D, double* Y, int no_dims, double perplexity,
	 	 int max_iter, double theta) {

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
	#ifdef NUMERIC_CHECK
	printf("save X_normalized\n");
	save_data(X, N, D, "./datum/X_normalized");
	#endif
	free(mean); mean = NULL;


	// Compute pairsiwe affinity with perplexity which include binary search
	// for the best perplexity value
	const unsigned int K = min((unsigned int) (3 * perplexity),
								(unsigned int) N);
	dt* DD = (dt*) malloc(N * N * sizeof(dt));
	unsigned int *row_P = (unsigned int*)    malloc((N + 1) * sizeof(unsigned int));
    unsigned int *col_P = (unsigned int*)    calloc(N * K, sizeof(unsigned int));
    dt *val_P = (dt*) calloc(N * K, sizeof(dt));

	if(row_P == NULL) { printf("[row_P] Memory allocation failed!\n"); exit(1); }
	if(col_P == NULL) { printf("[col_P] Memory allocation failed!\n"); exit(1); }
	if(val_P == NULL) { printf("[val_P] Memory allocation failed!\n"); exit(1); }

	#ifdef DEBUG
	printf("compute_pairwise_affinity_perplexity_nlogn\n");
	#endif

	#ifdef BENCHMARK
	start_perplexity = start_tsc();
	#endif
	// Compute
	compute_pairwise_affinity_perplexity_nlogn(X, N, D,
									     val_P, row_P, col_P,
									     perplexity,
									     (unsigned int) (3 * perplexity));
	// End compute
	#ifdef BENCHMARK
	cycles_perplexity += (double) stop_tsc(start_perplexity);
	#endif
	// P and DD will not be remove because future use
	#ifdef NUMERIC_CHECK
	save_csr_data(row_P, col_P, val_P, N, N, "./datum/P_sparse");
	#endif

	#ifdef DEBUG
	printf("symmetrize_affinities_nlogn\n");
	#endif

	// Symmetrize affinities
	#ifdef BENCHMARK
	start_symmetrize = start_tsc();
	#endif
	// Compute
	symmetrize_affinities_nlogn(&row_P, &col_P, &val_P, N);
	// End compute
	#ifdef BENCHMARK
	cycles_symmetrize += (double) stop_tsc(start_symmetrize);
	#endif

	#ifdef NUMERIC_CHECK
	printf("saving P matrix\n");
	save_csr_data(row_P, col_P, val_P, N, N, "./datum/P_sparse");
	#endif


	// Early exageration
    #ifdef BENCHMARK
    start_early_exageration = start_tsc();
    #endif
    // Compute
    early_exageration_sparse(val_P, N*K, 12.0);
    // End compute
    #ifdef BENCHMARK
    cycles_early_exageration += (double) stop_tsc(start_early_exageration);
    #endif

	#ifdef DEBUG
	printf("Allocate memory\n");
	#endif

	// Initialize Q low dimensionality affinity matrix and gradient dC
	dt* Q 		= (dt*) malloc(N * N * sizeof(dt));
	dt* dC 		= (dt*) malloc(N * no_dims * sizeof(dt));
	dt* uY 		= (dt*) calloc(N * no_dims, sizeof(dt));
	dt* gains 	= (dt*) malloc(N * no_dims * sizeof(dt));
	mean 		= (dt*) malloc(no_dims * sizeof(dt));
	dt* pos_f 	= (dt*) malloc(N * D * sizeof(dt));
	dt* neg_f 	= (dt*) malloc(N * D * sizeof(dt));
	if(Q == NULL) { printf("[Q] Memory allocation failed!\n"); exit(1); }
	if(dC == NULL) { printf("[dC] Memory allocation failed!\n"); exit(1); }
	if(uY == NULL) { printf("[uY] Memory allocation failed!\n"); exit(1); }
	if(gains == NULL) {printf("[gains] Memory allocation failed!\n"); exit(1);}
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }
	if(pos_f == NULL) { printf("[pos_f] Memory allocation failed!\n"); exit(1);}
	if(neg_f == NULL) { printf("[neg_f] Memory allocation failed!\n"); exit(1);}
	for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

	#ifdef DEBUG
	printf("Memory allocated\n");
	#endif

	// Training parameters
	dt eta = 200.0;
	dt momentum = .5;
	dt final_momentum = .8;
	// Perform the main training loop
	for (int iter = 0; iter < max_iter; iter++) {
		#ifdef DEBUG
		if (iter % 50 == 0) printf(".");
		#endif

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

		// QuadTree creation

		#ifdef BENCHMARK
		start_tree = start_tsc();
		#endif
		// Compute
		// SPTree interface: (dimension, data, num_elements)
		SPTree* tree = new SPTree(no_dims, Y, N);
		// End compute
		#ifdef BENCHMARK
		cycles_tree += (double) stop_tsc(start_tree);
		#endif

		// Gradient Computation
		// Make sure the gradient contains all zeros
		for (int i = 0; i < N * no_dims; i++) dC[i] = 0.;
		for (int i = 0; i < N * D; i++) {
			pos_f[i] = 0.;
			neg_f[i] = 0.;
		}
		#ifdef BENCHMARK
		start_gradient = start_tsc();
		#endif
		// Compute
		gradient_computation_nlogn(Y, row_P, col_P, val_P, N, no_dims, dC, theta, pos_f, neg_f, tree);
		// End compute
		#ifdef BENCHMARK
		cycles_gradient += (double) stop_tsc(start_gradient);
		#endif
		#ifdef NUMERIC_CHECK
		if (iter == 0) save_data(dC, N, no_dims, "./datum/dC_0");
		if (iter == 300) save_data(dC, N, no_dims, "./datum/dC_300");
		#endif
		// Free tree
		delete tree;

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
            early_exageration_sparse(val_P, N*K, 1/12.0);
            // End compute
            #ifdef BENCHMARK
            cycles_early_exageration += (double) stop_tsc(start_early_exageration);
            #endif
        }
		// Switch momentum
		if (iter == 250) momentum = final_momentum;
	}


	#ifdef COUNTING
	printf("it_ins %ld\nit_sub %ld\n", ITERS_insert, ITERS_subdivide);
	#endif

	#ifdef BENCHMARK
	cycles += cycles_normalize + cycles_perplexity + cycles_symmetrize + cycles_early_exageration + cycles_ld_affinity + cycles_tree + cycles_gradient + cycles_update + cycles_normalize_2;
	#endif

	free(DD); 		DD = NULL;
	free(Q);		Q = NULL;
	free(dC);		dC = NULL;
	free(gains);	gains = NULL;
	free(mean);		mean = NULL;
	free(pos_f);	pos_f = NULL;
	free(neg_f);	neg_f = NULL;
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

    // Parse arguments
    char *data_file = argv[1];
    char *result_file = argv[2];
    int N = atoi(argv[3]);
    dt perplexity = (dt) atof(argv[4]);
    int no_dims = atoi(argv[5]);
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
	dt* data = (dt*) malloc(N * inputDim * sizeof(double));
	dt* X = (dt*) malloc(N * inputDim * sizeof(dt));
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
		run(data, N, D, Y, no_dims, perplexity, max_iter, 0.5);
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