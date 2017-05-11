#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "../utils/io.h"
#include "../utils/tsc_x86.h"
#include "../utils/random.h"
#include "../utils/data_type.h"
#include "computations/compute_low_dimensional_affinities.h"
#include "computations/compute_pairwise_affinity_perplexity.h"
#include "computations/early_exageration.h"
#include "computations/gradient_computation.h"
#include "computations/gradient_update.h"
#include "computations/normalize.h"
#include "computations/symmetrize_affinities.h"


#define NUM_RUNS 1
//#define VALIDATION 1
//#define STORE_CALCULATION 1
//#define SINGLE_PRECISION 1

double cycles_normalize = 0;
myInt64 start_normalize;

char* merge_chars_int(const char* name, int number){
    char appended_no[10];
    sprintf(appended_no, "_%d", number);
    int len = strlen(name) + strlen(appended_no);

    char *fin = (char*)malloc(len);
    strncpy(fin, name, len);
    strncat(fin, appended_no, len);
    return fin;
}

// Run function
void run(dt* X, int N, int D, dt* Y, int no_dims, dt perplexity,
	 	 int max_iter) {
    #ifdef STORE_CALCULATION
        #ifdef SINGLE_PRECISION
            save_data(X, N, no_dims, merge_chars_int("./calculated/X_f_in", N));
        #else
            save_data(X, N, no_dims, merge_chars_int("./calculated/X_d_in", N));
        #endif
    #endif

    #ifdef VALIDATION
        #ifdef SINGLE_PRECISION
            if(validate_data(X, N, D, merge_chars_int("./reference/X_f_in", N))){
                printf("X vector matches input");
            }
        #else
            if(validate_data(X, N, D, merge_chars_int("./reference/X_f_in", N))){
                printf("X vector matches input");
            }
        #endif
    #endif

	// Normalize input X (substract mean and normalize to the maximum value
	// to avoid numerical inestabilities)
	dt* mean = (dt*) calloc(D, sizeof(dt));
	if(mean == NULL) { printf("[mean] Memory allocation failed!\n"); exit(1); }

	start_normalize = start_tsc();
	// Compute
	normalize(X, N, D, mean, 1);
	// End compute
	cycles_normalize += (double) stop_tsc(start_normalize);

    #ifdef STORE_CALCULATION
        #ifdef SINGLE_PRECISION
            save_data(X, N, no_dims, merge_chars_int("./calculated/X_f_out", N));
        #else
            save_data(X, N, no_dims, merge_chars_int("./calculated/X_d_out", N));
        #endif
    #endif

    #ifdef VALIDATION
        #ifdef SINGLE_PRECISION
            if(validate_data(X, N, D, merge_chars_int("./reference/X_f_out", N))){
                printf("X vector matches input");
            }
        #else
            if(validate_data(X, N, D, merge_chars_int("./reference/X_f_out", N))){
                printf("X vector matches input");
            }
        #endif
    #endif

	free(mean); mean = NULL;
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
    int N = atoi(argv[2]);
    dt perplexity = (dt) atof(argv[3]);
    int no_dims = atoi(argv[4]);
    int max_iter = atoi(argv[5]);

	// Set random seed
    int rand_seed = 23;
    srand((unsigned int) rand_seed);

	#ifdef DEBUG
	printf("Data file: %s\n", data_file);
	printf("---------\nN = %d\n", N);
	printf("perplexity = %.2f\n", perplexity);
	printf("no_dims = %d\n", no_dims);
	printf("max_iter = %d\n", max_iter);
	printf("Using random seed: %d\n", rand_seed);
	#endif

	// Define some variables
	int D;
	dt *data = (dt*) malloc(N * 784 * sizeof(dt));
	dt* X = (dt*) malloc(N * 784 * sizeof(dt));
    dt* Y = (dt*) malloc(N * no_dims * sizeof(dt));
    if(data == NULL) { printf("[data] Memory allocation failed!\n"); exit(1); }
    if(X == NULL) { printf("[X] Memory allocation failed!\n"); exit(1); }
    if(Y == NULL ) { printf("[Y] Memory allocation failed!\n"); exit(1); }

	// Randomly intialize Y array
    for (int i = 0; i < N * no_dims; i++) {
        Y[i] = randn() * .0001;
    }

	#ifdef STORE_CALCULATION
	save_data(Y, N, no_dims, "./calculated/Y");
	#endif

    #ifdef VALIDATION
    if(validate_data(X, N, D, merge_chars_int("./reference/Y", N))){
        printf("X vector matches input");
    }
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

	cycles_normalize /= (double) num_runs;
	printf("%lf \n", cycles_normalize);

	// Clean up the memory
	free(data);   	data = NULL;
	free(X);   		X = NULL;
	free(Y);      	Y = NULL;
}
