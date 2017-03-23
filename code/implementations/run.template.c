/*
Template code to run as main and perform the benchmarking, debuging and flop
counting. Everything is parametrized with C Macros. The timing tool is the
rdtsc. */

#include "../utils/io.h"
#include "../utils/tsc_x86.h"
// Include the header of our own implementation
//#include "tsne.h"


#define DEBUG 1             // Printing process of the algorithm into stdout
#define COUNTING 0          // Count the flops while the algorithm runs
#define NUMERIC_CHECK 0     // Save values of vectors to check numerical comp.
#define BENCHMARK 1         // Count the cycles while running the algorithm
#define NUM_RUNS 2          // Number of times the algorithm run to count cycles

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


// Function that runs the Barnes-Hut implementation of t-SNE
int main(int argc, char **argv) {
    /* Usage:
    data:       data file
    result:     result file
    N:          number of samples
    theta:      gradient accuracy       (best value: .5)
    perp:       perplexity              (best value: 50)
    o_dim:      output dimensionality   (best value: 2)
    max_iter:   max iterations          (best value: 1000)
    */

    // Parse arguments
    char *data_file = argv[1];
    char *result_file = argv[2];
    int N = atoi(argv[3]);
    double theta = atof(argv[4]);
    double perplexity = atof(argv[5]);
    int no_dims = atoi(argv[6]);
    int max_iter = atoi(argv[7]);

    printf("Data file: %s\n", data_file);
    printf("Result file: %s\n", result_file);
    printf("---------\nN = %d\n", N);
    printf("theta = %.2f\n", theta);
    printf("perplexity = %.2f\n", perplexity);
    printf("no_dims = %d\n", no_dims);
    printf("max_iter = %d\n", max_iter);

    // Set random seed
    int rand_seed = 23;
    if(rand_seed >= 0) {
        printf("Using random seed: %d\n", rand_seed);
        srand((unsigned int) rand_seed);
    } else {
        printf("Using current time as random seed...\n");
        srand(time(NULL));
    }

    // Define some variables
	int D;
	double *data = (double*) malloc(N * 784 * sizeof(double));
    double* Y = (double*) malloc(N * no_dims * sizeof(double));
    double* costs = (double*) calloc(N, sizeof(double));
    if(data == NULL) { printf("[data] Memory allocation failed!\n"); exit(1); }
    if(costs == NULL) { printf("[costs] Memory allocation failed!\n"); exit(1); }
    if(Y == NULL ) { printf("[Y] Memory allocation failed!\n"); exit(1); }

    // Read the data
    bool success = load_data(data, N, &D, data_file);
    if (!success) exit(1);


#ifdef BENCHMARK
    double cycles;
    myInt64 start;
    start = start_tsc();
#endif
    int num_runs = NUM_RUNS;

    for (int i = 0; i < num_runs; i++) {
        // Run t-SNE algorithm
        // from tsne.h call the run function to run the algorithm
    	// run(data, N, D, Y, no_dims, perplexity, theta, false, max_iter);
    }

#ifdef BENCHMARK
    cycles = (double) stop_tsc(start) / num_runs;
    printf("RDTSC instruction:\n%lf cycles measured\n", cycles);
#endif

	// Save the results
	save_data(Y, costs, N, no_dims, result_file);

    // Clean up the memory
	free(data);   data = NULL;
	free(Y);      Y = NULL;
	free(costs);  costs = NULL;
}
