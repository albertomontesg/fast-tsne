#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../../utils/tsc_x86.h"
// #include "../../utils/random.h"
#include "compute_low_dimensional_affinities.h"


#define NUM_RUNS    31
#define CYCLES_REQUIRED 1e5
#define N_START     500
#define N_STOP      10000
#define N_INTERVAL  500
#define EPS         1e-3
#define MEDIAN

const int D = 2;


/* prototype of the function you need to optimize */
typedef float(*comp_func)(float *, int, int, float *);

comp_func userFuncs[32];
char *funcNames[32];
int numFuncs = 0;

void add_function(comp_func f, char *name);

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() {
    add_function(&base_version, (char *) "base_version");
    // Add your functions here
    // add_function(&your_function, "function: Optimization X");
    //the number of flops should not change
    add_function(&fused, (char *) "fused");
    add_function(&unfold_sr, (char *) "unfold_sr");
    add_function(&blocking, (char *) "blocking");
    add_function(&blocking_avx, (char *) "blocking_avx");
    add_function(&blocking2_avx, (char *) "blocking2_avx");
    // add_function(&blocking_32_block_8_unfold_sr_vec_2, (char *) "blocking_32_block_8_unfold_sr_vec_2");
}

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(comp_func f, char *name) {
  if (numFuncs >= 32)
  {
    printf("Couldn't register %s, too many functions registered (Max: %d)",
      name, 32);
    return;
  }

  userFuncs[numFuncs] = f;
  funcNames[numFuncs] = name;

  numFuncs++;
}

void rand_matrix(float* m, size_t row, size_t col) {
	for (size_t i = 0; i < row*col; i++) {
		m[i] = static_cast<float>(rand() + 1) / RAND_MAX;
	}
}

void build(float ** d, int row, int col) {
    *d = (float*) _mm_malloc(row * col * sizeof(float), 32);
    rand_matrix(*d, row, col);
}

void destroy(float* d) {
    _mm_free(d);
    d = NULL;
}

double perf_test(comp_func f, int n) {
    double cycles = 0;
    int num_runs = NUM_RUNS;
    int warm_runs = 10;
    myInt64 start, end;
    double multiplier = 1.0;


    // Create the input and output arrays
    float *X, *DD;
    // Input:
    build(&X, n, D);
    build(&DD, n, n);

    // Warm up the cache
    do {
        warm_runs = warm_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(X, n, D, DD);
        }
        end = stop_tsc(start);

        cycles = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);

    #ifdef MEDIAN
    std::vector<double> num_cycles(num_runs);
    #else
    cycles = 0;
    #endif

    for (size_t i = 0; i < num_runs; ++i) {
        // Put here the function
        start = start_tsc();
        f(X, n, D, DD);
        end = stop_tsc(start);

        #ifdef MEDIAN
        num_cycles[i] = (double) end;
        #else
        cycles += (double) end;
        #endif
    }

    destroy(X);
    destroy(DD);

    #ifdef MEDIAN
    std::sort(num_cycles.begin(), num_cycles.end());
    int pos = num_runs / 2 + 1;
    return num_cycles[pos];
    #else
    cycles /= (double) num_runs;
    return cycles;
    #endif
}

int main(int argc, char **argv) {

    register_functions();
    int n_start = N_START, n_stop = N_STOP, n_interval = N_INTERVAL;


    // Check the correct output of the functions
    int N = 527;
    float *X, *DDr, *DDc;
    build(&X, N, D);
    build(&DDc, N, N);
    comp_func base_f = userFuncs[0];
    float sum_qc = base_f(X, N, D, DDc);

    float error, error_sum;
    float sum_qr;
    for (int i = 1; i < numFuncs; i++) {
        comp_func f = userFuncs[i];
        build(&DDr, N, N);
        sum_qr = f(X, N, D, DDr);
        error_sum = fabs(sum_qr - sum_qc);
        if (error_sum > (EPS * fabs(sum_qc))) {
            printf("Error!! the sum of the Q matrix is different from the correct implementations in \"%s\": %f != %f\n", funcNames[i], sum_qr, sum_qc);
        }

        for (int j = 0; j < N*N; j++) {
            error = fabs(DDr[j] - DDc[j]);
            int row = j / N;
            int col = j - row * N;
            if (error > (EPS * fabs(DDc[j]))) {

                printf("ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d = (%d,%d) with n=%d\nError: %lf != %lf\n", funcNames[i], j, row, col, N, DDr[j], DDc[j]);
                exit(1);
            }
        }
        destroy(DDr);
    }
    destroy(X);
    destroy(DDc);

    double cycles;
    printf("N");
    for (int i = 0; i < numFuncs; i++) printf(",%s", funcNames[i]);
    printf("\n");
    for (int n = n_start; n <= n_stop; n += n_interval) {
        printf("%d", n);

        for (int i = 0; i < numFuncs; i++) {
            cycles = perf_test(userFuncs[i], n);
            printf(",%lf", cycles);
            fflush(stdout);
        }
        printf("\n");
    }
}
