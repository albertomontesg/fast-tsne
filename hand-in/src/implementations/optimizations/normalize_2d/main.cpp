#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../../utils/tsc_x86.h"
#include "../../utils/random.h"
#include "normalize.h"


#define NUM_RUNS    11
#define CYCLES_REQUIRED 1e6
#define N_START     4
#define N_STOP      16384
#define N_INTERVAL  2
#define EPS         1e-3
// #define MEDIAN

const int D = 2;


/* prototype of the function you need to optimize */

comp_func userFuncs[32];
char *funcNames[32];
int numFuncs = 0;

void add_function(comp_func f, char *name);

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() {
    add_function(&base_version,(char *) "base_version");
    // Add your functions here
    // add_function(&your_function, "function: Optimization X");
    //the number of flops should not change
    add_function(&unroll_accum4, (char *) "unroll_accum4");
    add_function(&unroll_accum8, (char *) "unroll_accum8");
    add_function(&unroll_accum8_vec, (char *) "unroll_accum8_vec");
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

void build(float ** d, int row, int col) {
    *d = (float*) malloc(row * col * sizeof(float));
    rand_matrix(*d, row, col);
}

void build_zeros(float ** d, int row, int col) {
    *d = (float*) calloc(row * col, sizeof(float));
}

void zero_diag(float *d, int N) {
    for (int i = 0; i < N; i++) d[i*N + i] = 0.;
}

float sum(float* M, int n) {
    float sum = 0.;
    for (int i = 0; i < n; i++) sum += M[i];
    return sum;
}

void copy(float* src, float* dst, int N) {
    for (int i = 0; i < N; i++) dst[i] = src[i];
}

void destroy(float* d) {
    free(d);
    d = NULL;
}

double perf_test(comp_func f, int n) {
    double cycles = 0;
    int num_runs = NUM_RUNS;
    int warm_runs = 10;
    myInt64 start, end;
    double multiplier = 1.0;


    // Create the input and output arrays
    float *Y;
    // Input:
    build(&Y, n, D);

    // Warm up the cache
    do {
        warm_runs = warm_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(Y, n, D);
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
        f(Y, n, D);
        end = stop_tsc(start);

        #ifdef MEDIAN
        num_cycles[i] = (double) end;
        #else
        cycles += (double) end;
        #endif
    }

    destroy(Y);

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
    float *Y_o, *Y_c, *Y_r;
    build(&Y_o, N, D);
    build(&Y_c, N, D);
    copy(Y_o, Y_c, N*D);


    comp_func base_f = userFuncs[0];
    base_f(Y_c, N, D);

    float error;

    for (int i = 1; i < numFuncs; i++) {
        comp_func f = userFuncs[i];
        build(&Y_r, N, D);
        copy(Y_o, Y_r, N*D);
        f(Y_r, N, D);

        for (int j = 0; j < N*D; j++) {
            error = fabs(Y_r[j] - Y_c[j]);
            if (error > (EPS * fabs(Y_c[j]))) {
                printf("ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d with n=%d\nError: %lf != %lf\n", funcNames[i], j, N, Y_r[j], Y_c[j]);
                exit(1);
            }
        }
        destroy(Y_r);
    }
    destroy(Y_c);
    destroy(Y_o);

    double cycles;
    printf("N");
    for (int i = 0; i < numFuncs; i++) printf(",%s", funcNames[i]);
    printf("\n");
    for (int n = n_start; n <= n_stop; n *= n_interval) {
        printf("%d", n);

        for (int i = 0; i < numFuncs; i++) {
            cycles = perf_test(userFuncs[i], n);
            printf(",%lf", cycles);
        }
        printf("\n");
    }
}
