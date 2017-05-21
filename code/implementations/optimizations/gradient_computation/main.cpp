#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../../utils/tsc_x86.h"
#include "../../utils/random.h"
#include "gradient_computation.h"


#define NUM_RUNS    11
#define CYCLES_REQUIRED 1e5
#define N_START     4
#define N_STOP      8192
#define N_INTERVAL  2
#define EPS         1e-3
// #define MEDIAN

const int D = 2;


/* prototype of the function you need to optimize */
typedef void(*comp_func)(float *, float *, float *, float, int, int, float *);

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
    add_function(&unfold_d,(char *) "unfold_d");
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

float sum(float* M, int n) {
    float sum = 0.;
    for (int i = 0; i < n; i++) sum += M[i];
    return sum;
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
    float *Y, *P, *Q, *dC;
    // Input:
    build(&Y, n, D);
    build(&P, n, n);
    build(&Q, n, n);
    float sum_Q = sum(Q, n*n);
    build(&dC, n, D);

    // Warm up the cache
    do {
        warm_runs = warm_runs * multiplier;
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(Y, P, Q, sum_Q, n, D, dC);
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
        f(Y, P, Q, sum_Q, n, D, dC);
        end = stop_tsc(start);

        #ifdef MEDIAN
        num_cycles[i] = (double) end;
        #else
        cycles += (double) end;
        #endif
    }

    destroy(Y);
    destroy(P);
    destroy(Q);
    destroy(dC);

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
    int N = 512;
    float *Y, *P, *Q, *dC_c, *dC_r;
    build(&Y, N, D);
    build(&P, N, N);
    build(&Q, N, N);
    float sum_Q = sum(Q, N*N);
    build(&dC_c, N, D);     // Correct result from base_version

    comp_func base_f = userFuncs[0];
    base_f(Y, P, Q, sum_Q, N, D, dC_c);

    float error;

    for (int i = 1; i < numFuncs; i++) {
        comp_func f = userFuncs[i];
        build(&dC_r, N, D);     // Result from any other function
        f(Y, P, Q, sum_Q, N, D, dC_r);

        for (int j = 0; j < N*N; j++) {
            error = fabs(dC_r[j] - dC_c[j]);
            if (error > (EPS * fabs(dC_c[j]))) {
                printf("ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d with n=%d\nError: %lf != %lf\n", funcNames[i], j, N, dC_r[j], dC_c[j]);
                exit(1);
            }
        }
        destroy(dC_r);
    }
    destroy(Y);
    destroy(P);
    destroy(Q);
    destroy(dC_c);

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
