#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../utils/io.h"
#include "../../utils/tsc_x86.h"
#include "../../utils/random.h"
#include "../../utils/data_type.h"
#include "compute_squared_euclidean_distance.h"

#define NUM_RUNS    10
#define CYCLES_REQUIRED 1e6
#define N_START     200
#define N_STOP      2000
#define N_INTERVAL  200
#define EPS         1e-5

int D = 28*28;

/* prototype of the function you need to optimize */
typedef void(*comp_func)(dt *, int, int, dt *);

comp_func userFuncs[32];
char *funcNames[32];
int numFuncs = 0;

void add_function(comp_func f, char *name);

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() {
    add_function(&base_version, "Base Version");
    // Add your functions here
    // add_function(&your_function, "function: Optimization X");
    //the number of flops should not change
    add_function(&compute_squared_euclidean_distance, "compute_squared_euclidean_distance");
}

/*
* Registers a user function to be tested by the driver program. Registers a
* string description of the function as well
*/
void add_function(comp_func f, char *name)
{
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

void build(dt ** d, int row, int col) {
    *d = (dt*) malloc(row * col * sizeof(dt));
    rand_matrix(*d, row, col);
}

void destroy(dt* d) {
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
    dt *X, *DD;
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

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

      } while (multiplier > 2);


    start = start_tsc();
    for (size_t i = 0; i < num_runs; ++i) {
        // Put here the function
        f(X, n, D, DD);
    }
    end = stop_tsc(start);

    cycles = ( (double) end ) / num_runs;


    destroy(X);
    destroy(DD);

    return cycles;
}

int main(int argc, char **argv) {

    register_functions();
    int n_start = N_START, n_stop = N_STOP, n_interval = N_INTERVAL;


    // Check the correct output of the functions
    int N = 600;
    dt *X, *DD, *DD_correct;
    build(&X, N, D);
    build(&DD_correct, N, N);
    build(&DD, N, N);
    comp_func base_f = userFuncs[0];
    base_f(X, N, D, DD_correct);

    double error = 0.;
    for (int i = 1; i < numFuncs; i++) {
        comp_func f = userFuncs[i];
        f(X, N, D, DD);
        for (int j = 0; j < N*N; j++) {
            error = fabs(DD[j] - DD_correct[j]);
            if (error > EPS) {
                printf("ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d\n", funcNames[i], j);
                exit(1);
            }
        }
    }

    double cycles;
    printf("N");
    for (int i = 0; i < numFuncs; i++) printf(",%s", funcNames[i]);
    printf("\n");
    for (int n = n_start; n <= n_stop; n += n_interval) {
        printf("%d", n);

        for (int i = 0; i < numFuncs; i++) {
            cycles = perf_test(userFuncs[i], n);
            printf(",%lf", cycles);
        }
        printf("\n");
    }
}
