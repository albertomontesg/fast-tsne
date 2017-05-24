#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../../utils/tsc_x86.h"
#include "../../utils/random.h"
#include "symmetrize_affinities.h"


#define NUM_RUNS    11
#define CYCLES_REQUIRED 1e5
#define N_START     8
#define N_STOP      8192
#define N_INTERVAL  2
#define EPS         1e-3

const int D = 2;


/* prototype of the function you need to optimize */
typedef void(*comp_func)(float *, int, float);

comp_func userFuncs[32];
char *funcNames[32];
int numFuncs = 0;

void add_function(comp_func f, char *name);

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() {
    add_function(&base_version, "base_version");
    // Add your functions here
    // add_function(&your_function, "function: Optimization X");
    //the number of flops should not change
    add_function(unrolling, "unrolling");
    add_function(blocking, "blocking");
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
    float *P, *P_copy; 
    float scale = 1.0/12.0;
    // Input:
    build(&P, n, n);
    build(&P_copy, n, n);
    std::copy(P, P+(n*n), P_copy);
    // Warm up the cache
    do {
        warm_runs = warm_runs * multiplier;
        std::copy(P_copy, P_copy+(n*n), P);
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(P, n, scale);
        }
        end = stop_tsc(start);

        cycles = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);
    std::vector<double> num_cycles(num_runs);

    for (size_t i = 0; i < num_runs; ++i) {
        // Put here the function
        std::copy(P_copy, P_copy+(n*n), P);
        for(int j=0; j<n*n; j++) P[j]*=1.001;
        start = start_tsc();
        f(P, n, scale);
        end = stop_tsc(start);
        num_cycles[i] = (double) end;
    }
    //destroy(P);
    //destroy(P_copy);
    // Why does this cause a double free error??
    
    std::sort(num_cycles.begin(), num_cycles.end());
    int pos = num_runs / 2 + 1;

    return num_cycles[pos];
}

int main(int argc, char **argv) {

    register_functions();
    int n_start = N_START, n_stop = N_STOP, n_interval = N_INTERVAL;


    // Check the correct output of the functions
    int N = 512;
    float *P1, *P2, *P_copy; float scale=1.0/12.0;
    build(&P1, N, N);
    P_copy = (float*) malloc( N * N * sizeof(float) );
    P2 = (float*) malloc( N * N * sizeof(float) );
    std::copy(P1, P1+(N*N), P_copy);
    comp_func base_f = userFuncs[0];
    base_f(P1, N, scale);

    double error = 0.;
    for (int i = 1; i < numFuncs; i++) {
        comp_func f = userFuncs[i];
        std::copy(P_copy, P_copy+(N*N), P2);
        f(P2, N, scale);
        for (int j = 0; j < N*N; j++) {
            error = fabs(P1[j] - P2[j]);
            if (error > EPS) {
                printf("\n");
                printf("ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d with n=%d\nError: %lf != %lf\n", funcNames[i], j, N, P1[j], P2[j]);
                printf("\n");
                exit(1);
            }
        }
    }
    //destroy(P1);
    //destroy(P_copy);
    //destroy(P2);
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
    
    return 0;
}
