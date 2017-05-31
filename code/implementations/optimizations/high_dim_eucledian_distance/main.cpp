#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include "../../utils/io.h"
#include "../../utils/tsc_x86.h"
#include "../../utils/random.h"
#include "../../utils/data_type.h"
#include "compute_squared_euclidean_distance.h"


#define NUM_RUNS    11
#define CYCLES_REQUIRED 1e5
#define N_START     8
// #define N_STOP      8192
#define N_STOP      129
#define N_INTERVAL  2
#define D_START     8
// #define D_STOP      8192
#define D_STOP      129
#define D_INTERVAL  2
#define EPS         1e-4


/* prototype of the function you need to optimize */
typedef void(*comp_func)(float *, int, int, float *, float*, int);

comp_func userFuncs[32];
char *funcNames[32];
int numFuncs = 0;

void add_function(comp_func f, char *name);

/*
* Called by the driver to register your functions
* Use add_function(func, description) to add your own functions
*/
void register_functions() {

    //scalar test
    #ifdef SCALAR
    add_function(&base_version, "base_version");
    add_function(&fast_scalar, "fast_scalar");
    add_function(&fast_scalar_start_matter, "fast_scalar_start_matter");
    add_function(&fast_scalar_8x8x8_big_unroll, "fast_scalar_8x8x8_big_unroll");
    #endif

    //vec test
    #ifndef SCALAR
    add_function(&base_version, "base_version");
    add_function(&fast_scalar, "fast_scalar");
    add_function(&fast_scalar_start_matter, "fast_scalar_start_matter");
    add_function(&fast_scalar_8x8x8_big_unroll, "fast_scalar_8x8x8_big_unroll");
    add_function(&fast_scalar_8x8x8_avx, "fast_scalar_8x8x8_avx");
    add_function(&fast_scalar_8x8x8_avx_with_start, "fast_scalar_8x8x8_avx_with_start");
    #endif

    // add_function(&base_version, "base_version");
    // add_function(&fast_scalar, "fast_scalar");
    // add_function(&fast_scalar_start_matter, "fast_scalar_start_matter");

    // // add_function(&fast_scalar_8x8_base, "fast_scalar_8x8_base");
    // add_function(&fast_scalar_avx, "fast_scalar_avx");
    // // add_function(&fast_scalar_avx_start_matter, "fast_scalar_avx_start_matter");

    // add_function(&fast_scalar_8x8x8_big_unroll, "fast_scalar_8x8x8_big_unroll");
    // // add_function(&fast_scalar_8x8x8_avx, "fast_scalar_8x8x8_avx");
    // add_function(&fast_scalar_8x8x8_avx_with_start, "fast_scalar_8x8x8_avx_with_start");
    // add_function(&fast_scalar_8x8x8_avx_with_start_more_block, "fast_scalar_8x8x8_avx_with_start_more_block");


    //not faster than normal fast_scaler
    // add_function(&fast_scalar_unroll_first, "fast_scalar_unroll_first");

    //
    // add_function(&fast_scalar_4x4_base, "fast_scalar_4x4_base");
    // add_function(&fast_scalar_8x8_8_base, "fast_scalar_8x8_8_base");
    // add_function(&fast_scalar_8x8_base_k, "fast_scalar_8x8_base_k");

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

double perf_test(comp_func f, int n, int D) {
    double cycles = 0;
    int num_runs = NUM_RUNS;
    int warm_runs = 10;
    myInt64 start, end;
    double multiplier = 1.0;


    // Create the input and output arrays
    float *Xc, *DD, *means;
    // Input:
    build(&Xc, n, D);
    float *X = (float *) malloc(n*D*sizeof(float));
    memcpy(X, Xc, n*D*sizeof(float));
    DD = (float *) calloc(n* n, sizeof(float));
    // build(&DD, n, n);
    means = (float *) calloc(D, sizeof(float));
    // Warm up the cache
    do {
        warm_runs = warm_runs * multiplier;
        memset(means, 0, D*sizeof(float));
        memcpy(X, Xc, n*D*sizeof(float));
        start = start_tsc();
        for (size_t i = 0; i < num_runs; i++) {
            f(X, n, D, DD, means, 1);
        }
        end = stop_tsc(start);

        cycles = (double) end;
        multiplier = (CYCLES_REQUIRED) / (cycles);

    } while (multiplier > 2);

    std::vector<double> num_cycles(num_runs);
    free(DD);
    for (size_t i = 0; i < num_runs; ++i) {
        // Put here the function
        memset(means, 0, D*sizeof(float));
        memcpy(X, Xc, n*D*sizeof(float));
        DD = (float *) calloc(n*n, sizeof(float));
        start = start_tsc();
        f(X, n, D, DD, means, 1);
        end = stop_tsc(start);
        free(DD);
        num_cycles[i] = (double) end;
    }

    std::sort(num_cycles.begin(), num_cycles.end());
    int pos = num_runs / 2 + 1;

    destroy(X);
    destroy(Xc);
    destroy(means);
    // destroy(DD);

    return num_cycles[pos];
}

int main(int argc, char **argv) {

    register_functions();
    int n_start = N_START, n_stop = N_STOP, n_interval = N_INTERVAL;
    int d_start = D_START, d_stop = D_STOP, d_interval = D_INTERVAL;


    // Check the correct output of the functions

    //nice uneven numbers, to see edgecase handling
    // int N = 513;
    // int D = 66;
    int N = 512;
    int D = 64;

    
    float *Xc, *DDr, *DDc, *meansc, *meansr;
    // DDr = (float *) calloc(N* N, sizeof(float));
    DDc = (float *) calloc(N* N, sizeof(float));
    build(&Xc, N, D);
    float *X = (float *) malloc(N*D*sizeof(float));
    float *Xr = (float *) malloc(N*D*sizeof(float));
    memcpy(X, Xc, N*D*sizeof(float));
    memcpy(Xr, Xc, N*D*sizeof(float));
    // build(&DDc, N, N);
    meansc = (float *) calloc(D, sizeof(float));
    comp_func base_f = userFuncs[0];
    base_f(Xc, N, D, DDc, meansc, 1);

    double error = 0.;
    for (int i = 1; i < numFuncs; i++) {
        comp_func f = userFuncs[i];
        // build(&DDr, N, N);
        // memset(DDr, 0, N*N*sizeof(float));
        DDr = (float *) calloc(N* N, sizeof(float));
        memcpy(Xr, X, N*D*sizeof(float));
        meansr = (float *) calloc(D, sizeof(float));
        f(Xr, N, D, DDr, meansr, 1);
        // for (int j = 0; j < D; j++)
        // {
        //     error = fabs(meansc[j] - meansr[j]);
        //     if (error > EPS) {
        //         printf("means ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d with n=%d\nError: %lf != %lf\n", funcNames[i], j, N, meansr[j], meansc[j]);
        //         exit(1);
        //     }
        // }
        // for (int j = 0; j < D*N; j++)
        // {
        //     error = fabs(Xr[j] - Xc[j]);
        //     if (error > EPS) {
        //         printf("X ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d with n=%d\nError: %lf != %lf\n", funcNames[i], j, N, Xr[j], Xc[j]);
        //         exit(1);
        //     }
        // }
        for (int j = 0; j < N*N; j++) {
            error = fabs(DDr[j] - DDc[j]);
            if (error > EPS) {
                printf("D ERROR!!!! the results for the \"%s\" function are different to the correct implementation at position %d with n=%d\nError: %lf != %lf\n", funcNames[i], j, N, DDr[j], DDc[j]);
                exit(1);
            }
        }
        destroy(DDr);
    }
    destroy(X);
    destroy(Xr);
    destroy(Xc);
    destroy(meansr);
    destroy(meansc);

    double cycles;
    printf("N,D");
    for (int i = 0; i < numFuncs; i++) printf(",%s", funcNames[i]);
    printf("\n");
    for (int n = n_start; n <= n_stop; n *= n_interval) {
    for (int d = d_start; d <= d_stop; d *= d_interval) {
        printf("%d,%d", n, d);

        for (int i = 0; i < numFuncs; i++) {
            cycles = perf_test(userFuncs[i], n,d);
            printf(",%lf", cycles);
        }
        printf("\n");
    }
    }
}
