#include <math.h>
#include <float.h>

// Define data type
#ifdef SINGLE_PRECISION
typedef float dt;
#define log_c(float) logf(float)
#define exp_c(float) expf(float)
#define fabs_c(float) fabsf(float)
#define sqrt_c(float) sqrtf(float)
#else
typedef double dt;
#define log_c(double) log(double)
#define exp_c(double) exp(double)
#define fabs_c(double) fabs(double)
#define sqrt_c(double) sqrt(double)
#endif


// Normalize X substracting mean and
void normalize(dt* X, int N, int D, dt* mean, int max_value);
// Compute squared euclidean disctance for all pairs of vectors X_i X_j
void compute_squared_euclidean_distance(dt* X, int N, int D, dt* DD);
// Compute pairwise affinity perplexity
void compute_pairwise_affinity_perplexity(dt* X, int N, int D, dt* P,
										  dt perplexity, dt* DD);
// Symmetrize pairwise affinities P_ij
void symmetrize_affinities(dt* P, int N);
// Early exageration (Multiply all the values of P to the given value)
void early_exageration(dt* P, int N, dt scale);
// Compute low dimensional affinities
dt compute_low_dimensional_affinities(dt* Y, int N, int no_dims,
										  dt* Q, dt* DD);
// Gradient computation dC_dy
void gradient_computation(dt* Y, dt* P, dt* Q, dt sum_Q,
						  int N, int D, dt* dC);
// Update gains and update Y with the computed gradient
void gradient_update(dt* Y, dt* dC, dt* uY, dt* gains, int N,
					 int no_dims, dt momentum, dt eta);
