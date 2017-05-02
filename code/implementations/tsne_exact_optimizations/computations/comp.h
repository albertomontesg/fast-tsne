#include <math.h>
#include <float.h>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// Normalize X substracting mean and
void normalize(double* X, int N, int D, double* mean, int max_value);
// Compute squared euclidean disctance for all pairs of vectors X_i X_j
void compute_squared_euclidean_distance(double* X, int N, int D, double* DD);
// Compute pairwise affinity perplexity
void compute_pairwise_affinity_perplexity(double* X, int N, int D, double* P,
										  double perplexity, double* DD);
// Symmetrize pairwise affinities P_ij
void symmetrize_affinities(double* P, int N);
// Early exageration (Multiply all the values of P to the given value)
void early_exageration(double* P, int N, double scale);
// Compute low dimensional affinities
double compute_low_dimensional_affinities(double* Y, int N, int no_dims,
										  double* Q, double* DD);
// Gradient computation dC_dy
void gradient_computation(double* Y, double* P, double* Q, double sum_Q,
						  int N, int D, double* dC);
// Update gains and update Y with the computed gradient
void gradient_update(double* Y, double* dC, double* uY, double* gains, int N,
					 int no_dims, double momentum, double eta);
