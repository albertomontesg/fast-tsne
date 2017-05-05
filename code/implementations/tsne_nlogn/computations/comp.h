#include <math.h>
#include <float.h>
#include <cstddef>
#include <stdlib.h>
#include <vector>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// Normalize X substracting mean and
void normalize(double* X, int N, int D, double* mean, int max_value);
// Compute squared euclidean disctance for all pairs of vectors X_i X_j
void compute_squared_euclidean_distance(double* X, int N, int D, double* DD);
// Compute pairwise affinity perplexity
void compute_pairwise_affinity_perplexity(double* X, int N, int D, double* P,
										  double perplexity, double* DD);
void compute_pairwise_affinity_perplexity_nlogn(double* X, int N, int D, double* val_P,
										  unsigned int* row_P, unsigned int* col_P,
										  double perplexity, unsigned int K);
// Symmetrize pairwise affinities P_ij
void symmetrize_affinities(double* P, int N);
void symmetrize_affinities_nlogn(unsigned int** row_P, unsigned int** col_P, double** val_P, int N);
// Early exageration (Multiply all the values of P to the given value)
void early_exageration(double* P, int N, double scale);
void early_exageration_sparse(double* val_P, int no_elements, double scale);
// Compute low dimensional affinities
double compute_low_dimensional_affinities(double* Y, int N, int no_dims,
										  double* Q, double* DD);
// Gradient computation dC_dy
void gradient_computation(double* Y, unsigned int* row_P, unsigned int* col_P, double* val_p, int N,
						  int D, double* dC, double theta);
// Update gains and update Y with the computed gradient
void gradient_update(double* Y, double* dC, double* uY, double* gains, int N,
					 int no_dims, double momentum, double eta);
