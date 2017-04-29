#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

const size_t D = 3; //dimensionality of data
const size_t d = 2; //dimensionality of output
const size_t N = 5; //number of data points
const size_t T = 1000; //number of iterations
const double minSigmaSq = 10e-4;
const double maxSigmaSq = 10e4;
const double sigmaSqStart = (minSigmaSq+maxSigmaSq)/2;
const double targetPerp = 2;
const double eta = 0.9;
const size_t maxIterations = 200;

void calcHP(size_t i, size_t N, double* distances, double* betas, double* h, double *p)
{
	double sum = 0;
	for (size_t j = 0; j < N; ++j)
	{
		p[i * N + j] = expf(- distances[i * N + j] * betas[i]);
		sum += p[i * N + j];
	}
	double sumDP = 0;
	for (size_t j = 0; j < N; ++j)
	{
		sumDP += p[i * N + j] * distances[i * N + j];
	}
	h[i] = log(sum) + sumDP/sum;
}


void binary_search_p_for_perplexity(double *X, size_t N, size_t D, double *p, double tol = 1e-5, double perplexity = 30.0)
{
	double sigmaMean = 0;
	const double logU = logf(perplexity)/logf(2.0);
	double *h = (double *) calloc(N, sizeof(double));
	double *betas = (double*) malloc(N * sizeof(double));
	memset(betas, 1.0, N); //TODO does this work?
	double *distances = (double *) calloc(N*N, sizeof(double));
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			for (int k = 0; k < D; ++k)
			{
				//TODO: use symitry
				const double d = X[i*k + k] - X[j*k + k];
				distances[i * N + j] += d*d;
			}
		}
	}

	for (int i = 0; i < N; ++i)
	{
		double betamin = -INFINITY	;
		double betamax =  INFINITY;
		//all distances but self iteractons
		// Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
		calcHP(i, N, distances, betas, h, p);
		double hdiff = h[i] - logU;
		for (size_t tries = 0; (i < 50) && fabs(hdiff) > tol; ++tries)
		{
			if (hdiff > 0)
			{
				betamin = betas[i];
				if (betamax == INFINITY || betamax == -INFINITY)
					betas[i] = betas[i] * 2;
				else
					betas[i] = (betas[i] + betamax) / 2;
			} else
			{
				betamax = betas[i];
				if (betamin == INFINITY || betamin == -INFINITY)
					betas[i] = betas[i] / 2;
				else
					betas[i] = (betas[i] + betamin) / 2;
			}
			
			//Recompute the values
			calcHP(i, N, distances, betas, h, p);
			hdiff = h[i] - logU;
			sigmaMean += sqrt(1/betas[i]);
		}
	}
	sigmaMean /= N;
	printf("Mean value of sigma: %lf", sigmaMean);
}



double calcPij(double* from, double* to, size_t n, double sigmaSq)
{
	double const fact = 2*sigmaSq;
	double denom = 0;
	double perplexity = 0;
	for(size_t k = 0; k < N; ++k)
	{
		to[k] = exp(-from[k]/ fact);
		if (k != n) denom += to[k];
	}
	for(size_t k = 0; k < N; ++k)
	{
		to[k] = to[k]/denom;
		perplexity -= to[k] * log(to[k]);
	}
	perplexity = pow(2, perplexity);
	return perplexity;
}

// code from https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
double randn (double mu, double sigma)
{
  double U1, U2, W, mult;
  static double X1, X2;
  static int call = 0;
 
  if (call == 1)
    {
      call = !call;
      return (mu + sigma * (double) X2);
    }
 
  do
    {
      U1 = -1 + ((double) rand () / RAND_MAX) * 2;
      U2 = -1 + ((double) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    }
  while (W >= 1 || W == 0);
 
  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;
 
  call = !call;
 
  return (mu + sigma * (double) X1);
}


int main(int argc, char const *argv[])
{

	double *data;

	//setup data
	data = (double *) malloc(D*N*sizeof(double));
	
	printf("initializing data\n");
	data[0*3 + 0] = 3.0;
	data[0*3 + 1] = 2.0;
	data[0*3 + 2] = 1.0;

	data[1*3 + 0] = 1.0;
	data[1*3 + 1] = 2.0;
	data[1*3 + 2] = 3.0;

	data[2*3 + 0] = 9.0;
	data[2*3 + 1] = 0.0;
	data[2*3 + 2] = 0.0;

	data[3*3 + 0] = 0.0;
	data[3*3 + 1] = 10.0;
	data[3*3 + 2] = 0.0;

	data[4*3 + 0] = 0.0;
	data[4*3 + 1] = 0.0;
	data[4*3 + 2] = 25.0;

	double *p = (double *) calloc(N*N, sizeof(double));
	double *iY = (double *) calloc(N*d, sizeof(double));
	double *yd = (double *) malloc(N*d*sizeof(double));
	double pSum = 0;
	binary_search_p_for_perplexity(data, N, D, p);
	// p = p + p.T
	for (size_t i = 0; i < N; ++i)
	for (size_t j = i; j < N; ++j)
	{
		const double pij = p[i * N + j] + p[j * N + i];
		p[i * N + j] = pij;
		if (i != j)
			pSum += 2*pij;
		else
			pSum += pij;
	}

	for (size_t i = 0; i < N; ++i)
	for (size_t j = i; j < N; ++j)
	{
		//TODO check max function
		const double pij = fmax(4*p[i * N + j]/pSum, 1e-12);
		p[i * N + j] = pij;
		p[j * N + i] = pij;
	}

	//TODO initialize Y here

	double *Y = (double *) calloc(N*d, sizeof(double));
	double *one_over_y_dist_plus1 = (double *) calloc(N*N, sizeof(double));

	for (size_t iteration = 0; iteration < maxIterations; ++iteration)
	{
		//Compute pairwise affinities
		double one_over_y_dist_plus1_sum = 0;
		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				if (i != j)
				{
					double dy = 1;
					for (int k = 0; k < d; ++k)
					{
						const double yi =  Y[i*d + k];
						const double yj =  Y[j*d + k];
						const double dist_yij = yi-yj;
						dy += dist_yij*dist_yij;
					}
					one_over_y_dist_plus1[i * N + j] = 1.0/dy;
					one_over_y_dist_plus1_sum += 1.0/dy;
				}
			}
		}

		double momentum = final_momentum;
		if (iter < 20)
			momentum = initial_momentum;

		for (int i = 0; i < N; ++i)
		{
			for (int j = 0; j < N; ++j)
			for (int k = 0; k < d; ++k)
				yd[i * N + j * d + k] = 0; //TODO rename dY - this is the gradient

			for (int j = 0; j < N; ++j)
			{
				//TODO max
				const double qij = max(1e-12, one_over_y_dist_plus1[i * N + j]/one_over_y_dist_plus1_sum);
				double pq = p[i * N + j] - qij;
				for (int k = 0; k < d; ++k)
				{
					const double yi =  Y[i*d + k];
					const double yj =  Y[j*d + k];
					const double gradij = pq * (yi - yj) * one_over_y_dist_plus1[i * N + j];
					yd[i * N + j * d + k] += gradij;
				}
			}
		}

		for (int i = 0; i < N; ++i)
		{
			for (int k = 0; k < d; ++k)
			{
				iY[i*d + k] = momentum * iY[i*d + k] - eta * gains
				const double qij = fmax(1e-12, one_over_y_dist_plus1[i * N + j]/one_over_y_dist_plus1_sum);
				double pq = p[i * N + j] - qij;
				for (int k = 0; k < d; ++k)
				{
					const double yi =  Y[i*d + k];
					const double yj =  Y[j*d + k];
					const double gradij = pq * (yi - yj) * one_over_y_dist_plus1[i * N + j];
					yd[i * N + j * d + k] += gradij;
				}
			}
		}

		//Stop lying about P-values
		for (size_t i = 0; i < N; ++i)
		for (size_t j = i; j < N; ++j)
		{
			p[i * N + j] /= 4;
			p[j * N + i] /= 4;
		}
	}

	//TODO frees
	// free(tmp);
	// free(Yt);
	// free(Yt1);
	// free(Yt2);
	// free(data);
	// free(pi_j);
	// free(qij);
	// free(y_diff);
	return 0;
}