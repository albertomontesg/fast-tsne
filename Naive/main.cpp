#include <math.h>
#include <stdlib.h>
#include <stdio.h>

const size_t D = 3; //dimensionality of data
const size_t d = 2; //dimensionality of output
const size_t N = 5; //number of data points
const size_t T = 1000; //number of iterations
const double minSigmaSq = 10e-4;
const double maxSigmaSq = 10e4;
const double sigmaSqStart = (minSigmaSq+maxSigmaSq)/2;
const double targetPerp = 2;
const double eta = 0.9;

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


	printf("calculating distances for pij\n");
	double *pi_j = (double *) malloc(N*N*sizeof(double));
	for(size_t i = 0; i < N; ++i)
	{
		for(size_t j = 0; j < i; ++j)
		{
			double nd = 0;
			for(size_t k = 0; k < D; ++k)
			{
				double diff = data[i*D + k] - data[j*D + k];
				nd += diff*diff;
			}
			pi_j[ i*N + j] = nd;
			pi_j[ j*N + i] = nd;
		}
		pi_j[i*N + i] = 0;
	}

	printf("calculating pij\n");
	double *tmp = (double *) malloc(N*sizeof(double));
	for(size_t i = 0; i < N; ++i)
	{
		printf("calculating for point i =%lu\n", i);
		for(size_t j = 0; j < N; ++j) tmp[j] = pi_j[j];
		double lower = minSigmaSq;
		double upper = maxSigmaSq;
		double current = sigmaSqStart;
		double perp = calcPij(tmp, pi_j + (i*N) , i, current);
		while ( fabs(perp - targetPerp) > 10e-1 )
		{
			printf("perp: %lf, targetPerp: %lf\n", perp, targetPerp);
			if(perp > targetPerp)
			{
				upper = current;
				current = (upper+lower)/2;
			} else
			{
				lower = current;
				current = (upper+lower)/2;
			}
			printf("calculating with %lf\n", current);
			perp = calcPij(tmp, pi_j+(i*N), i, current);
		}
	}

	for(size_t i = 0; i < N; ++i)
	{
		for(size_t j = 0; j < i; ++j)
		{
			double p = (pi_j[i*N + j] + pi_j[j*N + i])/(2*N);
			pi_j[i*N + j] = p;
			pi_j[j*N + i] = p;
		}
		pi_j[i*N + i] /= N; 
	}

	printf("initializing Y\n");
	double *Yt = (double *) malloc(d*N*sizeof(double));
	double *Yt1 = (double *) malloc(d*N*sizeof(double));
	double *Yt2 = (double *) malloc(d*N*sizeof(double));
	for(size_t i = 0; i < d*N; ++i)
		Yt[i] = randn(0, 10e-4);


	double *qij = (double *) malloc(N*N*sizeof(double));
	double *y_diff = (double *) malloc(N*N*sizeof(double));
	for(size_t t = 0; t < T; ++t)
	{
		printf("stepping t=%lu\n", t);
		const double alpha = 0.1; //TODO adapt
		double denom = 0;
		for(size_t i = 0; i < N; ++i)
		{
			for(size_t j = 0; j < i; ++j)
			{
				double nd = 0;
				for(size_t k = 0; k < D; ++k)
				{
					double diff = Yt[i*D + k] - Yt[j*D + k];
					nd += diff*diff;
				}
				nd = 1.0/(1+nd);				
				denom += nd;
				qij[ i*N + j] = nd;
				qij[ j*N + i] = nd;
				y_diff[i*N + j] = nd;
				y_diff[j*N + i] = nd;
			}
			qij[i*N + i] = 0;
		}
		for(size_t i = 0; i < N; ++i)
			for(size_t j = 0; j < i; ++j)
			{
				qij[ i*N + j] /= denom;
				qij[ j*N + i] /= denom;
			}
		for(size_t i = 0; i < N; ++i)
		{
			double grad[d] = {0};
			double tmp[d];
			for(size_t j = 0; j < N; ++j)
			{
				const double fact = eta*4*(pi_j[i*N + j] - qij[i*N + j])*y_diff[i*N + j];
				for(size_t k = 0; k < d; ++k)
				{
					tmp[k] = Yt[i*d + k] + fact*(Yt[i*d + k] - Yt[j*d + k]);
					if ( t > 1)
						tmp[k] += alpha*(Yt1[i*d + k] - Yt2[i*d + k]);
				}
			}
			for(size_t k = 0; k < d; ++k)
			{
		
				Yt2[i*d + k] = Yt1[i*d + k];
				Yt1[i*d + k] = Yt[i*d + k];
				Yt[i*d + k] = tmp[k];
			}
		}
	}

	for(size_t i = 0; i < N; ++i)
	{
		printf("[");
		for (size_t k = 0; k < d; ++k)
			printf("%lf,", Yt[i*d + k]);
		printf("]^T\n");
	}

	free(tmp);
	free(Yt);
	free(Yt1);
	free(Yt2);
	free(data);
	free(pi_j);
	free(qij);
	free(y_diff);
	return 0;
}