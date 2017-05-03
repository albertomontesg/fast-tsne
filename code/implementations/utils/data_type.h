#ifndef DATA_TYPE_H
#define DATA_TYPE_H

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

#endif
