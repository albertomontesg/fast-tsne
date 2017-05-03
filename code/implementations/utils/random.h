#ifndef RANDOM_H
#define RANDOM_H
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "data_type.h"

// Generates a Gaussian random number
dt randn() {
	dt x, y, radius;
	do {
		x = 2 * (rand() / ((dt) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((dt) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt_c(-2 * log_c(radius) / radius);
	x *= radius;
	return x;
}

#endif
