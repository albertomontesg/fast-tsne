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

void randn_matrix(dt* m, size_t row, size_t col) {
	for (size_t i = 0; i < row*col; i++) {
		m[i] = randn();
	}
}

void rand_matrix(dt* m, size_t row, size_t col) {
	for (size_t i = 0; i < row*col; i++) {
		m[i] = static_cast<double>(rand() + 1) / RAND_MAX;
	}
}

#endif
