#include "comp.h"

// Early exageration (Multiply all the values of P to the given value)
void early_exageration(double* P, int N, double scale) {
    for (int i = 0; i < N * N; i++) P[i] *= scale;
}
