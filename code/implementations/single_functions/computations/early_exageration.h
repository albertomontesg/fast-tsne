#ifndef EARLY_EXAGERATION_H
#define EARLY_EXAGERATION_H

#include "../../utils/data_type.h"

// Early exageration (Multiply all the values of P to the given value)
inline void early_exageration(dt* P, int N, dt scale) {
    for (int i = 0; i < N * N; i++) P[i] *= scale;
}

#endif
