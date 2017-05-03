#include "comp.h"

static inline dt sign(dt x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// Update gains and update Y with the computed gradient
void gradient_update(dt* Y, dt* dC, dt* uY, dt* gains, int N,
					 int no_dims, dt momentum, dt eta){
	// Update gains
	for(int i = 0; i < N * no_dims; i++)
		gains[i] = (sign(dC[i]) != sign(uY[i])) ? (gains[i] + .2) :
												  (gains[i] * .8);
	for(int i = 0; i < N * no_dims; i++) if(gains[i] < .01) gains[i] = .01;

	// Perform gradient update (with momentum and gains)
	for(int i = 0; i < N * no_dims; i++)
		uY[i] = momentum * uY[i] - eta * gains[i] * dC[i];
	for(int i = 0; i < N * no_dims; i++)  Y[i] = Y[i] + uY[i];
}
