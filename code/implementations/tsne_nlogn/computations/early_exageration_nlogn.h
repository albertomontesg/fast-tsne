#ifndef EARLY_EXAGERATION_H
#define EARLY_EXAGERATION_H

#include "../../utils/data_type.h"

inline void early_exageration_sparse(dt* val_P, int no_elements, dt scale){
	for(int i=0; i<no_elements; i++) {
		val_P[i] *= scale;
	}
}

#endif

