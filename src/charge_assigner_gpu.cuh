#pragma once
	// device function for weights computing
__device__
void compute_weights_dev(iris_real dx, iris_real dy, iris_real dz, 
                         iris_real* m_coeff, iris_real (&weights)[3][IRIS_MAX_ORDER], int order);