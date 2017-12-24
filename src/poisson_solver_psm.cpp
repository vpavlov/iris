#include <math.h>
#include <stdio.h>
#include "poisson_solver_psm.h"

using namespace ORG_NCSA_IRIS;

poisson_solver_psm::poisson_solver_psm()
{
}

// For the pseudo-spectral method we prefer the decomposition to be
// as uniform as possible: the closest to the cubic root of N, the better
// Also, we prefer X > Y > Z
float poisson_solver_psm::eval_dd_conf(int x, int y, int z)
{

    int xyz = x*y*z;
    float xyz_cube_root = pow(xyz, 0.3333333);
    float dx = fabs((float)x - xyz_cube_root);
    float dy = fabs((float)y - xyz_cube_root);
    float dz = fabs((float)z - xyz_cube_root);
    float crdiff = 1.0 / sqrt(dx*dx+dy*dy+dz*dz);

    if(x >= y) {
	crdiff *= 1.000001;
	if(y >= z) {
	    crdiff *= 1.000001;
	}
    }

    return crdiff;
}
