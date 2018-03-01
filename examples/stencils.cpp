#include <stdio.h>
#include <iris/first_derivative_taylor.h>
#include <iris/laplacian3D_taylor.h>
#include <iris/laplacian3D_pade.h>
#include <iris/utils.h>

using namespace ORG_NCSA_IRIS;

main()
{    
    first_derivative_taylor d_dx(6, 0.01);
    d_dx.commit();
    d_dx.trace("d/dx");

    laplacian3D_taylor del(12, 0.01, 0.01, 0.01);
    del.commit();
    del.trace("Δ");

    laplacian3D_pade del2(0, 2, false, 0.01, 0.01, 0.01);
    del2.commit();
}
