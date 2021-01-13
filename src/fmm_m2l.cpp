// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2018, the National Center for Supercomputing Applications
//
// Primary authors:
//     Valentin Pavlov <vpavlov@rila.bg>
//     Peicho Petkov <peicho@phys.uni-sofia.bg>
//     Stoyan Markov <markov@acad.bg>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include "utils.h"
#include "openmp.h"
#include "fmm_kernels.h"

using namespace ORG_NCSA_IRIS;

//
// P2L CPU kernel
//
void ORG_NCSA_IRIS::h_p2l(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_L)
{
    iris_real r2 = x * x + y * y + z * z;
    iris_real r = sqrt(r2);
    complex<iris_real> I_m_m(q/r, 0);
    complex<iris_real> xy(x, y);

    for(int m = 0; m < order; m++) {
	multipole_add(out_L, m, m, I_m_m);

	complex<iris_real> I_mplus1_m = ((2*m+1)*z/r2) * I_m_m;
	multipole_add(out_L, m+1, m, I_mplus1_m);

	complex<iris_real> prev2 = I_m_m;
	complex<iris_real> prev1 = I_mplus1_m;
	for(int l = m+2; l <= order; l++) {
	    complex<iris_real> t = prev2;
	    t *= ((l-1)*(l-1) - m*m);
	    complex<iris_real> I_l_m = (2*l-1) * z * prev1 - t;
	    I_l_m /= r2;
	    multipole_add(out_L, l, m, I_l_m);
	    prev2 = prev1;
	    prev1 = I_l_m;
	}

	I_m_m *= (2*m+1);
	I_m_m *= xy;
	I_m_m /= r2;
    }
    multipole_add(out_L, order, order, I_m_m);
}


//
// M2L CPU Kernel
//
void ORG_NCSA_IRIS::h_m2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M, iris_real *out_L, iris_real *in_scratch)
{
    h_p2l(order, x, y, z, 1.0, in_scratch);
    for(int n=0;n<=order;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re = 0.0, im = 0.0;
	    for(int k=0;k<=order-n;k++) {
		for(int l=-k;l<=k;l++) {
		    iris_real a, b;
		    multipole_get(in_M, k, l, &a, &b);
		    b = -b;

		    iris_real c, d;
		    multipole_get(in_scratch, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }

	    int idx = multipole_index(n, m);
	    
#if defined _OPENMP
#pragma omp atomic
#endif
	    out_L[idx] += re;
#if defined _OPENMP
#pragma omp atomic
#endif
	    out_L[idx+1] += im;
	}
    }
}
