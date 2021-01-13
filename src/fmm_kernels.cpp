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
#include <assert.h>
#include "cuda.h"
#include "fmm_kernels.h"

using namespace ORG_NCSA_IRIS;


IRIS_CUDA_DEVICE_HOST
int ORG_NCSA_IRIS::multipole_index(int l, int m)
{
    return l*(l+1) + 2*m;
}

IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::multipole_get(iris_real *M, int l, int m, iris_real *out_re, iris_real *out_im)
{
    assert(l >= 0);
    if(m < 0) {
	if(-m > l) {
	    *out_re = 0.0;
	    *out_im = 0.0;
	    return;
	}
	int i = multipole_index(l, -m);
	iris_real a = M[i];
	iris_real b = M[i+1];
	if(m % 2) {
	    a = -a;
	}else {
	    b = -b;
	}
	*out_re = a;
	*out_im = b;
    }else {
	if(m > l) {
	    *out_re = 0.0;
	    *out_im = 0.0;
	    return;
	}
	int i = multipole_index(l, m);
	*out_re = M[i];
	*out_im = M[i+1];
    }
}

IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::multipole_add(iris_real *M, int l, int m, complex<iris_real> &val)
{
    assert(l >= 0);
    assert(m >= 0);
    int i = multipole_index(l, m);
    M[i] += val.real();
    M[i+1] += val.imag();
}

// This version is used by the CPU kernel and by the M2M, L2L and L2P CPU/GPU kernels
void ORG_NCSA_IRIS::p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_M)
{
    iris_real r2 = x * x + y * y + z * z;

    complex<iris_real> R_m_m(q, 0);
    complex<iris_real> xy(x, y);
    
    for(int m = 0; m < order; m++) {
	multipole_add(out_M, m, m, R_m_m);

	complex<iris_real> R_mplus1_m = z * R_m_m;
	multipole_add(out_M, m+1, m, R_mplus1_m);

	complex<iris_real> prev2 = R_m_m;
	complex<iris_real> prev1 = R_mplus1_m;
	for(int l = m+2; l <= order; l++) {
	    complex<iris_real> R_l_m = (2*l-1) * z * prev1 - r2 * prev2;
	    R_l_m /= (l * l - m * m);
	    multipole_add(out_M, l, m, R_l_m);
	    
	    prev2 = prev1;
	    prev1 = R_l_m;
	}

	R_m_m *= xy;
	R_m_m /= 2*(m+1);
    }
    multipole_add(out_M, order, order, R_m_m);
}


//
// M2M CPU Kernel
//
IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::m2m(int order, iris_real x, iris_real y, iris_real z,
			iris_real *in_M, iris_real *out_M, iris_real *scratch)
{
    p2m(order, x, y, z, 1.0, scratch);
    for(int n=0;n<=order;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re = 0.0, im = 0.0;

	    for(int k=0; k<=n;k++) {
		for(int l=-k;l<=k;l++) {

		    iris_real a, b;
		    multipole_get(scratch, k, l, &a, &b);

		    iris_real c, d;
		    multipole_get(in_M, n-k, m-l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }

	    int idx=multipole_index(n, m);
#ifdef __CUDA_ARCH__
	    atomicAdd(out_M+idx, re);
	    atomicAdd(out_M+idx+1, im);
#else
	    out_M[idx] += re;
	    out_M[idx+1] += im;
#endif
	}
    }
}


//
// P2L CPU kernel
//
IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::p2l(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_L)
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
IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::m2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M, iris_real *out_L, iris_real *in_scratch)
{
    p2l(order, x, y, z, 1.0, in_scratch);
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
	    
#ifdef __CUDA_ARCH__
	    atomicAdd(out_L+idx, re);
	    atomicAdd(out_L+idx+1, im);
#else
	    out_L[idx] += re;
	    out_L[idx+1] += im;
#endif
	}
    }
}


//
// L2L CPU Kernel
//
IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::l2l(int order, iris_real x, iris_real y, iris_real z,
			iris_real *in_L, iris_real *out_L, iris_real *scratch)
{
    p2m(order, x, y, z, 1.0, scratch);
    
    for(int n=0;n<=order;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re = 0.0, im = 0.0;
	    for(int k=0; k<=order-n;k++) {
		for(int l=-k;l<=k;l++) {

		    iris_real a, b;
		    multipole_get(scratch, k, l, &a, &b);
		    b = -b;
		    
		    iris_real c, d;
		    multipole_get(in_L, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }

	    int idx=multipole_index(n, m);
#ifdef __CUDA_ARCH__
	    atomicAdd(out_L+idx, re);
	    atomicAdd(out_L+idx+1, im);
#else
	    out_L[idx] += re;
	    out_L[idx+1] += im;
#endif
	}
    }
}

//
// L2P kernel
//
// NOTE: This is the same as L2L, just n<2 and the result is not written in out_L, but in out_phi, Ex,y,z ...
//       So this can be "write once and only once" optimized, or not -- depends...
IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::l2p(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *in_L, iris_real *scratch,
			iris_real *out_phi, iris_real *out_Ex, iris_real *out_Ey, iris_real *out_Ez)
{
    p2m(order, x, y, z, 1.0, scratch);
    for(int n=0;n<=1;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re = 0.0, im = 0.0;
	    for(int k=0;k<=order-n;k++) {
		for(int l=-k; l<=k; l++) {
		    
		    iris_real a, b;
		    multipole_get(scratch, k, l, &a, &b);
		    b = -b;

		    iris_real c, d;
		    multipole_get(in_L, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }
	    if(n == 0 && m == 0) {
		*out_phi = re;  // multipoles of the L^a_a are real
	    }else if(n == 1 && m == 1) {
		*out_Ex = re*q;  // ??? the minus sign gone to make forces correct ???
		*out_Ey = im*q;  // ??? the minus sign gone to make forces correct ???
	    }else if(n == 1 && m == 0) {
		*out_Ez = re*q;  // ??? the minus sign gone to make forces correct ???
	    }
	}
    }
}

