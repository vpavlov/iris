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
#include "cuda.h"
#include "fmm_kernels.h"

using namespace ORG_NCSA_IRIS;

IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::mget(iris_real *M, int n, int m, iris_real *out_re, iris_real *out_im)  // n lower, m upper index
{
    int c = n * (n + 1);
    int r = c + m;
    int l = c - m;
    
    if(m == 0) {
	*out_re = M[c];
	*out_im = 0;
    }else if(m > 0) {
	if(m > n) {
	    *out_re = 0.0;
	    *out_im = 0.0;
	}else {
	    *out_re = M[r];
	    *out_im = M[l];
	}
    }else {
	if(-m > n) {
	    *out_re = 0.0;
	    *out_im = 0.0;
	}else if(m % 2 == 0) {
	    *out_re = M[l];
	    *out_im = -M[r];
	}else {
	    *out_re = -M[l];
	    *out_im = M[r];
	}
    }
}


IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::madd(iris_real *M, int n, int m, complex<iris_real> &val)
{
    int c = n * (n + 1);
    M[c+m] += val.real();
    M[c-m] += val.imag();
}


// This version is used by the CPU kernel and by the M2M, L2L and L2P CPU/GPU kernels
void ORG_NCSA_IRIS::p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_M)
{
    iris_real r2 = x * x + y * y + z * z;

    complex<iris_real> R_m_m(q, 0);
    complex<iris_real> xy(x, y);
    
    for(int m = 0; m < order; m++) {
	madd(out_M, m, m, R_m_m);

	complex<iris_real> R_mplus1_m = z * R_m_m;
	madd(out_M, m+1, m, R_mplus1_m);

	complex<iris_real> prev2 = R_m_m;
	complex<iris_real> prev1 = R_mplus1_m;
	for(int l = m+2; l <= order; l++) {
	    complex<iris_real> R_l_m = (2*l-1) * z * prev1 - r2 * prev2;
	    R_l_m /= (l * l - m * m);
	    madd(out_M, l, m, R_l_m);
	    
	    prev2 = prev1;
	    prev1 = R_l_m;
	}

	R_m_m *= xy;
	R_m_m /= 2*(m+1);
    }
    madd(out_M, order, order, R_m_m);
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
		    mget(scratch, k, l, &a, &b);

		    iris_real c, d;
		    mget(in_M, n-k, m-l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }

	    int idx = n * (n + 1);
	    
#ifdef __CUDA_ARCH__
	    atomicAdd(out_M + idx + m, re);
	    atomicAdd(out_M + idx - m, im);
#else
	    out_M[idx + m] += re;
	    out_M[idx - m] += im;
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
	madd(out_L, m, m, I_m_m);

	complex<iris_real> I_mplus1_m = ((2*m+1)*z/r2) * I_m_m;
	madd(out_L, m+1, m, I_mplus1_m);

	complex<iris_real> prev2 = I_m_m;
	complex<iris_real> prev1 = I_mplus1_m;
	for(int l = m+2; l <= order; l++) {
	    complex<iris_real> t = prev2;
	    t *= ((l-1)*(l-1) - m*m);
	    complex<iris_real> I_l_m = (2*l-1) * z * prev1 - t;
	    I_l_m /= r2;
	    madd(out_L, l, m, I_l_m);
	    prev2 = prev1;
	    prev1 = I_l_m;
	}

	I_m_m *= (2*m+1);
	I_m_m *= xy;
	I_m_m /= r2;
    }
    madd(out_L, order, order, I_m_m);
}


#ifdef IRIS_CUDA

//
// M2L GPU Kernel
//
IRIS_CUDA_DEVICE_HOST
void ORG_NCSA_IRIS::m2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M1, iris_real *out_L2, iris_real *in_scratch,
			iris_real *in_M2, iris_real *out_L1, bool do_other_side)
{
    p2l(order, x, y, z, 1.0, in_scratch);
    for(int n=0;n<=order;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re1 = 0.0, im1 = 0.0;
	    iris_real re2 = 0.0, im2 = 0.0;
	    for(int k=0;k<=order-n;k++) {
		for(int l=-k;l<=k;l++) {
		    iris_real a, b;
		    multipole_get(in_M1, k, l, &a, &b);
		    b = -b;

		    iris_real c, d;
		    multipole_get(in_scratch, n+k, m+l, &c, &d);

		    re2 += a*c - b*d;
		    im2 += a*d + b*c;

		    if(do_other_side) {
			multipole_get(in_M2, k, l, &a, &b);
			b = -b;

			if((n+k) % 2) {
			    c = -c;
			    d = -d;
			}
			
			re1 += a*c - b*d;
			im1 += a*d + b*c;
		    }
		}
	    }

	    int idx = multipole_index(n, m);
	    
	    atomicAdd(out_L2+idx, re2);
	    atomicAdd(out_L2+idx+1, im2);
	    if(do_other_side) {
		atomicAdd(out_L1+idx, re1);
		atomicAdd(out_L1+idx+1, im1);
	    }
	}
    }
}

#endif

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
		    mget(scratch, k, l, &a, &b);
		    b = -b;
		    
		    iris_real c, d;
		    mget(in_L, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }

	    int idx = n * (n + 1);
#ifdef __CUDA_ARCH__
	    atomicAdd(out_L + idx + m, re);
	    atomicAdd(out_L + idx - m, im);
#else
	    out_L[idx + m] += re;
	    out_L[idx - m] += im;
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
		    mget(scratch, k, l, &a, &b);
		    b = -b;

		    iris_real c, d;
		    mget(in_L, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }
	    if(n == 0 && m == 0) {
		*out_phi = re;  // multipoles of the L^a_a are real
	    }else if(n == 1 && m == 1) {
		*out_Ex = re*q;
		*out_Ey = im*q;
	    }else if(n == 1 && m == 0) {
		*out_Ez = re*q;
	    }
	}
    }
}

