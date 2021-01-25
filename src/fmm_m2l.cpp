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
#include <string.h>
#include "utils.h"
#include "openmp.h"
#include "fmm.h"
#include "fmm_kernels.h"
#include "fmm_swapxz.h"

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


//
// M2L CPU Kernel
//
void ORG_NCSA_IRIS::h_m2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M1, iris_real *out_L2, iris_real *in_scratch,
			  iris_real *in_M2, iris_real *out_L1, bool do_other_side)
{
    h_p2l(order, x, y, z, 1.0, in_scratch);
    for(int n=0;n<=order;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re1 = 0.0, im1 = 0.0;
	    iris_real re2 = 0.0, im2 = 0.0;
	    for(int k=0;k<=order-n;k++) {
		for(int l=-k;l<=k;l++) {
		    iris_real a, b;
		    mget(in_M1, k, l, &a, &b);
		    b = -b;

		    iris_real c, d;
		    mget(in_scratch, n+k, m+l, &c, &d);

		    re2 += a*c - b*d;
		    im2 += a*d + b*c;

		    mget(in_M2, k, l, &a, &b);
		    b = -b;

		    if(do_other_side) {
			if((n+k) % 2) {
			    c = -c;
			    d = -d;
			}
			
			re1 += a*c - b*d;
			im1 += a*d + b*c;
		    }
		}
	    }

	    int idx = n * (n + 1);
	    
#if defined _OPENMP
#pragma omp atomic
#endif
	    out_L2[idx + m] += re2;
	    
#if defined _OPENMP
#pragma omp atomic
#endif
	    out_L2[idx - m] += im2;

	    if(do_other_side) {
#if defined _OPENMP
#pragma omp atomic
#endif
		out_L1[idx + m] += re1;
		
#if defined _OPENMP
#pragma omp atomic
#endif
		out_L1[idx - m] += im1;
	    }
	}
    }
}

void mrot(iris_real *out, iris_real *in, iris_real alpha, int p)
{    
    for(int n=0;n<=p;n++) {
	int idx = n * (n + 1);

	// no need to rotate for m=0
	iris_real re, im;
	mget(in, n, 0, &re, &im);
	out[idx] = re;
	
	for(int m=1;m<=n;m++) {
	    iris_real re, im;
	    iris_real cos_ma = cos_fn(m*alpha);
	    iris_real sin_ma = sin_fn(m*alpha);
	    mget(in, n, m, &re, &im);
	    out[idx + m] = cos_ma * re - sin_ma * im;
	    out[idx - m] = sin_ma * re + cos_ma * im;
	}
    }
}

void swapT_xz(iris_real *mult, int p)
{
    iris_real tmp[(IRIS_FMM_MAX_ORDER+1)*(IRIS_FMM_MAX_ORDER+1)];
    
    for(int i=1;i<=p;i++) {
	iris_real *src = mult + i*i;
	int cnt = 2*i + 1;
	swapT_fns[i-1](tmp, src);
	memcpy(src, tmp, cnt * sizeof(iris_real));
    }
}

void swap_xz(iris_real *mult, int p)
{
    iris_real tmp[(IRIS_FMM_MAX_ORDER+1)*(IRIS_FMM_MAX_ORDER+1)];
    
    for(int i=1;i<=p;i++) {
	iris_real *src = mult + i*i;
	int cnt = 2*i + 1;
	swap_fns[i-1](tmp, src);
	memcpy(src, tmp, cnt * sizeof(iris_real));
    }
}

iris_real fact[] = {
		    1.0,
		    1.0,
		    2.0,
		    6.0,
		    24.0,
		    120.0,
		    720.0,
		    5040.0,
		    40320.0,
		    362880.0,
		    3628800.0
};

void ORG_NCSA_IRIS::h_m2l_v2(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M1, iris_real *out_L2, iris_real *in_scratch,
			     iris_real *in_M2, iris_real *out_L1, bool do_other_side)
{
    iris_real f[(IRIS_FMM_MAX_ORDER+1)*(IRIS_FMM_MAX_ORDER+1)];
    memset(f, 0, (IRIS_FMM_MAX_ORDER+1)*(IRIS_FMM_MAX_ORDER+1)*sizeof(iris_real));

    iris_real xxyy = x * x + y * y;
    iris_real az = atan_fn(x, y);
    iris_real ax = - atan_fn(sqrt_fn(xxyy), z);
    iris_real r2 = xxyy + z * z;
    iris_real r = sqrt_fn(r2);
    
    mrot(in_scratch, in_M1, az, order);
    swapT_xz(in_scratch, order);
    mrot(in_scratch, in_scratch, ax, order);
    swapT_xz(in_scratch, order);

    for(int n=0;n<=order;n++) {
	int idx = n * (n + 1);
	for(int m=0;m<=n;m++) {
	    iris_real s = 1.0;
	    for(int k=m;k<=order-n;k++) {
		iris_real re, im;
		mget(in_scratch, k, m, &re, &im);

		iris_real fct = s * fact[n+k] / pow_fn(r, n+k+1);
		re *= fct;
		im *= fct;
		
		f[idx + m] += re;
		f[idx - m] += im;
	    }
	    s *= -1.0;
	}
    }

    swap_xz(f, order);
    mrot(f, f, -ax, order);
    swap_xz(f, order);
    mrot(out_L2, f, -az, order);
}
