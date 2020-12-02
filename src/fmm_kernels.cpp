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
#include <complex>
#include <assert.h>
#include "fmm_kernels.h"

using namespace ORG_NCSA_IRIS;


//
// P2M CPU Kernel
//
void ORG_NCSA_IRIS::p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q,
			iris_real *out_M)
{
    iris_real r2 = x * x + y * y + z * z;
    iris_real zz = z + z;
    std::complex<iris_real> x_plus_iy(x, y);
    std::complex<iris_real> t = 1;
    iris_real next = q;
    iris_real itz = z + zz;
    for(int m = 0; m < order; m++) {
	// 1. R^m_m
	iris_real R_m_m = next;
	int i = multipole_index(m, m);
	out_M[i] += t.real() * R_m_m;
	out_M[i+1] += t.imag() * R_m_m;

	next = R_m_m / (2*(m+1));

	// 2. R_m+1^m
	iris_real R_mplus1_m = z * R_m_m;
	i = multipole_index(m+1, m);
	out_M[i] += t.real() * R_mplus1_m;
	out_M[i+1] += t.imag() * R_mplus1_m;

	iris_real prev2 = R_m_m;
	iris_real prev1 = R_mplus1_m;
	iris_real itz1 = itz;
	for(int l = m+2; l <= order; l++) {

	    // 3. R_l_m
	    iris_real R_l_m = (itz1 * prev1 - r2 * prev2) / (l * l - m * m);
	    i = multipole_index(l, m);
	    out_M[i] += t.real() * R_l_m;
	    out_M[i+1] += t.imag() * R_l_m;

	    prev2 = prev1;
	    prev1 = R_l_m;
	    itz1 += zz;
	}
	t *= x_plus_iy;
	itz += zz;
    }
    int i = multipole_index(order, order);
    out_M[i] += t.real() * next;
    out_M[i+1] += t.imag() * next;
}


//
// M2M CPU Kernel
//
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
	    out_M[idx] += re;
	    out_M[idx+1] += im;
	}
    }
}


//
// P2L CPU kernel
//
void ORG_NCSA_IRIS::p2l(int order, iris_real x, iris_real y, iris_real z, iris_real q,
			iris_real *out_L)
{
    iris_real r2 = x * x + y * y + z * z;
    iris_real r = sqrt(r2);
    iris_real zz = z + z;
    std::complex<iris_real> x_plus_iy(x, y);
    std::complex<iris_real> t = 1;            // (x+iy)^m
    iris_real next = q / r;                   // q * I^0_0
    iris_real twomplus1_over_r2 = 1/r2;       // (2m+1)/r^2
    iris_real two_over_r2 = twomplus1_over_r2 * 2;  // 2/r^2
    iris_real z_over_r2 = z / r2;
    iris_real twoz_over_r2 = z_over_r2 + z_over_r2;
    
    for(int m = 0; m < order; m++) {

	// 1. I^m_m
	iris_real I_m_m = next;
	int i = multipole_index(m, m);
	out_L[i] += t.real() * I_m_m;
	out_L[i+1] += t.imag() * I_m_m;

	next = twomplus1_over_r2 * I_m_m;

	// 2. I_m+1^m
	iris_real I_mplus1_m = z * next;
	i = multipole_index(m+1, m);
	out_L[i] += t.real() * I_mplus1_m;
	out_L[i+1] += t.imag() * I_mplus1_m;

	iris_real itz1 = twomplus1_over_r2 + two_over_r2;
	iris_real prev2 = I_m_m;
	iris_real prev1 = I_mplus1_m;
	iris_real t1 = itz1 * z;
	iris_real t2 = twomplus1_over_r2;
	iris_real t3 = t2;
	for(int l = m+2; l <= order; l++) {

	    // 3. I_l_m
	    iris_real I_l_m = (t1 * prev1 - t3 * prev2);
	    i = multipole_index(l, m);
	    out_L[i] += t.real() * I_l_m;
	    out_L[i+1] += t.imag() * I_l_m;

	    prev2 = prev1;
	    prev1 = I_l_m;
	    t1 += twoz_over_r2;
	    t2 += two_over_r2;
	    t3 += t2;
	}
	t *= x_plus_iy;
	twomplus1_over_r2 = itz1;
    }
    int i = multipole_index(order, order);
    out_L[i] += t.real() * next;
    out_L[i+1] += t.imag() * next;
}


//
// M2L CPU Kernel
//
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
	    out_L[idx] += re;
	    out_L[idx+1] += im;
	}
    }
}


//
// L2L CPU Kernel
//
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
		    if(l % 2) {
			a = -a;
		    }else {
			b = -b;
		    }
		    
		    iris_real c, d;
		    multipole_get(in_L, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }

	    int idx=multipole_index(n, m);
	    out_L[idx] += re;
	    out_L[idx+1] += im;
	}
    }
}

//
// L2P kernel
//
// NOTE: This is the same as L2L, just n<2 and the result is not written in out_L, but in out_phi, Ex,y,z ...
//       So this can be "write once and only once" optimized, or not -- depends...
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
		    if(l % 2) {
			a = -a;
		    }else {
			b = -b;
		    }

		    iris_real c, d;
		    multipole_get(in_L, n+k, m+l, &c, &d);

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }
	    if(n == 0 && m == 0) {
		*out_phi = re;  // multipoles of the L^a_a are real
	    }else if(n == 1 && m == 1) {
		*out_Ex = re;
		*out_Ey = im;
	    }else if(n == 1 && m == 0) {
		*out_Ez = re;
	    }
	}
    }
}

