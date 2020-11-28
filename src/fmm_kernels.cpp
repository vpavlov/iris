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
#include "fmm_kernels.h"

using namespace ORG_NCSA_IRIS;



//
// P2M CPU Kernel
//
void ORG_NCSA_IRIS::p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_gamma)
{
    iris_real r2 = x * x + y * y + z * z;
    iris_real zz = z + z;
    std::complex<iris_real> x_plus_iy(x, y);
    std::complex<iris_real> t = 1;
    iris_real next = q;
    iris_real itz = z + zz;
    for(int m = 0; m < order; m++) {
	// 1. gamma_m^m
	iris_real gamma_m_m = next;
	int i = multipole_index(m, m);
	out_gamma[i] += t.real() * gamma_m_m;
	out_gamma[i+1] += t.imag() * gamma_m_m;

	next = gamma_m_m / (2*(m+1));

	// 2. gamma_m+1^m
	iris_real gamma_mplus1_m = z * gamma_m_m;
	i = multipole_index(m+1, m);
	out_gamma[i] += t.real() * gamma_mplus1_m;
	out_gamma[i+1] += t.imag() * gamma_mplus1_m;

	iris_real prev2 = gamma_m_m;
	iris_real prev1 = gamma_mplus1_m;
	iris_real itz1 = itz;
	for(int l = m+2; l <= order; l++) {

	    // 3. gamma_l_m
	    iris_real gamma_l_m = (itz1 * prev1 - r2 * prev2) / (l * l - m * m);
	    i = multipole_index(l, m);
	    out_gamma[i] += t.real() * gamma_l_m;
	    out_gamma[i+1] += t.imag() * gamma_l_m;

	    prev2 = prev1;
	    prev1 = gamma_l_m;
	    itz1 += zz;
	}
	t *= x_plus_iy;
	itz += zz;
    }
    int i = multipole_index(order, order);
    out_gamma[i] += t.real() * next;
    out_gamma[i+1] += t.imag() * next;
}


//
// M2M CPU Kernel
//
void ORG_NCSA_IRIS::m2m(int order, iris_real x, iris_real y, iris_real z, iris_real *in_gamma, iris_real *out_gamma, iris_real *scratch)
{
    p2m(order, x, y, z, 1.0, scratch);
    
    for(int n=0;n<=order;n++) {
	for(int m=0;m<=n;m++) {
	    iris_real re = 0.0, im = 0.0;

	    for(int k=0; k<=n;k++) {
                int l_min = std::max(-k, m - (n - k));
                int l_max = std::min(+k, m + (n - k));
		for(int l=l_min;l<=l_max;l++) {

		    int idx = multipole_index(k, l);
		    iris_real a = in_gamma[idx];
		    iris_real b = in_gamma[idx+1];

		    idx = multipole_index(n-k, m-l);
		    iris_real c = scratch[idx];
		    iris_real d = scratch[idx+1];

		    re += a*c - b*d;
		    im += a*d + b*c;
		}
	    }
	    
	    int idx=multipole_index(n, m);
	    out_gamma[idx] += re;
	    out_gamma[idx+1] += im;
	}
    }
}
