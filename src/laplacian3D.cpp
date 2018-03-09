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
#include <stdio.h>
#include <math.h>
#include "laplacian3D.h"
#include "utils.h"
#include "cdo3D.h"

using namespace ORG_NCSA_IRIS;

// The first several terms (19 in this case) of the Taylor expansion of
// 
// (hD)^2 = 4*asinh(δ/2)^2
// 
// See iris/lisp/stencil_coeff.lisp for explanation of the theory behind this
//
// These should be more than enough, since they cover accuracy order of at
// least up to h^20, which in case h = 1/128 is O(10^-43). If however, this is
// somehow not enough, load the above mentioned lisp file and execute
// (d2/dx2-taylor coeff 100) or something. It gives only odd (non-zero) coeffs.
//
static iris_real taylor_coeff[] = {
    1.0, 0.0, -1.0/12, 0.0, 1.0/90, 0.0, -1.0/560, 0.0, 1.0/3150, 0.0,
    -1.0/16632, 0.0, 1.0/84084, 0.0, -1.0/411840, 0.0, 1.0/1969110, 0.0,
    -1.0/9237800
};

void laplacian3D::trace(const char *in_name)
{
    printf("---------------------------------\n");
    printf("%s: 3D Laplacian stencil\n", in_name);
    printf("---------------------------------\n");
    printf("Δx:          % g\n", m_hx);
    printf("Δy:          % g\n", m_hy);
    printf("Δz:          % g\n", m_hz);
    printf("Accurate to: % g (Δx^%d)\n\n", pow(m_hx, m_acc), m_acc);
    iris_real *data = (iris_real *)m_delta;

    int cnt = m_acc/2 + 1;
    int all = m_acc + 1;

    for(int i=0;i<all;i++) {
	for(int j=0;j<all;j++) {
	    for(int k=0;k<all;k++) {
		int ii = (i < cnt) ? i : (all-1-i);
		int jj = (j < cnt) ? j : (all-1-j);
		int kk = (k < cnt) ? k : (all-1-k);
		
		if(data[ROW_MAJOR_OFFSET(ii, jj, kk, cnt, cnt)] != 0.0) {
		    printf("%s[%+02d,%+02d,%+02d] =   % g\n", in_name,
			   cnt-i-1, cnt-j-1, cnt-k-1,
			   data[ROW_MAJOR_OFFSET(ii, jj, kk, cnt, cnt)]);
		}
	    }
	}
    }
}

void laplacian3D::trace2(const char *in_name)
{
    printf("---------------------------------\n");
    printf("%s: 3D Laplacian stencil (extents %d/%d)\n", in_name,
	   get_delta_extent(), get_gamma_extent());
    printf("---------------------------------\n");
    printf("hx:          % g\n", m_hx);
    printf("hy:          % g\n", m_hy);
    printf("hz:          % g\n", m_hz);
    printf("Accurate to: % g (hx^%d)\n\n", pow(m_hx, m_acc), m_acc);
    cdo3D *delta = (cdo3D *)m_delta;
    cdo3D *gamma = (cdo3D *)m_gamma;
    delta->dump("Δ");
    if(!m_lhs_only) {
	gamma->dump("Γ");
    }
}
