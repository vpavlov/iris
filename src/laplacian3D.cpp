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

using namespace ORG_NCSA_IRIS;

void laplacian3D::trace(const char *in_name)
{
    printf("---------------------------------\n");
    printf("%s: 3D Laplacian stencil\n", in_name);
    printf("---------------------------------\n");
    printf("Δx:          % g\n", m_hx);
    printf("Δy:          % g\n", m_hy);
    printf("Δz:          % g\n", m_hz);
    printf("Accurate to: % g (Δx^%d)\n\n", pow(m_hx, m_acc), m_acc);
    iris_real *data = (iris_real *)m_data;

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
