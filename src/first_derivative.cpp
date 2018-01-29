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
#include "first_derivative.h"

using namespace ORG_NCSA_IRIS;

void first_derivative::trace(const char *in_name)
{
    printf("---------------------------------\n");
    printf("%s: 1D first derivative stencil\n", in_name);
    printf("---------------------------------\n");
    printf("Δx:          % g\n", m_h);
    printf("Accurate to: % g (Δx^%d)\n\n", pow(m_h, m_acc), m_acc);
    iris_real *data = (iris_real *)m_data;

    for(int i=0;i<m_acc;i++) {
	printf("%s[%+02d] =   % g\n", in_name, m_acc-i, data[i]);
    }

    printf("%s[ 0] =   % g\n", in_name, 0.0);

    for(int i=m_acc-1;i>=0;i--) {
	printf("%s[%+02d] =   % g\n", in_name, i-m_acc, -data[i]);
    }
    printf("\n");
}
