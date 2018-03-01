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
#include <stdexcept>
#include "first_derivative_taylor.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

// First derivative approximation, based on Taylor expansion
// Theory and calculator about what follows:
// http://web.media.mit.edu/~crtaylor/calculator.html

static int coeff1[1] = { 1 };
static int denom1 = 2;

static int coeff2[2] = { -1, 8 };
static int denom2 = 12;

static int coeff3[3] = { 1, -9, 45 };
static int denom3 = 60;

static int coeff4[4] = { -3, 32, -168, 672 };
static int denom4 = 840;

static int coeff5[5] = { 2, -25, 150, -600, 2100 };
static int denom5 = 2520;

static int coeff6[6] = { -5, 72, -495, 2200, -7425, 23760 };
static int denom6 = 27720;

first_derivative_taylor::first_derivative_taylor(int in_acc)
    : first_derivative(in_acc)
{
}

first_derivative_taylor::first_derivative_taylor(int in_acc, iris_real in_h)
    : first_derivative(in_acc)
{
    set_h(in_h);
}

void first_derivative_taylor::commit()
{
    if(!m_dirty) {
	return;
    }

    int *coeff;
    iris_real denom;

#define ACC_ORDER_CASE(N)			\
    case N:					\
	denom = m_h * denom##N;			\
	coeff = coeff##N;		        \
	break

    switch(m_acc) {
	ACC_ORDER_CASE(1);
	ACC_ORDER_CASE(2);
	ACC_ORDER_CASE(3);
	ACC_ORDER_CASE(4);
	ACC_ORDER_CASE(5);
	ACC_ORDER_CASE(6);
	    
    default:
	throw std::invalid_argument("First derivative taylor stencil only supports accuracy orders between 1 and 6!");
	
    }
#undef ACC_ORDER_CASE

    if(m_delta != NULL) {
	memory::wfree(m_delta);
    }
    m_delta = memory::wmalloc(m_acc * sizeof(iris_real));

    iris_real *data = (iris_real *)m_delta;
    for(int i=0;i<m_acc;i++) {
	data[i] = coeff[i] / denom;
    }
    
    m_lhs_only = true;
    m_dirty = false;
}

first_derivative_taylor::~first_derivative_taylor()
{
    if(m_delta != NULL) {
	memory::wfree(m_delta);
    }
}
