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
#include "laplacian3D_taylor.h"
#include "memory.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

// 3D Laplacian approximation, based on Taylor expansion
// Theory and calculator about what follows:
// http://web.media.mit.edu/~crtaylor/calculator.html
//
// Also, for the even parts, we have another tool:
//
// Load IRIS/lisp/stencil.lisp
//
// CL-USER> (stencil-coeff 4)
// #(-1/12 4/3 -5/2 4/3 -1/12)
// CL-USER> (lcm 12 3 2)
// 12
// CL-USER> (map 'vector #'(lambda (x) (* x 12)) (stencil-coeff 4))
// #(-1 16 -30 16 -1)
//
// Same for others
static int coeff2[2] = {1, -2};
static int denom2 = 1;

static int coeff4[3] = {-1, 16, -30};
static int denom4 = 12;

static int coeff6[4] = {2, -27, 270, -490};
static int denom6 = 180;

static int coeff8[5] = {-9, 128, -1008, 8064, -14350};
static int denom8 = 5040;

static int coeff10[6] = {344, -5375, 43000, -258000, 1806000, -3171938};
static int denom10 = 1083600;

static int coeff12[7] = {-50, 864, -7425, 44000, -222750, 1425600, -2480478};
static int denom12 = 831600;

laplacian3D_taylor::laplacian3D_taylor(int in_acc)
    : laplacian3D(in_acc)
{
}

laplacian3D_taylor::laplacian3D_taylor(int in_acc, iris_real in_hx,
				       iris_real in_hy, iris_real in_hz)
    : laplacian3D(in_acc)
{
    set_hx(in_hx);
    set_hy(in_hy);
    set_hz(in_hz);
}

void laplacian3D_taylor::commit()
{
    if(!m_dirty) {
	return;
    }
    
    iris_real h2 = m_hx * m_hx;
    iris_real ry = h2 / (m_hy * m_hy);
    iris_real rz = h2 / (m_hz * m_hz);
    
    int *coeff;
    iris_real denom;

#define ACC_ORDER_CASE(N)			\
    case N:					\
	denom = h2 * denom##N;			\
	coeff = coeff##N;		        \
	break

    switch(m_acc) {
	ACC_ORDER_CASE(2);
	ACC_ORDER_CASE(4);
	ACC_ORDER_CASE(6);
	ACC_ORDER_CASE(8);
	ACC_ORDER_CASE(10);
	ACC_ORDER_CASE(12);
	    
    default:
	throw std::invalid_argument("3D Laplacian taylor stencil only supports even accuracy orders between 2 and 12!");
    }
#undef ACC_ORDER_CASE

    if(m_data != NULL) {
	memory::wfree(m_data);
    }
    int cnt = m_acc/2 + 1;
    int cnt3 = cnt * cnt * cnt;
    m_data = memory::wmalloc(cnt3 * sizeof(iris_real));
    iris_real *data = (iris_real *)m_data;
    for(int i=0;i<cnt3;i++) {
	data[i] = 0.0;
    }

    int c = cnt-1;
    for(int i=0;i<cnt;i++) {
	data[ROW_MAJOR_OFFSET(i, c, c, cnt, cnt)] += coeff[i] / denom;
	data[ROW_MAJOR_OFFSET(c, i, c, cnt, cnt)] += (ry*coeff[i]) / denom;
	data[ROW_MAJOR_OFFSET(c, c, i, cnt, cnt)] += (rz*coeff[i]) / denom;
    }
    m_dirty = false;
}

laplacian3D_taylor::~laplacian3D_taylor()
{
    if(m_data != NULL) {
	memory::wfree(m_data);
    }
}
