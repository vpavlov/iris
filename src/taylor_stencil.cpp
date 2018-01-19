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
#include "taylor_stencil.h"
#include "logger.h"
#include "mesh.h"
#include "memory.h"
#include "poisson_solver.h"

using namespace ORG_NCSA_IRIS;

// How we got this:
// ----------------
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

static int denom2 = 1;
static int coeff2[3] = { 1, -2, 1};

static int denom4 = 12;
static int coeff4[5] = { -1, 16, -30, 16, -1 };

static int denom6 = 180;
static int coeff6[7] = { 2, -27, 270, -490, 270, -27, 2 };

static int denom8 = 5040;
static int coeff8[9] = { -9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9 };

static int denom10 = 1083600;
static int coeff10[11] = { 344, -5375, 43000, -258000, 1806000, -3171938,
			   1806000, -258000, 43000, -5375, 344 };

static int denom12 = 831600;
static int coeff12[13] = { -50, 864, -7425, 44000, -222750, 1425600, -2480478,
			   1425600, -222750, 44000, -7425, 864, -50 };

taylor_stencil::taylor_stencil(iris *in_obj, int in_order)
    : stencil(in_obj), m_order(in_order)
{
    m_logger->info("Will use %d-order Taylor approximation scheme", m_order);
}

taylor_stencil::~taylor_stencil()
{
}

void taylor_stencil::commit()
{
    if(m_dirty) {
	m_h2 = m_mesh->m_h[0] * m_mesh->m_h[0];

	iris_real ry = m_h2 / (m_mesh->m_h[1] * m_mesh->m_h[1]);
	iris_real rz = m_h2 / (m_mesh->m_h[2] * m_mesh->m_h[2]);

	iris_real denom;
	int *coeff;

#define ORDER_CASE(N)				\
	case N:					\
	    denom = m_h2 * denom##N;		\
	    coeff = coeff##N;			\
		break;

	switch(m_order) {
	    ORDER_CASE(2)
	    ORDER_CASE(4)
	    ORDER_CASE(6)
	    ORDER_CASE(8)
	    ORDER_CASE(10)
	    ORDER_CASE(12)

	default:
	    throw std::invalid_argument("Taylor stencil only supports even orders between 2 and 12!");

	}

#undef ORDER_CASE

	m_size_1d = m_order + 1;
	memory::destroy_3d(m_coeff);
	memory::create_3d(m_coeff, m_size_1d, m_size_1d, m_size_1d);
	for(int i=0;i<m_size_1d;i++) {
	    for(int j=0;j<m_size_1d;j++) {
		for(int k=0;k<m_size_1d;k++) {
		    m_coeff[i][j][k] = 0.0;
		}
	    }
	}

	int center = m_order/2;
	for(int i=0;i<m_size_1d;i++) {
	    m_coeff[i][center][center] +=     coeff[i] / denom ;
	    m_coeff[center][i][center] += (ry*coeff[i] / denom);
	    m_coeff[center][center][i] += (rz*coeff[i] / denom);
	}

	// other configurations that depend on ours must be reset
	if(m_solver != NULL) {
	    m_solver->m_dirty = true;
	}
	m_dirty = false;
    }
}

