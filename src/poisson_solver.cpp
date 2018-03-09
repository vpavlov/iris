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
#include "poisson_solver.h"
#include "laplacian3D_pade.h"
#include "first_derivative_taylor.h"

using namespace ORG_NCSA_IRIS;

poisson_solver::poisson_solver(class iris *obj)
    : state_accessor(obj), m_dirty(true), m_style(0), m_arg1(0), m_arg2(0),
      m_laplacian(NULL), m_ddx(NULL), m_ddy(NULL), m_ddz(NULL)
{
};

poisson_solver::~poisson_solver()
{
    if(m_laplacian != NULL) { delete m_laplacian; }
    if(m_ddx != NULL) { delete m_ddx; }
    if(m_ddy != NULL) { delete m_ddy; }
    if(m_ddz != NULL) { delete m_ddz; }
};

void poisson_solver::set_laplacian(int in_style, int in_arg1, int in_arg2)
{
    if(in_style == m_style && in_arg1 == m_arg1 && in_arg2 == m_arg2) {
	return;
    }

    m_style = in_style;
    m_arg1 = in_arg1;
    m_arg2 = in_arg2;
    m_dirty = true;
}

void poisson_solver::commit()
{
    // set defaults (if not configured)
    if(m_style == 0) {
	set_laplacian(IRIS_LAPL_STYLE_PADE, 0, 0);
    }

    if(!m_dirty) {
	return;
    }

    if(m_laplacian != NULL) { delete m_laplacian; }
    if(m_ddx != NULL) { delete m_ddx; }
    if(m_ddy != NULL) { delete m_ddy; }
    if(m_ddz != NULL) { delete m_ddz; }

    switch(m_style) {
    case IRIS_LAPL_STYLE_PADE:
	// TODO: also add cut as parameter
	m_laplacian = new laplacian3D_pade(m_arg1, m_arg2, false);
	m_ddx = new first_derivative_taylor((m_arg1 + 2)/2);
	m_ddy = new first_derivative_taylor((m_arg1 + 2)/2);
	m_ddz = new first_derivative_taylor((m_arg1 + 2)/2);
	break;

    default:
	throw std::logic_error("Unknown laplacian style selected!");
    }

    m_dirty = false;
}
