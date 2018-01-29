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
#include "poisson_solver.h"
#include "laplacian3D_taylor.h"

using namespace ORG_NCSA_IRIS;

poisson_solver::poisson_solver(class iris *obj)
    : state_accessor(obj), m_dirty(true), m_style(0), m_order(0),
      m_laplacian(NULL)
{
};

poisson_solver::~poisson_solver()
{
    if(m_laplacian != NULL) {
	delete m_laplacian;
    }
};

void poisson_solver::set_laplacian(int in_style, int in_order)
{
    if(in_style == m_style && in_order == m_order) {
	return;
    }

    m_style = in_style;
    m_order = in_order;
    m_dirty = true;
}

void poisson_solver::commit()
{
    // set defaults (if not configured)
    if(m_style == 0 || m_order == 0) {
	set_laplacian(IRIS_LAPL_STYLE_TAYLOR, 2);
    }

    if(!m_dirty) {
	return;
    }

    if(m_laplacian != NULL) {
	delete m_laplacian;
    }

    switch(m_style) {
    case IRIS_LAPL_STYLE_TAYLOR:
	m_laplacian = new laplacian3D_taylor(m_order);
	break;

    default:
	throw std::logic_error("Unknown laplacian style selected!");
    }

    m_dirty = false;
}
