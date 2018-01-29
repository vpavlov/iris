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
#ifndef __IRIS_POISSON_SOLVER_H__
#define __IRIS_POISSON_SOLVER_H__

#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

#define IRIS_POISSON_SOLVER_PSM  1  // pseudo-spectral method

    class poisson_solver : protected state_accessor {

    public:
	poisson_solver(class iris *obj);
	~poisson_solver();

	void set_laplacian(int in_style, int in_order);

	virtual void commit();
	virtual void solve() = 0;

    public:
	bool m_dirty;  // wether to recalculate on commit
	int m_style;   // which approximation to use (Taylor or Pade)
	int m_order;   // accuracy order of the Laplacian
	class laplacian3D *m_laplacian;  // the Laplacian stencil
    };
}

#endif
