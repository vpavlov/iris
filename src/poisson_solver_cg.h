// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2019, the National Center for Supercomputing Applications
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
#ifndef __IRIS_POISSON_SOLVER_CG_H__
#define __IRIS_POISSON_SOLVER_CG_H__

#include "poisson_solver.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver_cg : public poisson_solver {

    public:
	poisson_solver_cg(class iris *obj);
	~poisson_solver_cg();

	void commit();
	void solve();

	// TODO: write these
	void set_stencil_width(int in_width);
	void set_max_iters(int in_max_iters);
	void set_epsilon(iris_real in_epsilon);

    private:
	void init_stencil();
	void init_stencil3();
	void adot(iris_real ***in, iris_real ***out, class haloex *hex);
	iris_real dot(iris_real ***v1, iris_real ***v2, bool v1_has_halo, bool v2_has_halo);
	void axpby(iris_real a, iris_real ***x, iris_real b, iris_real ***y, iris_real ***out,
		   bool x_has_halo, bool y_has_halo, bool out_has_halo);

	// config
	int m_stencil_width;
	int m_max_iters;
	iris_real m_epsilon;

	iris_real ***m_stencil;

	int m_ext_size[3];    // size of own mesh + halo

	iris_real ***m_phi;   // φ - the result; needs halo
	iris_real ***m_Ap;    // no need for halo
	iris_real ***m_p;     // needs halo
	iris_real ***m_r;     // no need for halo

	class haloex *m_phi_haloex;
	class haloex *m_p_haloex;
    };
}

#endif
