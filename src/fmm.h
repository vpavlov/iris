// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2021, the National Center for Supercomputing Applications
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
#ifndef __IRIS_FMM_H__
#define __IRIS_FMM_H__
#include <vector>
#include "solver.h"
#include "real.h"
#include "memory.h"

namespace ORG_NCSA_IRIS {
	
    class fmm : public solver {

    public:
	fmm(class iris *obj);
	~fmm();

	void commit();
	void solve();
	void handle_box_resize();

    private:

	void free_cells();
	void get_local_boxes();
	
	void p2m(iris_real in_x, iris_real in_y, iris_real in_z, iris_real in_q, iris_real *out_gamma);
	void m2m(iris_real x, iris_real y, iris_real z, iris_real *in_gamma, iris_real *out_gamma);
	
	void print_multipoles(int cellID, iris_real *m);

    private:
	int         m_order;         // order of expansion (p)
	iris_real  *m_m2m_scratch;   // M2M scratch space
	box_t<iris_real> *m_local_boxes;  // Local boxes from all ranks
	struct fmm_tree *m_local_tree;    // the local tree
	struct fmm_tree *m_LET;           // the local essential tree (includes contributions from other procs)
    };
}

#endif
