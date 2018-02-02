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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_GRID_H__
#define __IRIS_GRID_H__

#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

    class grid : protected state_accessor {

    public:
	grid(class iris *obj, const char *in_name);
	~grid();

	void set_pref(int x, int y, int z);
	virtual void commit();

    protected:
	void select_grid_size();
	int select_best_factor(int n, int **factors, int *out_best);
	void setup_grid_details();
	void setup_splits();

    public:

	char *m_name;      // Name of the grid (used in logging)
	int m_size[3];     // MxNxK procs in each direction
	int m_coords[3];   // This process' coords in the grid
	int m_hood[3][2];  // for each of the 3 directions, top/bottom neighbour

	iris_real *m_xsplit;    // M ranges (rel 0 - 1) for each proc in X dir
	iris_real *m_ysplit;    // N ranges (rel 0 - 1) for each proc in Y dir
	iris_real *m_zsplit;    // K ranges (rel 0 - 1) for each proc in Z dir

    protected:
	bool m_dirty;      // if we need to re-calculate upon commit
	int ***m_ranks;  // = rank of the proc at [i][j][k] point in grid
	int m_pref[3];   // User preference about procs in each dir
    };
}

#endif
