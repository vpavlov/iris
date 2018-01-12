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
#ifndef __IRIS_PROC_GRID_H__
#define __IRIS_PROC_GRID_H__

#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

    class proc_grid : protected state_accessor {

    public:
	proc_grid(class iris *obj);
	~proc_grid();

	void set_pref(int x, int y, int z);
	void commit();

    private:
	void select_grid_size();
	int select_best_factor(int n, int **factors, int *out_best);
	void setup_grid_details();
	void setup_splits();

    public:

	int m_size[3];     // MxNxK procs in each direction
	int m_coords[3];   // This process' coords in the grid

	// Process neighbourhood: Each proc has neighbours, which
	// are stored in the m_hood array. The index in the array can be
	// treated as a number in ternary numeral system. Each trit
	// corresponds to the location of the proc in relation to me:
	// 0 = same coord, 1 = one below, 2 = one above.
	// Rightmost trit is for x dir, Middle is for y, Leftmost for z
	// 
	// For example, index 14 (in decimal) = 112 (in ternary), which means
	// that m_hood[14] is the processor that is right/bottom/front of me
	// 
	// Also note that m_hood[0] = me
	int m_hood[27];

	iris_real *m_xsplit;    // M ranges (rel 0 - 1) for each proc in X dir
	iris_real *m_ysplit;    // N ranges (rel 0 - 1) for each proc in Y dir
	iris_real *m_zsplit;    // K ranges (rel 0 - 1) for each proc in Z dir

    private:
	bool m_dirty;      // if we need to re-calculate upon commit
	int ***m_ranks;  // = rank of the proc at [i][j][k] point in grid
	int m_pref[3];   // User preference about procs in each dir
    };
}

#endif
