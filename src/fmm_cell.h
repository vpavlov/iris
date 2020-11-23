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
#ifndef __IRIS_FMM_CELL_H__
#define __IRIS_FMM_CELL_H__

#include "memory.h"
#include "real.h"
#include "box.h"

namespace ORG_NCSA_IRIS {

    // The FMM tree is comprised of cells. Each cell has a unique ID from which its position
    // in the tree can be uniquely determined. Cells are kept in arrays sorted by their cellID
    // for each level and each cell on a higher level keeps the index of its first child and
    // the number of children. If this is a cell on the lowest level, the first_child and
    // num_children are references to the array of particles instead.
    // Along with this, the cell also knows its center and the multipole and local expansions
    //
    // An alien leaf cell references particles coming from other processors. They are only
    // used for P2P calculation and first_child/num_children refers to m_alien_paricles array
    // of the tree...
    struct cell_t {
	int       cellID;
	int       first_child;
	int       num_children;
	iris_real center[3];
	iris_real radius;
	iris_real *m;
	iris_real *l;
	bool      alien_leaf;

	cell_t(int dummy = 0) : m(NULL), l(NULL), alien_leaf(false) {};  // dummy is needed for memory::create_1d

	~cell_t() {
	    memory::destroy_1d(m);
	    memory::destroy_1d(l);
	}

	//
	// CellID is constructed as an index in a virtual (non-existent) array, ordered
	// by level and then by row-major-offset of [nx][ny][nz] indices for the cells
	// for that level. Thus we have:
	//
	// CellID [0]: root cell    ( 1 level 0 cell )
	// CellID [1, 2, 3, ..., 8] ( 8 level 1 cells)
	// CellID [9, 10, ..., 72]  (64 level 2 cells)
	// ...etc.
	//
	// This function returns the CellID of the first cell for the given level.
	//
	static int offset_for_level(int level) { return ((1 << 3 * level)-1) / 7; };


	//
	// Given a cellID, determine the level of the cell
	//
	static int level_of(int in_cellID) {
	    int retval = -1;
	    for(int i=in_cellID;i>=0;i-=(1 << 3 * retval)) {
		retval++;
	    }
	    return retval;
	}
	
	//
	// Given a cellID, find the cellID of its parent
	//
	static int parent_of(int in_cellID)
	{
	    int level = level_of(in_cellID);
	    int curr_off = offset_for_level(level);
	    int parent_off = offset_for_level(level-1);
	    int retval = ((in_cellID - curr_off) >> 3) + parent_off;
	    return retval;
	}


	void set_center(const box_t<iris_real> *in_gbox, iris_real *in_leaf_size);
	void set_radius(iris_real *in_leaf_size, int in_max_level);

	static void sort(cell_t *in_out_data, int count, bool desc);
    };


}

#endif
