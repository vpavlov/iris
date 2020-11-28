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

#define IRIS_FMM_CELL_LOCAL      0x0001
#define IRIS_FMM_CELL_ALIEN0     0x0002  // non-leaf alien cell
#define IRIS_FMM_CELL_ALIEN1     0x0004  // leaf alien cell coming from left
#define IRIS_FMM_CELL_ALIEN2     0x0008  // leaf alien cell coming from right
#define IRIS_FMM_CELL_ALIEN3     0x0010  // ...
#define IRIS_FMM_CELL_ALIEN4     0x0020
#define IRIS_FMM_CELL_ALIEN5     0x0040
#define IRIS_FMM_CELL_ALIEN6     0x0080
#define IRIS_FMM_CELL_HAS_CHILD1 0x0100  // child slot 1 is occupied
#define IRIS_FMM_CELL_HAS_CHILD2 0x0200  // child slot 2 is occupied
#define IRIS_FMM_CELL_HAS_CHILD3 0x0400  // ...
#define IRIS_FMM_CELL_HAS_CHILD4 0x0800
#define IRIS_FMM_CELL_HAS_CHILD5 0x1000
#define IRIS_FMM_CELL_HAS_CHILD6 0x2000
#define IRIS_FMM_CELL_HAS_CHILD7 0x4000
#define IRIS_FMM_CELL_HAS_CHILD8 0x8000
    
#define IRIS_FMM_CELL_ALIEN_LEAF (IRIS_FMM_CELL_ALIEN1 | IRIS_FMM_CELL_ALIEN2 | IRIS_FMM_CELL_ALIEN3 | IRIS_FMM_CELL_ALIEN4 | IRIS_FMM_CELL_ALIEN5 | IRIS_FMM_CELL_ALIEN6)
#define IRIS_FMM_CELL_ALIEN (IRIS_FMM_CELL_ALIEN_LEAF | IRIS_FMM_CELL_ALIEN0)
#define IRIS_FMM_CELL_HAS_CHILDREN (IRIS_FMM_CELL_HAS_CHILD1 | IRIS_FMM_CELL_HAS_CHILD2 | IRIS_FMM_CELL_HAS_CHILD3 | IRIS_FMM_CELL_HAS_CHILD4 | \
				    IRIS_FMM_CELL_HAS_CHILD5 | IRIS_FMM_CELL_HAS_CHILD6 | IRIS_FMM_CELL_HAS_CHILD7 | IRIS_FMM_CELL_HAS_CHILD8)
    
    // Once tree parameters (depth, etc.) are known, these can be calculated only once
    struct cell_meta_t {
	iris_real center[3];
	iris_real radius;

	cell_meta_t(int dummy = 0) {};  // to satisfy the compiler (memory::create_1d)
	void set(int cellID, const box_t<iris_real> *in_gbox, iris_real *in_leaf_size, int in_max_level);

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
	
    };
    
    struct cell_t {
	int num_children;  // only for leafs: number of particles
	int first_child;   // only for leafs: index in particles/xparticles1..6
	int flags;         // IRIS_FMM_CELL_*
	cell_t(int dummy = 0) {};  // to satisfy the compiler
    };


}

#endif
