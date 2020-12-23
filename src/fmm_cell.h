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
#include "sphere.h"

namespace ORG_NCSA_IRIS {

#define IRIS_FMM_CELL_LOCAL      0x00001
#define IRIS_FMM_CELL_ALIEN_NL   0x00002  // non-leaf alien cell
#define IRIS_FMM_CELL_ALIEN_LEAF 0x00004  // leaf alien cell
#define IRIS_FMM_CELL_HAS_CHILD1 0x00008  // child slot 1 is occupied
#define IRIS_FMM_CELL_HAS_CHILD2 0x00010  // child slot 2 is occupied
#define IRIS_FMM_CELL_HAS_CHILD3 0x00020  // ...
#define IRIS_FMM_CELL_HAS_CHILD4 0x00040
#define IRIS_FMM_CELL_HAS_CHILD5 0x00080
#define IRIS_FMM_CELL_HAS_CHILD6 0x00100
#define IRIS_FMM_CELL_HAS_CHILD7 0x00200
#define IRIS_FMM_CELL_HAS_CHILD8 0x00400
#define IRIS_FMM_CELL_VALID_M    0x00800  // cell has valid multipole expansion
#define IRIS_FMM_CELL_VALID_L    0x01000  // cell has valid local expansion
    
#define IRIS_FMM_CELL_ALIEN (IRIS_FMM_CELL_ALIEN_LEAF | IRIS_FMM_CELL_ALIEN_NL)
#define IRIS_FMM_CELL_HAS_CHILDREN (IRIS_FMM_CELL_HAS_CHILD1 | IRIS_FMM_CELL_HAS_CHILD2 | IRIS_FMM_CELL_HAS_CHILD3 | IRIS_FMM_CELL_HAS_CHILD4 | \
				    IRIS_FMM_CELL_HAS_CHILD5 | IRIS_FMM_CELL_HAS_CHILD6 | IRIS_FMM_CELL_HAS_CHILD7 | IRIS_FMM_CELL_HAS_CHILD8)
    
    // Once tree parameters (depth, etc.) are known, these can be calculated only once
    struct cell_meta_t {
	int rank;
	iris_real geomc[3];  // geometrical center
	iris_real maxr;      // maximum theoretical radius (1/2 cell diagonal)
	
	cell_meta_t(int dummy = 0) {};  // to satisfy the compiler (memory::create_1d)
	void set(cell_meta_t *in_meta, int cellID, const box_t<iris_real> *in_gbox, iris_real *in_leaf_size, int in_max_level, int in_comm_size, int in_local_root_level);

	IRIS_CUDA_DEVICE_HOST static int offset_for_level(int level);
	IRIS_CUDA_DEVICE_HOST static int level_of(int in_cellID);
	IRIS_CUDA_DEVICE_HOST static int parent_of(int in_cellID);
	
    };
    
    struct cell_t {
	sphere_t ses;      // smallest sphere enclosing all particles
	int num_children;  // only for leafs: number of particles
	int first_child;   // only for leafs: index in particles/xparticles1..6
	int flags;         // IRIS_FMM_CELL_*
	cell_t(int dummy = 0) {};  // to satisfy the compiler

	void compute_ses(struct particle_t *in_particles);
    };


}

#endif
