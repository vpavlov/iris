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
#define IRIS_FMM_CELL_ALIEN_L1   0x00002
#define IRIS_FMM_CELL_ALIEN_L2   0x00004
#define IRIS_FMM_CELL_ALIEN_L3   0x00008
#define IRIS_FMM_CELL_ALIEN_L4   0x00010
#define IRIS_FMM_CELL_ALIEN_L5   0x00020
#define IRIS_FMM_CELL_ALIEN_L6   0x00040
#define IRIS_FMM_CELL_ALIEN_NL   0x00080  // non-leaf alien cell
#define IRIS_FMM_CELL_HAS_CHILD1 0x00100  // child slot 1 is occupied
#define IRIS_FMM_CELL_HAS_CHILD2 0x00200  // child slot 2 is occupied
#define IRIS_FMM_CELL_HAS_CHILD3 0x00400  // ...
#define IRIS_FMM_CELL_HAS_CHILD4 0x00800
#define IRIS_FMM_CELL_HAS_CHILD5 0x01000
#define IRIS_FMM_CELL_HAS_CHILD6 0x02000
#define IRIS_FMM_CELL_HAS_CHILD7 0x04000
#define IRIS_FMM_CELL_HAS_CHILD8 0x08000
#define IRIS_FMM_CELL_VALID_M    0x10000  // cell has valid multipole expansion
#define IRIS_FMM_CELL_VALID_L    0x20000  // cell has valid local expansion
    
#define IRIS_FMM_CELL_ALIEN_LEAF (IRIS_FMM_CELL_ALIEN_L1 | IRIS_FMM_CELL_ALIEN_L2 | IRIS_FMM_CELL_ALIEN_L3 | IRIS_FMM_CELL_ALIEN_L4 | IRIS_FMM_CELL_ALIEN_L5 | IRIS_FMM_CELL_ALIEN_L6)
#define IRIS_FMM_CELL_ALIEN (IRIS_FMM_CELL_ALIEN_LEAF | IRIS_FMM_CELL_ALIEN_NL)
#define IRIS_FMM_CELL_HAS_CHILDREN (IRIS_FMM_CELL_HAS_CHILD1 | IRIS_FMM_CELL_HAS_CHILD2 | IRIS_FMM_CELL_HAS_CHILD3 | IRIS_FMM_CELL_HAS_CHILD4 | \
				    IRIS_FMM_CELL_HAS_CHILD5 | IRIS_FMM_CELL_HAS_CHILD6 | IRIS_FMM_CELL_HAS_CHILD7 | IRIS_FMM_CELL_HAS_CHILD8)
    
    // Once tree parameters (depth, etc.) are known, these can be calculated only once
    struct cell_meta_t {
	//int rank;
	iris_real geomc[3];  // geometrical center
	iris_real maxr;      // maximum theoretical radius (1/2 cell diagonal)
	
	cell_meta_t(int dummy = 0) {};  // to satisfy the compiler (memory::create_1d)
	void set(cell_meta_t *in_meta, int cellID, const box_t<iris_real> *in_gbox, iris_real *in_leaf_size, int in_max_level, int in_comm_size, int in_local_root_level);

	IRIS_CUDA_DEVICE_HOST static int offset_for_level(int level);
	IRIS_CUDA_DEVICE_HOST static int level_of(int in_cellID);
	IRIS_CUDA_DEVICE_HOST static int parent_of(int in_cellID);
	IRIS_CUDA_DEVICE_HOST static int leaf_coords_to_ID(int lx, int ly, int lz, int max_level);
	IRIS_CUDA_DEVICE_HOST static void leaf_ID_to_coords(int cellID, int max_level, int *lx, int *ly, int *lz);
    };
    
    struct cell_t {
	int num_children;  // only for leafs: number of particles
	int first_child;   // only for leafs: index in particles/xparticles1..6
	sphere_t ses;      // smallest sphere enclosing all particles
	int flags;         // IRIS_FMM_CELL_*
	
	cell_t(int dummy = 0) {};  // to satisfy the compiler
	void compute_ses(struct particle_t *in_particles);

	template <typename T>
	void compute_com(T *in_particles)
	{
	    for(int i=0;i<num_children;i++) {
		ses.c.r[0] += in_particles[first_child+i].xyzq[0];
		ses.c.r[1] += in_particles[first_child+i].xyzq[1];
		ses.c.r[2] += in_particles[first_child+i].xyzq[2];
	    }
	    
	    ses.c.r[0] /= num_children;
	    ses.c.r[1] /= num_children;
	    ses.c.r[2] /= num_children;
	    
	    iris_real max_dist2 = 0.0;
	    for(int i=0;i<num_children;i++) {
		iris_real dx = in_particles[first_child+i].xyzq[0] - ses.c.r[0];
		iris_real dy = in_particles[first_child+i].xyzq[1] - ses.c.r[1];
		iris_real dz = in_particles[first_child+i].xyzq[2] - ses.c.r[2];
		iris_real dist2 = dx*dx + dy*dy + dz*dz;
		if(dist2 > max_dist2) {
		    max_dist2 = dist2;
		}
	    }
	    ses.r = sqrt(max_dist2);
	};
	
    };


}

#endif
