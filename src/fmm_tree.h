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
#ifndef __IRIS_FMM_TREE_H__
#define __IRIS_FMM_TREE_H__

#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

    // The tree has a depth (e.g. tree with depth 3 has levels 0, 1 and 2).
    // For each level we store arrays of cells, sorted by their cellID.
    // Below all cells live the particles. The link between parents and children
    // is formed in the cell structure by using the first_child/num_children.
    
    class fmm_tree : protected state_accessor {

    public:

	fmm_tree(class iris *in_iris);
	~fmm_tree();

	int max_level() { return m_depth - 1; }
	int depth() { return m_depth; }
	int nterms() { return m_nterms; }
	void set_leaf_size();
	
	void charges2particles();
	void particles2leafs();
	void children2parent(int level);
	void eval_p2m();
	void eval_m2m(iris_real *in_scratch);
	void bottom_up();
	
    private:
	void free_cells();
	void determine_depth();
	void print_multipoles(int cellID, iris_real *gamma);

	int                m_depth;       // # of items in arrays below
	int                m_nterms;      // number of items in the multipole expansions
	iris_real          m_leaf_size[3];  // size of leaf cells
	int               *m_ncells;      // number of cells on each level
	struct cell_t    **m_cells;       // arrays of cells, one for each level
	int                m_nparticles;  // number of particle
	struct particle_t *m_particles;   // array of particles themselves
	
    };
}

#endif
