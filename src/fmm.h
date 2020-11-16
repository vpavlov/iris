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

    struct particle_t {
	int rank;    // from which rank this came?
	int index;   // which idx in m_charges{rank} this has ?
	size_t cellID;  // in which cell this resides
    };

    struct leaf_t {
	size_t cellID;
	int first_particle;  // index in particle_t array where particles in this cell start
	int num_particles;   // number of particles in this leaf
	iris_real *m;        // multipole expansion (complex)
	iris_real *l;        // multipole expansion (complex)

	// dummy is needed for memory::create_1d's array[i] = (T)0 (which is not used anyway)
	// this is here just to satisfy the compiler
	leaf_t(int dummy = 0) : m(NULL), l(NULL) {};  // dummy is needed for memory::create_1d's 

	~leaf_t() {
	    memory::destroy_1d(m);
	    memory::destroy_1d(l);
	}
	
    };
	
    class fmm : public solver {

    public:
	fmm(class iris *obj);
	~fmm();

	void commit();
	void solve();
	void handle_box_resize();

    private:
	void set_max_level();     // sets m_max_level based on m_natoms and m_ncrit
	void set_leaf_size();     // sets the m_leaf_size member based on m_max_level
	void set_natoms_local();  // get the number of local atoms (sum m_iris->m_ncharges)
	
	particle_t *charges2particles();
	leaf_t *particles2leafs(particle_t *in_particles, size_t *out_nleafs);
	void eval_p2m(leaf_t *in_leafs, particle_t *in_particles, size_t in_nleafs);
	void p2m(iris_real in_x, iris_real in_y, iris_real in_z, iris_real in_q, iris_real *out_gamma);
	void print_multipoles(size_t cellID, iris_real *m);

    private:
	size_t    m_natoms;        // total number of atoms (in all processors)
	size_t    m_natoms_local;  // number of atoms on this processor
	short     m_order;         // order of expansion (p)
	short     m_nterms;        // nubmer of multipole terms (p*(p+1)/2)
	short     m_ncrit;         // average number of particles in leafs
	short     m_max_level;     // tree depth-1
	iris_real m_leaf_size[3];  // size of 
	size_t    m_first_leafID;  // where leaf indices start
	int       m_num_leafs_1D;  // number of leafs in each dimension
	std::map<size_t, iris_real[3]> m_leaf_centers;
    };
}

#endif
