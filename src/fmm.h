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
#include <deque>
#include <cstring>
#include "solver.h"
#include "real.h"
#include "memory.h"
#include "domain.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "assert.h"
#include "logger.h"

namespace ORG_NCSA_IRIS {
	
    class fmm : public solver {

    public:
	fmm(class iris *obj);
	~fmm();

	void commit();
	void solve();
	void handle_box_resize();
	box_t<iris_real> *get_ext_boxes();
	
	int max_level() { return m_depth - 1; }
	
    private:
	void generate_cell_meta();
	
	void set_leaf_size();

	void local_tree_construction();
	void load_particles();
	
	void distribute_particles(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);

	void link_parents(cell_t *io_cells);
	void relink_parents(cell_t *io_cells);
	void eval_p2m(cell_t *in_cells, bool alien_only);
	void eval_m2m(cell_t *in_cells, bool alien_only);
	void eval_m2l(int srcID, int destID, int ix, int iy, int iz);
	void eval_p2p(int srcID, int destID, int ix, int iy, int iz);
	void eval_l2l();
	void eval_l2p();
	
	void exchange_LET();
	void exchange_rest_of_LET();
	void recalculate_LET();
	void get_LET(int rank, int cellID, unsigned char *sendbuf, int unit_size, int *out_cits);
	void inhale_xcells(unsigned char *recvbuf, int in_count);
	void print_tree(const char *label, cell_t *in_cells, int in_level);

	void dual_tree_traversal();
	void traverse_queue(int ix, int iy, int iz);
	void interact(int srcID, int destID, int ix, int iy, int iz);
	void compute_energy_and_virial();
	void send_forces_to(int peer, int start, int end, bool include_energy_virial);
	void send_back_forces();

	void calc_ext_boxes();
	
	
    private:
	int                 m_order;             // order of expansion (p)
	int                 m_depth;             // depth of the tree (max level + 1)
	iris_real           m_mac;               // Multipole acceptance criteria
	iris_real           m_mac_let_corr;      // Correction for MAC for calculating LET
	int                 m_nterms;            // number of items in the multipole expansions
	int                 m_local_root_level;  // the level of the local root (contains all subnodes here)
	iris_real           m_leaf_size[3];      // size of leaf cells
	iris_real          *m_scratch;           // M2M/M2L scratch space

	int                *m_sendcnt;           // these four are for the all2all communication of the LET
	int                *m_senddisp;
	int                *m_recvcnt;
	int                *m_recvdisp;
	
	// THE "TREE"
	int                 m_tree_size;         // Pre-computed tree size
	struct cell_meta_t *m_cell_meta;         // Static cell data (e.g. center, radius)
	iris_real         **m_M;                 // Multipole expansions, one per cell
	iris_real         **m_L;                 // Local expansions, one per cell
	struct cell_t      *m_cells;             // Local tree (actually cell_t *)
	struct cell_t      *m_xcells;            // Local essential tree (cell_t *, local tree + needed parts from other ranks)
	int                 m_nparticles;        // number of particle
	struct particle_t  *m_particles;         // array of particles themselves
	int                 m_nxparticles;       // number of alien particles
	struct particle_t  *m_xparticles;        // halo particles
	bool                m_dirty;

	std::deque<struct pair_t> m_queue;       // the Dual Tree Traversal queue

	// statistics
	int m_p2m_count;
	int m_m2m_count;
	int m_m2l_count;
	int m_p2p_count;
	int m_l2l_count;
	int m_l2p_count;
	int m_p2m_alien_count;
	int m_m2m_alien_count;

	bool m_one_sided;
	MPI_Win m_Mwin;
	
	box_t<iris_real> m_ext_box;
	box_t<iris_real> *m_ext_boxes;
    };
}

#endif
