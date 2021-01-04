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
#include <vector>
#include "solver.h"
#include "real.h"
#include "memory.h"
#include "domain.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "assert.h"
#include "logger.h"

#ifdef IRIS_CUDA
#include "cuda_runtime_api.h"
#include "fmm_gpu_halloc.h"
#endif

namespace ORG_NCSA_IRIS {


    // In order to avoid run-time memory allocation (VERY SLOW on CUDA), we
    // set the maximum # of particles per cell in compile time. This should
    // be more than enough (e.g. if NCRIT=64 - the default - there are about
    // 10 particles per cell)
#define IRIS_MAX_NCRIT 64
    
#ifdef IRIS_CUDA
#define IRIS_CUDA_FMM_NUM_STREAMS 4

#endif
    
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
	
	inline void load_particles();
	void load_particles_cpu();
#ifdef IRIS_CUDA
	void load_particles_gpu();
#endif
	
	inline void distribute_particles(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
	void distribute_particles_cpu(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
#ifdef IRIS_CUDA
	void distribute_particles_gpu(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
#endif

	inline void link_parents(cell_t *io_cells);
	void link_parents_cpu(cell_t *io_cells);
#ifdef IRIS_CUDA
	void link_parents_gpu(cell_t *io_cells);
#endif
	
	inline void relink_parents(cell_t *io_cells);
	void relink_parents_cpu(cell_t *io_cells);
#ifdef IRIS_CUDA
	void relink_parents_gpu(cell_t *io_cells);
#endif
	
	inline void eval_p2m(cell_t *in_cells, bool alien_only);
	void eval_p2m_cpu(cell_t *in_cells, bool alien_only);
#ifdef IRIS_CUDA
	void eval_p2m_gpu(cell_t *in_cells, bool alien_only);
#endif
	
	inline void eval_m2m(cell_t *in_cells, bool alien_only);
	void eval_m2m_cpu(cell_t *in_cells, bool alien_only);
#ifdef IRIS_CUDA
	void eval_m2m_gpu(cell_t *in_cells, bool alien_only);
#endif
	
	void eval_p2p(int srcID, int destID, int ix, int iy, int iz);
	void eval_m2l(int srcID, int destID, int ix, int iy, int iz);
	
	void eval_p2p_cpu();
	void eval_m2l_cpu();
	
#ifdef IRIS_CUDA
	void eval_p2p_gpu();
	void eval_m2l_gpu();
#endif
	
	inline void eval_l2l();
	void eval_l2l_cpu();
#ifdef IRIS_CUDA
	void eval_l2l_gpu();
#endif

	inline void eval_l2p();
	void eval_l2p_cpu();
#ifdef IRIS_CUDA
	void eval_l2p_gpu();
#endif
	
	void exchange_LET();
	inline void comm_LET();
	int comm_LET_cpu(cell_t *in_cells, iris_real *in_M);
#ifdef IRIS_CUDA
	int comm_LET_gpu();
#endif
	void recalculate_LET();
	void get_LET(int rank, int cellID, unsigned char *sendbuf, int unit_size, int *out_cits, cell_t *in_cells, iris_real *in_M);
	void inhale_xcells(int in_count);
#ifdef IRIS_CUDA
	void inhale_xcells_gpu(int in_count);
#endif
	
	void print_tree(const char *label, cell_t *in_cells, int in_level, iris_real *in_M);

#ifdef IRIS_CUDA
	void print_tree_gpu(const char *label, cell_t *in_cells);
#endif

	inline void dual_tree_traversal();
	void dual_tree_traversal_cpu(cell_t *src_cells, cell_t *dest_cells);
#ifdef IRIS_CUDA
	void dual_tree_traversal_gpu();
#endif
	
	void traverse_queue(cell_t *src_cells, cell_t *dest_cells, int ix, int iy, int iz);
	void interact(cell_t *src_cells, cell_t *dest_cells, int srcID, int destID, int ix, int iy, int iz);

	inline void compute_energy_and_virial();
	void compute_energy_and_virial_cpu();
#ifdef IRIS_CUDA
	void compute_energy_and_virial_gpu();
#endif
	
	void send_forces_to(particle_t *in_particles, int peer, int start, int end, bool include_energy_virial);


	inline void send_back_forces();
#ifdef IRIS_CUDA
	void send_back_forces_gpu();
#endif
	void send_back_forces_cpu(particle_t *in_particles);

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
	iris_real          *m_M;                 // Multipole expansions, one per cell
	iris_real          *m_L;                 // Local expansions, one per cell
	struct cell_t      *m_cells;             // Local tree (actually cell_t *)
	struct cell_t      *m_xcells;            // Local essential tree (cell_t *, local tree + needed parts from other ranks)
	int                 m_nparticles;        // number of particle
	int                 m_npart_cap;         // capacity of the allocated array of particles
	struct particle_t  *m_particles;         // array of particles themselves
	int                 m_nxpart_cap;        // capacity of the allocated array of halo particles
	int                 m_nxparticles;       // number of halo particles
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

	box_t<iris_real> m_ext_box;
	box_t<iris_real> *m_ext_boxes;

#ifdef IRIS_CUDA
	std::map<int, iris_real *> m_charges_gpu;
	std::map<int, int> m_charges_gpu_cap;
	cudaStream_t m_streams[IRIS_CUDA_FMM_NUM_STREAMS];

	int *m_atom_types;
	int m_at_cap;
	
	int *m_cellID_keys;
	int m_cellID_keys_cap;

	struct cell_t *m_cells_cpu;
	bool m_has_cells_cpu;
	struct cell_t *m_xcells_cpu;
	iris_real *m_M_cpu;

	unsigned char *m_recvbuf_gpu;
	int m_recvbuf_gpu_cap;

	std::vector<struct interact_item_t, HostAlloc<interact_item_t>> m_p2p_list;
	std::vector<struct interact_item_t, HostAlloc<interact_item_t>> m_m2l_list;

	struct interact_item_t *m_p2p_list_gpu;
	struct interact_item_t *m_m2l_list_gpu;
	int m_p2p_list_cap;
	int m_m2l_list_cap;

	iris_real *m_evir_gpu;
	particle_t *m_particles_cpu;
	int m_particles_cpu_cap;
	
#endif

	unsigned char *m_sendbuf;
	unsigned char *m_recvbuf;
	int m_sendbuf_cap;
	int m_recvbuf_cap;
    };
}

#endif
