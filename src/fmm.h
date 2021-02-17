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
#include "fmm_pair.h"
#include "timer.h"
#include "openmp.h"

#ifdef IRIS_CUDA
#include "cuda_runtime_api.h"
#include "fmm_gpu_halloc.h"
#endif

namespace ORG_NCSA_IRIS {


#define IRIS_MAX_NCRIT 64
#define IRIS_FMM_MAX_ORDER 20
    
#ifdef IRIS_CUDA
#define IRIS_CUDA_FMM_NUM_STREAMS 4

#endif
    
    class fmm : public solver {

    public:
	fmm(class iris *obj);
	~fmm();

#ifdef IRIS_CUDA
	void cuda_specific_construct();
	void cuda_specific_commit();
	void cuda_specific_step_init();
#endif
	
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

	void distribute_particles(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
	void distribute_xparticles(struct xparticle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
	
#ifdef IRIS_CUDA
	void distribute_particles_gpu_v1(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
	void distribute_xparticles_gpu_v1(struct xparticle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);

	void distribute_particles_gpu(struct particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
	void distribute_xparticles_gpu(struct xparticle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target);
#endif


	static int __compare_cellID(const void *a, const void *b)
	{
	    particle_t *aptr = (particle_t *)a;
	    particle_t *bptr = (particle_t *)b;
	    if(aptr->cellID < bptr->cellID) {
		return -1;
	    }else if(aptr->cellID > bptr->cellID) {
		return 1;
	    }else {
		return 0;
	    }
	}

	template <typename T>
	void distribute_particles_cpu(T *in_particles, int in_count, int in_flags, struct cell_t *out_target)
	{
	    timer tm;
	    tm.start();
	    
	    if(in_count == 0) {
		return;
	    }

	    int nleafs = (1 << 3 * max_level());
	    int offset = cell_meta_t::offset_for_level(max_level());
	    
#if defined _OPENMP
#pragma omp parallel
#endif
	    {
		int from, to;
		setup_work_sharing(nleafs, m_iris->m_nthreads, &from, &to);
		for(int i=from;i<to;i++) {
		    int cellID = offset + i;
		    T key;
		    key.cellID = cellID;
		    T *tmp = (T *)bsearch(&key, in_particles, in_count, sizeof(T), __compare_cellID);
		    if(tmp == NULL) {
			continue;
		    }
		    
		    int left = tmp - in_particles;
		    int right = left;
		    
		    while(left > 0 && in_particles[left].cellID >= cellID)             { left--; }
		    while(left < in_count && in_particles[left].cellID < cellID)       { left++; }
		    while(right < in_count-1 && in_particles[right].cellID <= cellID)  { right++; }
		    while(right >= 0 && in_particles[right].cellID > cellID)           { right--; }
		    
		    int num_children = (right - left + 1);
		    if(num_children <= 0) {
			continue;
		    }
		    
		    out_target[cellID].first_child = left;
		    out_target[cellID].num_children = num_children;
		    out_target[cellID].flags = in_flags;
		    // out_target[cellID].ses.c.r[0] = m_cell_meta[cellID].geomc[0];
		    // out_target[cellID].ses.c.r[1] = m_cell_meta[cellID].geomc[1];
		    // out_target[cellID].ses.c.r[2] = m_cell_meta[cellID].geomc[2];
		    // out_target[cellID].ses.r = m_cell_meta[cellID].maxr;
		    out_target[cellID].compute_com(in_particles);
		    
#if defined _OPENMP
#pragma omp critical
		    {
#endif
			m_max_particles = MAX(m_max_particles, num_children);
#if defined _OPENMP
		    }
#endif
		}
	    }
	    
	    tm.stop();
	    m_logger->time("Distribute particles wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
	};

	
	
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
	
	inline void eval_p2m(cell_t *in_cells);
	void eval_p2m_cpu(cell_t *in_cells);
#ifdef IRIS_CUDA
	void eval_p2m_gpu(cell_t *in_cells);
#endif
	
	inline void eval_m2m(cell_t *in_cells, bool alien_only);
	void eval_m2m_cpu(cell_t *in_cells, bool alien_only);
#ifdef IRIS_CUDA
	void eval_m2m_gpu(cell_t *in_cells, bool alien_only);
#endif
	
	void eval_p2p(int srcID, int destID, int ix, int iy, int iz);
	void eval_p2p_self();
	void eval_m2l(int srcID, int destID, int ix, int iy, int iz);
	
	void eval_p2p_cpu();
	void eval_p2p_self_cpu();
	void eval_m2l_cpu();
	
#ifdef IRIS_CUDA
	void eval_p2p_gpu();
	void eval_p2p_self_gpu();
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

	
	inline void exchange_p2p_halo();
#ifdef IRIS_CUDA
	void exchange_p2p_halo_gpu();
#endif
	void exchange_p2p_halo_cpu();

	
	void send_particles_to_neighbour_cpu(int rank, std::vector<xparticle_t> *out_sendbuf, MPI_Request *out_cnt_req, MPI_Request *out_data_req);
#ifdef IRIS_CUDA
	void send_particles_to_neighbour_gpu(int rank, void *out_sendbuf_gpu, std::vector<xparticle_t> *out_sendbuf_cpu,
					     MPI_Request *out_cnt_req, MPI_Request *out_data_req, cudaStream_t &stream,
					     int *in_halo_cnt, int *in_halo_disp);
#endif
	void recv_particles_from_neighbour_cpu(int rank, int alien_index, int alien_flag);
#ifdef IRIS_CUDA
	void recv_particles_from_neighbour_gpu(int rank, int alien_index, int alien_flag);
#endif
	
	void border_leafs(int rank);
	
	inline void comm_LET();
	int comm_LET_cpu(cell_t *in_cells, iris_real *in_M);
#ifdef IRIS_CUDA
	int comm_LET_gpu();
#endif
	void recalculate_LET();
	void get_LET(int rank, int cellID, unsigned char *sendbuf, int unit_size, int *out_cits, cell_t *in_cells, iris_real *in_M, std::map<int, int> *scheduled_parents);
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
	void send_back_forces_cpu(particle_t *in_particles, bool sort);

	void calc_ext_boxes();

	void do_m2l_interact(int srcID, int destID, int ix, int iy, int iz);
	void do_p2p_interact(int srcID, int destID, int ix, int iy, int iz);
	void do_p2p_interact_pbc(int srcID, int destID, int ix, int iy, int iz);
	void do_p2p_interact_nopbc(int srcID, int destID);
	
    private:
	int                 m_order;             // order of expansion (p)
	int                 m_depth;             // depth of the tree (max level + 1)
	iris_real           m_mac;               // Multipole acceptance criteria
	int                 m_nterms;            // number of items in the multipole expansions
	int                 m_local_root_level;  // the level of the local root (contains all subnodes here)
	iris_real           m_leaf_size[3];      // size of leaf cells

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
	size_t              m_npart_cap;         // capacity of the allocated array of particles
	struct particle_t  *m_particles;         // array of particles themselves
	size_t              m_nxpart_cap;        // capacity of the allocated array of halo particles
	int                 m_nxparticles;       // number of halo particles
	struct xparticle_t *m_xparticles[6];     // halo particles
	size_t              m_xparticles_cap[6];

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

	std::vector<struct interact_item_t> m_p2p_list;
	std::vector<struct interact_item_t> m_m2l_list;
	
#ifdef IRIS_CUDA
	std::map<int, iris_real *> m_charges_gpu;
	std::map<int, size_t> m_charges_gpu_cap;
	cudaStream_t m_streams[IRIS_CUDA_FMM_NUM_STREAMS];
	cudaEvent_t m_m2l_memcpy_done;
	cudaEvent_t m_p2p_memcpy_done;	
	
	int    *m_atom_types;
	size_t m_at_cap;
	
	int    *m_keys;
	size_t m_keys_cap;

	struct cell_t *m_cells_cpu;
	bool m_has_cells_cpu;
	struct cell_t *m_xcells_cpu;
	iris_real *m_M_cpu;

	unsigned char *m_recvbuf_gpu;
	size_t m_recvbuf_gpu_cap;

	struct interact_item_t *m_p2p_list_gpu;
	struct interact_item_t *m_m2l_list_gpu;
	size_t m_p2p_list_cap;
	size_t m_m2l_list_cap;

	iris_real *m_evir_gpu;
	particle_t *m_particles_cpu;
	size_t m_particles_cpu_cap;
	int *m_max_particles_gpu;

	void *m_halo_parts_gpu[2];

#endif

	int m_max_particles;
	unsigned char *m_sendbuf;
	unsigned char *m_recvbuf;
	size_t m_sendbuf_cap;
	size_t m_recvbuf_cap;

	std::map<struct pair_t, bool, pair_comparator_t> m_p2p_skip;
	std::map<struct pair_t, bool, pair_comparator_t> m_m2l_skip;

	int *m_halo_cell_cnt[2];
	int *m_halo_cell_disp[2];
	
	std::vector<struct xparticle_t> m_border_parts[2];
	
	iris_real m_let_corr;

	
	int collect_halo_for(int rank, int hwm);
	std::vector<int> m_a2a_cell_cnt;
	std::vector<int> m_a2a_cell_disp;
	std::vector<int> m_a2a_send_cnt;
	std::vector<int> m_a2a_send_disp;
	std::vector<int> m_a2a_recv_cnt;
	std::vector<int> m_a2a_recv_disp;
	std::vector<xparticle_t> m_a2a_sendbuf;

#ifdef IRIS_CUDA
	int collect_halo_for_gpu(int rank, int hwm);
	void *m_a2a_cell_cnt_gpu;
	void *m_a2a_cell_disp_gpu;
	void *m_a2a_sendbuf_gpu;
	std::vector<struct xparticle_t, HostAlloc<struct xparticle_t>> m_a2a_recvbuf;
	std::vector<struct xparticle_t, HostAlloc<struct xparticle_t>> m_a2a_sendbuf_cpu;
#endif
	
    };
}

#endif
