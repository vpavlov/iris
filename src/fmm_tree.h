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
	fmm_tree(fmm_tree *in_ther);
	
	~fmm_tree();

	int max_level() { return m_depth - 1; }
	int depth() { return m_depth; }
	int nterms() { return m_nterms; }
	int local_root_level() { return m_local_root_level; }
	
	void set_leaf_size();

	void compute_local(iris_real *in_scratch);
	fmm_tree *compute_LET(box_t<iris_real> *in_local_boxes);
	
    private:

	void debug(int i, int j, char *msg);
	void exchange_p2p_halo(box_t<iris_real> *in_local_boxes);
	void exchange_rest_of_LET(box_t<iris_real> *in_local_boxes);
	void get_LET(int rank, box_t<iris_real> *in_local_boxes, struct cell_t *start_from, int level, unsigned char *sendbuf, int unit_size, int *out_cits);
	
	void free_cells();
	void determine_depth();
	void print_multipoles(int cellID, iris_real *gamma);

	void create_particles();

	template <typename T>
	struct cell_t *create_leafs(T *in_particles, int in_nparticles, int *out_nleafs);
	    
	void children2parent(int level);
	cell_t *xchildren2parent(int level);
	void eval_p2m(cell_t *leafs, int nleafs);
	void eval_m2m(iris_real *in_scratch);
	void bottom_up();

	int *border_leafs(int rank, box_t<iris_real> *in_local_boxes, int *out_send_count, bool alien);
	void inhale_xparticles(struct xparticle_t *in_xparticles, int in_count);
	void inhale_xleafs(struct cell_t *in_xleafs, int in_count);
	void inhale_xcells(unsigned char *recvbuf, int in_count);
	
	void compute_leafs_to_send(box_t<iris_real> *in_local_boxes, int *&send_to_ranks, int *&send_cnt, int *&leafs_to_send, int *out_rank_count, int *out_hwm);
	void compute_particle_send_size(int *send_to_ranks, int *send_cell_count, int *leafs_to_send, int rank_size,
					struct pit_t *&send_buf, int *&send_body_count, int *&send_body_disp, int *&recv_body_count);
	
	
	
	void recv_particles_from_neighbour(int rank, int count_tag, int data_tag);
	struct particle_t *alien_charges2particles(iris_real *in_data, int in_count);

	

	
    };
}

#endif
