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
#include <math.h>
#include "fmm_tree.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "fmm_kernels.h"
#include "memory.h"
#include "domain.h"
#include "comm_rec.h"
#include "logger.h"
#include "timer.h"
#include "proc_grid.h"
#include "tags.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

#define _LN8 2.0794415416798357  // natural logarithm of 8

#define MIN_DEPTH 2   // minimum value for depth
#define MAX_DEPTH 16  // more than enough (e.g. 18 quadrillion particles)


fmm_tree::fmm_tree(iris *in_iris):
    state_accessor(in_iris),
    m_nparticles(0), m_particles(NULL),
    m_nxparticles(0), m_xparticles(NULL),
    m_leaf_size{0.0, 0.0, 0.0}
{
    solver_param_t t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC);
    m_mac = t.r;
    
    m_nterms = (m_iris->m_order + 1) * (m_iris->m_order + 2) / 2;
    determine_depth();
    
    memory::create_1d(m_ncells, m_depth, true);
    memory::create_1d(m_cells, m_depth, true);
    memory::create_1d(m_nxcells, m_depth, true);
    memory::create_1d(m_xcells, m_depth, true);
    
    // particles will be created in due time

}


fmm_tree::~fmm_tree()
{
    free_cells();
    memory::destroy_1d(m_ncells);
    memory::destroy_1d(m_nxcells);
    memory::destroy_1d(m_particles);
    memory::destroy_1d(m_xparticles);
}


void fmm_tree::free_cells()
{
    if(m_cells != NULL) {
	for(int i=0;i<m_depth;i++) {
	    memory::destroy_1d(m_cells[i]);
	}
	memory::destroy_1d(m_cells);
    }

    if(m_xcells != NULL) {
	for(int i=0;i<m_depth;i++) {
	    memory::destroy_1d(m_xcells[i]);
	}
	memory::destroy_1d(m_xcells);
    }
}


//
// Leaf size is determined by dividing the length of the global box in each dimension
// by the number of leaf cells per dimension.
//
void fmm_tree::set_leaf_size()
{
    int nd = 1 << max_level();

    m_leaf_size[0] = m_domain->m_global_box.xsize / nd;
    m_leaf_size[1] = m_domain->m_global_box.ysize / nd;
    m_leaf_size[2] = m_domain->m_global_box.zsize / nd;
}


//
// The maximum tree level of the FMM tree is determined so as each leaf cell has
// around ncrit particles (in case of homogenous distribution). This means
// log_8 (natoms/ncrit).
//
void fmm_tree::determine_depth()
{
    int natoms = m_iris->m_natoms;  // atoms won't change during simulation (hopefully)
    solver_param_t t = m_iris->get_solver_param(IRIS_SOLVER_FMM_NCRIT);
    int ncrit = t.i;
    
    m_depth = (natoms > ncrit) ? int(log(natoms / ncrit)/_LN8) + 1 : 0;
    m_depth = MAX(m_depth, MIN_DEPTH);
    m_depth = MIN(m_depth, MAX_DEPTH);
}


//
// Transform the list of charges on this processor to a list of 'particles'.
//
void fmm_tree::create_particles()
{
    int offset = cell_t::offset_for_level(max_level());
    int nd = 1 << max_level();

    m_nparticles = m_iris->num_local_atoms();
    memory::destroy_1d(m_particles);
    memory::create_1d(m_particles, m_nparticles);

    box_t<iris_real> *gbox = &m_domain->m_global_box;
    
    int n = 0;
    for(auto it = m_iris->m_ncharges.begin(); it != m_iris->m_ncharges.end(); it++) {
	int ncharges = it->second;
	iris_real *charges = m_iris->m_charges[it->first];

	for(int i=0;i<ncharges;i++) {
	    iris_real tx = (charges[i * 5 + 0] - gbox->xlo) / m_leaf_size[0];
	    iris_real ty = (charges[i * 5 + 1] - gbox->ylo) / m_leaf_size[1];
	    iris_real tz = (charges[i * 5 + 2] - gbox->zlo) / m_leaf_size[2];
	    
	    int ix = (int) tx;
	    int iy = (int) ty;
	    int iz = (int) tz;
	    
	    // row-major order of [ix][iy][iz] + offset for this level
	    int leaf_index = iz + nd * (iy + nd * ix);
	    int cellID = offset + leaf_index;

	    m_particles[n].rank = it->first;
	    m_particles[n].index = i;
	    m_particles[n].cellID = cellID;
	    memcpy(m_particles[n].xyzq, charges+i*5, 4*sizeof(iris_real));
	    n++;
	}
    }
    
    // sort the final list by cellID desending
    sort_particles(m_particles, m_nparticles, true);
}


#define FROB						\
    parents[n].cellID = last;				\
    parents[n].first_child = first_child;		\
    parents[n].num_children = num_children;		\
    parents[n].set_center(gbox, m_leaf_size);		\
    parents[n].set_radius(m_leaf_size, max_level());	\
    memory::create_1d(parents[n].m, 2*m_nterms, true);	\
    memory::create_1d(parents[n].l, 2*m_nterms, true)


//
// Based on the sorted list of particles, create the leaf structure of the tree
// A leaf cell has an ID, reference to the first particle in the sorted array,
// number of particles and a set of multipole and local expansions
//
template <typename T>
cell_t *fmm_tree::create_leafs(T *in_particles, int in_nparticles, int *out_nleafs)
{
    box_t<iris_real> *gbox = &m_domain->m_global_box;
    
    // theoretical maximum number of leafs here: number of leafs everywhere
    // divided by the # of server ranks (every rank has equal volume)
    // will be realloced at the end of this function
    int nparents = int((1 << 3*max_level()) / m_iris->m_local_comm->m_size) + 1;

    cell_t *parents;
    memory::create_1d(parents, nparents);

    int last = in_particles[0].cellID;
    int first_child = 0;
    int num_children = 0;
    int n = 0;
    for(int i=0;i<in_nparticles;i++) {
	if(in_particles[i].cellID != last) {
	    FROB;
	    first_child = i;
	    num_children = 0;
	    last = in_particles[i].cellID;
	    n++;
	}
	num_children++;
    }
    FROB;
    nparents = ++n;
    parents = (cell_t *)memory::wrealloc(parents, nparents * sizeof(cell_t));
    eval_p2m(parents, nparents);
    
    (*out_nleafs) = nparents;
    return parents;
}

// TODO: this is *almost* the same as particles2leafs.
// Figure out a way to avoid code duplication...
void fmm_tree::children2parent(int in_level)
{
    const box_t<iris_real> *gbox = &(m_domain->m_global_box);
    
    int nchildren = m_ncells[in_level];
    cell_t *children = m_cells[in_level];

    // theoretical maximum number of cells here: number of cells everywhere
    // divided by the # of server ranks (every rank has equal volume)
    // will be realloced at the end of this function
    int nparents = int((1 << 3*in_level) / m_iris->m_local_comm->m_size) + 1;
    
    cell_t *parents;
    memory::create_1d(parents, nparents);

    int last = cell_t::parent_of(children[0].cellID);
    int first_child = 0;
    int num_children = 0;
    int n = 0;
    for(int i=0;i<nchildren;i++) {
	int this_parent = cell_t::parent_of(children[i].cellID);
	if(this_parent != last) {
	    FROB;
	    first_child = i;
	    num_children = 0;
	    last = this_parent;
	    n++;
	}
	num_children++;
    }
    FROB;
    nparents = ++n;
    parents = (cell_t *)memory::wrealloc(parents, nparents * sizeof(cell_t));
    m_cells[in_level-1] = parents;
    m_ncells[in_level-1] = nparents;
}

#undef FROB


//
// Perform P2M
//
void fmm_tree::eval_p2m(cell_t *leafs, int nleafs)
{
    for(int i=0;i<nleafs;i++) {
	for(int j=0;j<leafs[i].num_children;j++) {
	    int rank = m_particles[leafs[i].first_child+j].rank;
	    int index = m_particles[leafs[i].first_child+j].index;
	    iris_real x = m_particles[leafs[i].first_child+j].xyzq[0] - leafs[i].center[0];
	    iris_real y = m_particles[leafs[i].first_child+j].xyzq[1] - leafs[i].center[1];
	    iris_real z = m_particles[leafs[i].first_child+j].xyzq[2] - leafs[i].center[2];
	    iris_real q = m_particles[leafs[i].first_child+j].xyzq[3];
	    p2m(m_iris->m_order, x, y, z, q, leafs[i].m);
	}
    }
}


//
// Perform M2M
//
void fmm_tree::eval_m2m(iris_real *in_scratch)
{
    for(int level = max_level()-1;level>0;level--) {
	int ntcells = m_ncells[level];
	cell_t *tcells = m_cells[level];
	cell_t *scells = m_cells[level+1];
	for(int i = 0;i<ntcells;i++) {
	    cell_t *tcell = tcells + i;
	    for(int j=0;j<tcell->num_children;j++) {
		cell_t *scell = scells + tcell->first_child + j;
		iris_real x = scell->center[0] - tcell->center[0];
		iris_real y = scell->center[1] - tcell->center[1];
		iris_real z = scell->center[2] - tcell->center[2];
		memset(in_scratch, 0, 2*nterms()*sizeof(iris_real));
		m2m(m_iris->m_order, x, y, z, scell->m, tcell->m, in_scratch);
	    }
	    //print_multipoles(tcell->cellID, tcell->m);
	}
    }
}


void fmm_tree::bottom_up()
{
    for(int level = max_level();level > 0; level--) {
	children2parent(level);
    }
}


void fmm_tree::print_multipoles(int cellID, iris_real *gamma)
{
    for(int l=0;l<=m_iris->m_order;l++) {
	for(int m=0;m<=l;m++) {
	    int i = multipole_index(l, m);
	    m_logger->info("Cell %d M[%d][%d] = %f + i*%f", cellID, l, m, gamma[i], gamma[i+1]);
	}
    }
}


void fmm_tree::compute_local(iris_real *in_scratch)
{
    timer tm;
    tm.start();
    
    create_particles();

    int nleafs;
    m_cells[max_level()] = create_leafs(m_particles, m_nparticles, &nleafs);
    m_ncells[max_level()] = nleafs;
    
    bottom_up();  // this doesn't depend on P2M, so can be done in parallel
    eval_m2m(in_scratch);  // this needs on P2M and bottom up finished

    tm.stop();
    m_logger->info("FMM: Compute local tree wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}


//
// This function returns an array of indices into the m_cells[max_level()] array of leafs
// that are bordering the RANK processor. The number of entries in the result is returned
// in the out_send_count output parameter.
//
int *fmm_tree::border_leafs(int rank, box_t<iris_real> *in_local_boxes, int *out_send_count, bool alien)
{
    int level = max_level();
    cell_t *leafs;
    int nleafs;
    if(alien) {
	leafs = m_xcells[level];
	nleafs = m_nxcells[level];
    }else {
	leafs = m_cells[level];
	nleafs = m_ncells[level];
    }
    
    (*out_send_count) = 0;
    int *leafs_to_send;
    memory::create_1d(leafs_to_send, nleafs);
    
    for(int n = 0; n<nleafs; n++) {
	bool send = false;
	iris_real dn = leafs[n].radius;
	iris_real cx = leafs[n].center[0];
	iris_real cy = leafs[n].center[1];
	iris_real cz = leafs[n].center[2];
	for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	    for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
		for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		    iris_real x = cx + ix * m_domain->m_global_box.xsize;
		    iris_real y = cy + iy * m_domain->m_global_box.ysize;
		    iris_real z = cz + iz * m_domain->m_global_box.zsize;
		    iris_real rn = in_local_boxes[rank].distance_to(x, y, z);
		    if (dn/rn < m_mac) {
			continue;
		    }
		    // D(n)/r(n) >= θ - this means that this cell is too close to the border
		    // and is needed by the other processor to do P2P
		    send = true;
		    break;
		}
		if(send) {
		    break;
		}
	    }
	    if(send) {
		break;
	    }
	}
	if(send) {
	    leafs_to_send[(*out_send_count)++] = n;
	}
    }
    return (int *)memory::wrealloc(leafs_to_send, (*out_send_count) * sizeof(int));
}


void fmm_tree::send_particles_to_neighbour(int rank, box_t<iris_real> *in_local_boxes,
					   int count_tag, int data_tag,
					   int *out_part_count, xparticle_t *&out_sendbuf,
					   MPI_Request *out_cnt_req, MPI_Request *out_data_req)

{
    int level = max_level();
    cell_t *leafs = m_cells[level];
    cell_t *xleafs = m_xcells[level];
    
    int bl_count;
    int *bl = border_leafs(rank, in_local_boxes, &bl_count, false);

    int bxl_count;
    int *bxl = border_leafs(rank, in_local_boxes, &bxl_count, true);
    
    (*out_part_count) = 0;
    for(int i=0;i<bl_count;i++) {
	int idx = bl[i];
	(*out_part_count) += leafs[idx].num_children;
    }

    for(int i=0;i<bxl_count;i++) {
	int idx = bxl[i];
	(*out_part_count) += xleafs[idx].num_children;
    }
    
    m_logger->info("Would send %d particles to %d", (*out_part_count), rank);

    MPI_Isend(out_part_count, 1, MPI_INT, rank, count_tag, m_local_comm->m_comm, out_cnt_req);

    memory::create_1d(out_sendbuf, (*out_part_count));
    int n = 0;
    for(int i=0;i<bl_count;i++) {
	cell_t *leaf = &leafs[bl[i]];
	for(int j=0;j<leaf->num_children;j++) {
	    out_sendbuf[n].xyzq[0] = m_particles[leaf->first_child + j].xyzq[0];
	    out_sendbuf[n].xyzq[1] = m_particles[leaf->first_child + j].xyzq[1];
	    out_sendbuf[n].xyzq[2] = m_particles[leaf->first_child + j].xyzq[2];
	    out_sendbuf[n].xyzq[3] = m_particles[leaf->first_child + j].xyzq[3];
	    out_sendbuf[n++].cellID = m_particles[leaf->first_child + j].cellID;
	}
    }
    for(int i=0;i<bxl_count;i++) {
	cell_t *leaf = &xleafs[bxl[i]];
	for(int j=0;j<leaf->num_children;j++) {
	    out_sendbuf[n].xyzq[0] = m_particles[leaf->first_child + j].xyzq[0];
	    out_sendbuf[n].xyzq[1] = m_particles[leaf->first_child + j].xyzq[1];
	    out_sendbuf[n].xyzq[2] = m_particles[leaf->first_child + j].xyzq[2];
	    out_sendbuf[n].xyzq[3] = m_particles[leaf->first_child + j].xyzq[3];
	    out_sendbuf[n++].cellID = m_particles[leaf->first_child + j].cellID;
	}
    }
    MPI_Isend(out_sendbuf, (*out_part_count)*sizeof(xparticle_t), MPI_BYTE, rank, data_tag, m_local_comm->m_comm, out_data_req);

    memory::destroy_1d(bl);
    memory::destroy_1d(bxl);
}


//
// Incorporate the incoming in_xparticles into our alien particle list
// 
void fmm_tree::inhale_xparticles(xparticle_t *in_xparticles, int in_count)
{
    if(m_xparticles == NULL) {
	memory::create_1d(m_xparticles, in_count);
    }else {
	int new_size = m_nxparticles + in_count;
	m_xparticles = (xparticle_t *)memory::wrealloc(m_xparticles, new_size * sizeof(xparticle_t));
    }
    
    memcpy(m_xparticles + m_nxparticles, in_xparticles, in_count*sizeof(xparticle_t));
    m_nxparticles += in_count;
    
    sort_particles(m_xparticles, m_nxparticles, true);
}


//
// Incorporate the incoming in_xleafs into our alien cell list
// 
void fmm_tree::inhale_xleafs(cell_t *in_xleafs, int in_count)
{
    if(m_nxcells[max_level()] == 0) {
	memory::create_1d(m_xcells[max_level()], in_count);
    }else {
	int new_size = m_nxcells[max_level()] + in_count;
	m_xcells[max_level()] = (cell_t *)memory::wrealloc(m_xcells[max_level()], new_size * sizeof(cell_t));
    }
    
    memcpy(m_xcells[max_level()] + m_nxcells[max_level()], in_xleafs, in_count*sizeof(cell_t));
    m_nxcells[max_level()] += in_count;
    
    cell_t::sort(m_xcells[max_level()], m_nxcells[max_level()], true);
}

void fmm_tree::recv_particles_from_neighbour(int rank, int count_tag, int data_tag)
{
    int part_count;
    MPI_Recv(&part_count, 1, MPI_INT, rank, count_tag, m_local_comm->m_comm, MPI_STATUS_IGNORE);
    m_logger->info("Would receive %d particles from %d", part_count, rank);

    xparticle_t *recvbuf;
    memory::create_1d(recvbuf, part_count);
    MPI_Recv(recvbuf, part_count*sizeof(xparticle_t), MPI_BYTE, rank, data_tag, m_local_comm->m_comm, MPI_STATUS_IGNORE);

    // the array of xparticles has to be sorted in order to create xleafs out of them
    // however, the array IS already sorted, because in send_particles_to_neighbour it is
    // taken cell by cell, and THEY are sorted...
    //sort_particles(recvbuf, part_count, true);  // need to be sorted to create xleafs out of them
    
    int nxleafs;
    cell_t *xleafs = create_leafs(recvbuf, part_count, &nxleafs);  // create the new xleafs
    
    inhale_xparticles(recvbuf, part_count);  // incorporate xparticles into our tree
    inhale_xleafs(xleafs, nxleafs);  // incorporate xleafs into our tree
    memory::destroy_1d(recvbuf);
    memory::destroy_1d(xleafs);
}

// void fmm_tree::compute_leafs_to_send(box_t<iris_real> *in_local_boxes, int *&send_to_ranks, int *&send_cnt, int *&leafs_to_send, int *out_rank_count, int *out_hwm)
// {
//     int level = max_level();
//     cell_t *leafs = m_cells[level];

//     memory::create_1d(send_to_ranks, m_local_comm->m_size);
//     memory::create_1d(send_cnt, m_local_comm->m_size);
//     memory::create_1d(leafs_to_send, m_ncells[level] * m_local_comm->m_size);

//     int rank_count = 0;
//     int hwm = 0;
//     int prev_hwm = 0;
//     for(int rank = 0;rank < m_local_comm->m_size; rank++) {
//     	if(rank == m_local_comm->m_rank) {
//     	    continue;
//     	}
//     	for(int n = 0; n<m_ncells[level]; n++) {
//     	    bool send = false;
//     	    iris_real dn = leafs[n].radius;
//     	    iris_real cx = leafs[n].center[0];
//     	    iris_real cy = leafs[n].center[1];
//     	    iris_real cz = leafs[n].center[2];
//     	    for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
//     		for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
//     		    for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
//     			iris_real x = cx + ix * m_domain->m_global_box.xsize;
//     			iris_real y = cy + iy * m_domain->m_global_box.ysize;
//     			iris_real z = cz + iz * m_domain->m_global_box.zsize;
//     			iris_real rn = in_local_boxes[rank].distance_to(x, y, z);
//     			if (dn/rn < m_mac) {
//     			    continue;
//     			}
//     			// D(n)/r(n) >= θ - this means that this leaf is too close to the border
//     			// and is needed by the other processor to do P2P
//     			send = true;
//     			break;
//     		    }
//     		    if(send) {
//     			break;
//     		    }
//     		}
//     		if(send) {
//     		    break;
//     		}
//     	    }
//     	    if(send) {
// 		leafs_to_send[hwm++] = n;
//     	    }
//     	}
// 	send_to_ranks[rank_count] = rank;
// 	send_cnt[rank_count++] = hwm - prev_hwm;
// 	prev_hwm = hwm;
//     }

//     memory::wrealloc(send_to_ranks, rank_count*sizeof(int));
//     memory::wrealloc(send_cnt, rank_count*sizeof(int));
//     memory::wrealloc(leafs_to_send, hwm*sizeof(int));
//     *out_rank_count = rank_count;
//     *out_hwm = hwm;
// }

// void fmm_tree::compute_particle_send_size(int *send_to_ranks, int *send_cell_count, int *leafs_to_send, int rank_size,
// 					  pit_t *&send_buf, int *&send_body_count, int *&send_body_disp, int *&recv_body_count)
// {
//     int level = max_level();
//     cell_t *leafs = m_cells[level];

//     int n = 0;
//     int pit_count = 0;

//     memory::create_1d(recv_body_count, m_local_comm->m_size, true);
//     memory::create_1d(send_body_count, m_local_comm->m_size, true);
//     memory::create_1d(send_body_disp, m_local_comm->m_size, true);
    
//     for(int i=0;i<rank_size;i++) {
// 	int rank = send_to_ranks[i];
// 	for(int j=0;j<send_cell_count[i];j++) {
// 	    int idx = leafs_to_send[n];
// 	    pit_count += leafs[idx].num_children;
// 	    n++;
// 	}
//     }

//     memory::create_1d(send_buf, pit_count);

//     n = 0;
//     int m = 0;
//     int prevm = 0;
//     for(int i=0;i<rank_size;i++) {
// 	int rank = send_to_ranks[i];
// 	for(int j=0;j<send_cell_count[i];j++) {
// 	    int idx = leafs_to_send[n];
// 	    for(int k=0;k<leafs[idx].num_children;k++) {
// 		send_buf[m].xyzq[0] = m_particles[leafs[idx].first_child + k].xyzq[0];
// 		send_buf[m].xyzq[1] = m_particles[leafs[idx].first_child + k].xyzq[1];
// 		send_buf[m].xyzq[2] = m_particles[leafs[idx].first_child + k].xyzq[2];
// 		send_buf[m].xyzq[3] = m_particles[leafs[idx].first_child + k].xyzq[3];
// 		send_buf[m].cellID = m_particles[leafs[idx].first_child + k].cellID;
// 		m++;
// 	    }
// 	    n++;
// 	}
// 	send_body_count[rank] = m - prevm;
// 	send_body_disp[rank] = prevm;
// 	prevm = m;
//     }

//     MPI_Alltoall(send_body_count, 1, MPI_INT, recv_body_count, 1, MPI_INT, m_local_comm->m_comm);
// }

fmm_tree *fmm_tree::compute_LET(box_t<iris_real> *in_local_boxes)
{
    timer tm;
    tm.start();
    
    // fmm_tree *retval = new fmm_tree(this);

    // int *send_to_ranks;
    // int *send_cell_count;
    // int *leafs_to_send;
    // int rank_count;
    // int hwm;

    // pit_t *send_buf;
    // int *send_body_count;
    // int *send_body_disp;
    // int *recv_body_count;
    // compute_leafs_to_send(in_local_boxes, send_to_ranks, send_cell_count, leafs_to_send, &rank_count, &hwm);
    // compute_particle_send_size(send_to_ranks, send_cell_count, leafs_to_send, rank_count, send_buf, send_body_count, send_body_disp, recv_body_count);

    // memory::destroy_1d(send_to_ranks);
    // memory::destroy_1d(send_cell_count);
    // memory::destroy_1d(leafs_to_send);
    // memory::destroy_1d(send_buf);
    // memory::destroy_1d(send_body_count);
    // memory::destroy_1d(send_body_disp);
    // memory::destroy_1d(recv_body_count);
    
    for(int i=0;i<3;i++) {
    	MPI_Request cnt_req[2];
    	MPI_Request data_req[2];
	xparticle_t *sendbuf[2];
	int part_count[2];
    	for(int j=0;j<2;j++) {
    	    cnt_req[j] = MPI_REQUEST_NULL;
    	    data_req[j] = MPI_REQUEST_NULL;
	    sendbuf[0] = NULL;
	    sendbuf[1] = NULL;
    	}
    	for(int j=0;j<2;j++) {
    	    int rank = m_proc_grid->m_hood[i][j];
    	    if(rank < 0 || rank > m_local_comm->m_size ||      // no pbc and no neighbour in this dir
    	       rank == m_local_comm->m_rank ||                 // not same rank
    	       (j == 1 && rank == m_proc_grid->m_hood[i][0]))  // this rank was processed on the previous iteration (e.g. PBC, 2 in dir, 0 has 1 as both left and right neighbours)
    	    {
    		continue;
    	    }
    	    send_particles_to_neighbour(rank, in_local_boxes,
					IRIS_TAG_FMM_P2P_HALO_CNT+i*j, IRIS_TAG_FMM_P2P_HALO+i*j,
					part_count + j, sendbuf[j],
					cnt_req + j, data_req + j);
    	}
    	for(int j=0;j<2;j++) {
    	    int rank = m_proc_grid->m_hood[i][j];
    	    if(rank < 0 || rank > m_local_comm->m_size ||      // no pbc and no neighbour in this dir
    	       rank == m_local_comm->m_rank ||                 // not same rank
    	       (j == 1 && rank == m_proc_grid->m_hood[i][0]))  // this rank was processed on the previous iteration (e.g. PBC, 2 in dir, 0 has 1 as both left and right neighbours)
    	    {
    		continue;
    	    }
    	    recv_particles_from_neighbour(rank, IRIS_TAG_FMM_P2P_HALO_CNT+i*j, IRIS_TAG_FMM_P2P_HALO+i*j);
    	}
    	for(int j=0;j<2;j++) {
    	    MPI_Wait(cnt_req+j, MPI_STATUS_IGNORE);
    	    MPI_Wait(data_req+j, MPI_STATUS_IGNORE);
	    memory::destroy_1d(sendbuf[j]);
    	}
    }
    
    
    tm.stop();
    m_logger->info("FMM: Compute LET wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());

    return this;
}
