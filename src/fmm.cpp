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
#include <assert.h>
#include "fmm.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "fmm_kernels.h"
#include "fmm_pair.h"
#include "logger.h"
#include "domain.h"
#include "comm_rec.h"
#include "timer.h"
#include "openmp.h"
#include "proc_grid.h"
#include "tags.h"

using namespace ORG_NCSA_IRIS;

#define _LN8 2.0794415416798357  // natural logarithm of 8

#define MIN_DEPTH 2   // minimum value for depth (2)
#define MAX_DEPTH 16  // more than enough (e.g. 18 quadrillion particles)

fmm::fmm(iris *obj):
    solver(obj), m_order(0), m_depth(0), m_mac(0.0), m_mac_let_corr(0.0), m_nterms(0),
    m_leaf_size{0.0, 0.0, 0.0}, m_local_root_level(0), m_local_boxes(NULL),
    m_scratch(NULL), m_tree_size(0), m_cell_meta(NULL), m_M(NULL), m_L(NULL),
    m_cells(NULL), m_xcells(NULL), m_nparticles(0), m_particles(NULL),
    m_nxparticles(0), m_xparticles{NULL, NULL, NULL, NULL, NULL, NULL},
    m_dirty(true), m_border_leafs(NULL), m_border_parts{NULL, NULL},
    m_sendcnt(NULL), m_senddisp(NULL), m_recvcnt(NULL), m_recvdisp(NULL),
    m_p2m_count(0), m_m2m_count(0), m_m2l_count(0), m_p2p_count(0),
    m_l2l_count(0), m_l2p_count(0)
{
}

fmm::~fmm()
{
    memory::destroy_1d(m_local_boxes);
    memory::destroy_1d(m_scratch);
    memory::destroy_1d(m_cell_meta);
    memory::destroy_2d(m_M);
    memory::destroy_2d(m_L);
    memory::destroy_1d(m_cells);
    memory::destroy_1d(m_xcells);
    memory::destroy_1d(m_particles);
    memory::destroy_1d(m_xparticles[0]);
    memory::destroy_1d(m_xparticles[1]);
    memory::destroy_1d(m_xparticles[2]);
    memory::destroy_1d(m_xparticles[3]);
    memory::destroy_1d(m_xparticles[4]);
    memory::destroy_1d(m_xparticles[5]);
    memory::destroy_1d(m_border_leafs);
    memory::destroy_1d(m_border_parts[0]);
    memory::destroy_1d(m_border_parts[1]);
    memory::destroy_1d(m_sendcnt);
    memory::destroy_1d(m_senddisp);
    memory::destroy_1d(m_recvcnt);
    memory::destroy_1d(m_recvdisp);
}

void fmm::commit()
{
    if(m_dirty) {
	m_order = m_iris->m_order;  // if p = 2, we expand multipoles until Y_2^2

	int natoms = m_iris->m_natoms;  // atoms won't change during simulation (hopefully)
	solver_param_t t = m_iris->get_solver_param(IRIS_SOLVER_FMM_NCRIT);
	int ncrit = t.i;
    
	m_depth = (natoms > ncrit) ? int(log(natoms / ncrit)/_LN8) + 1 : 0;
	m_depth = MAX(m_depth, MIN_DEPTH);
	m_depth = MIN(m_depth, MAX_DEPTH);

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC);
	m_mac = t.r;

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC_CORR);
	m_mac_let_corr = t.r;
	
	m_nterms = (m_order + 1) * (m_order + 2) / 2;

	m_local_root_level = int(log(m_local_comm->m_size-1) / _LN8) + 1;
	if(m_local_comm->m_size == 1) {
	    m_local_root_level = 0;
	}

	m_tree_size = ((1 << 3 * m_depth) - 1) / 7;
	
	handle_box_resize();

	memory::destroy_1d(m_scratch);
	memory::create_1d(m_scratch, 2*m_nterms);
	
	memory::destroy_2d(m_M);
	memory::create_2d(m_M, m_tree_size, 2*m_nterms, true);
	
	memory::destroy_2d(m_L);
	memory::create_2d(m_L, m_tree_size, 2*m_nterms, true);
	
	memory::destroy_1d(m_cells);
	memory::destroy_1d(m_xcells);
	
	memory::create_1d(m_cells, m_tree_size);
	memset(m_cells, 0, m_tree_size*sizeof(cell_t));  // make sure num_children are 0

	memory::create_1d(m_xcells, m_tree_size);
	
	// particles/xparticles will be re-created in solve, since
	// they depend on the number of local particles, which might not be
	// known at commit time...

	memory::destroy_1d(m_border_leafs);
	memory::create_1d(m_border_leafs, ((1 << (3*max_level())) / m_local_comm->m_size) + 1);

	memory::destroy_1d(m_sendcnt);
	memory::create_1d(m_sendcnt, m_local_comm->m_size);
	
	memory::destroy_1d(m_senddisp);
	memory::create_1d(m_senddisp, m_local_comm->m_size);
	
	memory::destroy_1d(m_recvcnt);
	memory::create_1d(m_recvcnt, m_local_comm->m_size);
	
	memory::destroy_1d(m_recvdisp);
	memory::create_1d(m_recvdisp, m_local_comm->m_size);
	
	m_dirty = false;
	m_logger->info("FMM: order = %d; depth = %d; tree size = %d; local root level = %d", m_order, m_depth, m_tree_size, m_local_root_level);
    }
}

void fmm::generate_cell_meta()
{
    memory::destroy_1d(m_cell_meta);
    memory::create_1d(m_cell_meta, m_tree_size);
    for(int i=0;i<m_tree_size;i++) {
	m_cell_meta[i].set(m_cell_meta, i, &m_domain->m_global_box, m_leaf_size, max_level());
    }
}

void fmm::set_leaf_size()
{
    int nd = 1 << max_level();

    m_leaf_size[0] = m_domain->m_global_box.xsize / nd;
    m_leaf_size[1] = m_domain->m_global_box.ysize / nd;
    m_leaf_size[2] = m_domain->m_global_box.zsize / nd;
}

void fmm::get_local_boxes()
{
    memory::destroy_1d(m_local_boxes);
    memory::create_1d(m_local_boxes, m_local_comm->m_size);
    
    MPI_Allgather(&m_domain->m_local_box, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_local_boxes, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_iris->m_local_comm->m_comm);
}

void fmm::handle_box_resize()
{
    set_leaf_size();
    get_local_boxes();
    generate_cell_meta();
}

void fmm::solve()
{
    m_p2m_count = m_m2m_count = m_m2l_count = m_p2p_count = m_l2l_count = m_l2p_count = 0;
    
    upward_pass_in_local_tree();
    exchange_LET();
    dual_tree_traversal();

    m_logger->info("P2M: %d, M2M: %d, M2L: %d, P2P: ~%d^2, L2L: %d, L2P: %d", m_p2m_count, m_m2m_count, m_m2l_count, (int)sqrt(m_p2p_count*1.0), m_l2l_count, m_l2p_count);

    sort_back_particles(m_particles, m_nparticles);

    iris_real ener = 0.0;
    iris_real fx_sum = 0.0;
    iris_real fy_sum = 0.0;
    iris_real fz_sum = 0.0;
    for(int i=0;i<m_nparticles;i++) {
	iris_real fx = m_particles[i].tgt[1] * m_particles[i].xyzq[3];
	iris_real fy = m_particles[i].tgt[2] * m_particles[i].xyzq[3];
	iris_real fz = m_particles[i].tgt[3] * m_particles[i].xyzq[3];

	fx_sum += fx;
	fy_sum += fy;
	fz_sum += fz;
	
	m_logger->info("F[%d] = (%f, %f, %f)", m_particles[i].index, fx, fy, fz);
	ener += m_particles[i].tgt[0] * m_particles[i].xyzq[3];
    }
    m_logger->info("Ftot = (%f, %f, %f)", fx_sum, fy_sum, fz_sum);
    m_logger->info("FMM Local Energy: %f", ener / 2);
    
    MPI_Barrier(m_iris->server_comm());
    exit(-1);
}

void fmm::upward_pass_in_local_tree()
{
    timer tm;
    tm.start();
    
    load_particles();                                          // creates and sorts the m_particles array
    distribute_particles(m_particles, m_nparticles, IRIS_FMM_CELL_LOCAL, m_cells);  // distribute particles into leaf cells
    link_parents(m_cells);
    eval_p2m(m_cells, false);                                  // eval P2M for leaf nodes
    eval_m2m(m_cells, false);                                  // eval M2M for non-leaf nodes
    
    tm.stop();
    m_logger->info("FMM: Local tree construction wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
    //print_tree("Cell", m_cells, 0);
}

void fmm::load_particles()
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int nd = 1 << max_level();

    m_nparticles = m_iris->num_local_atoms();
    m_logger->info("FMM: This rank owns %d particles", m_nparticles);
    memory::destroy_1d(m_particles);
    memory::create_1d(m_particles, m_nparticles);

    box_t<iris_real> *gbox = &m_domain->m_global_box;
    
    int n = 0;
    int lc[3];  // leaf global coords
    for(auto it = m_iris->m_ncharges.begin(); it != m_iris->m_ncharges.end(); it++) {
	int ncharges = it->second;
	iris_real *charges = m_iris->m_charges[it->first];
	for(int i=0;i<ncharges;i++) {
	    iris_real tx = (charges[i * 5 + 0] - gbox->xlo) / m_leaf_size[0];
	    iris_real ty = (charges[i * 5 + 1] - gbox->ylo) / m_leaf_size[1];
	    iris_real tz = (charges[i * 5 + 2] - gbox->zlo) / m_leaf_size[2];

	    lc[0] = (int) tx;
	    lc[1] = (int) ty;
	    lc[2] = (int) tz;

	    int id = 0;
	    for(int l=0;l<max_level(); l++) {
		for(int d=0;d<3;d++) {
		    id += (lc[d] & 1) << (3*l + d);
		    lc[d] >>= 1;
		}
	    }

	    int cellID = offset + id;
	    m_particles[n].rank = it->first;
	    m_particles[n].index = i;
	    m_particles[n].cellID = cellID;
	    memcpy(m_particles[n].xyzq, charges+i*5, 4*sizeof(iris_real));
	    memset(m_particles[n].tgt, 0, 4*sizeof(iris_real));
	    n++;
	}
    }

    // sort the final list by cellID
    sort_particles(m_particles, m_nparticles, false);
}

void fmm::relink_parents(cell_t *io_cells)
{
    // first, clear the num_children of all non-leaf cells
    int end = cell_meta_t::offset_for_level(max_level());
    for(int i=0;i<end;i++) {
	io_cells[i].flags & ~IRIS_FMM_CELL_HAS_CHILDREN;
    }

    // now, link all leafs upwards
    link_parents(io_cells);
}

void fmm::link_parents(cell_t *io_cells)
{
    for(int i=max_level();i>0;i--) {
	int start = cell_meta_t::offset_for_level(i);
	int end = cell_meta_t::offset_for_level(i+1);
	for(int j=start;j<end;j++) {
	    if((io_cells[j].num_children != 0) ||                   // cell is a non-empty leaf
	       (io_cells[j].flags & IRIS_FMM_CELL_HAS_CHILDREN) ||  // or cell is a non-leaf and has some children
	       (io_cells[j].flags & IRIS_FMM_CELL_ALIEN0)) {        // or is an alien cell
		int parent = cell_meta_t::parent_of(j);
		io_cells[parent].flags |= (IRIS_FMM_CELL_HAS_CHILD1 << ((j - start) % 8));
	    }
	}
    }
}

void fmm::eval_p2m(cell_t *in_cells, bool alien_only)
{
    int offset = cell_meta_t::offset_for_level(max_level());
    for(int i=offset;i<m_tree_size;i++) {
	cell_t *leaf = &in_cells[i];
	if(leaf->num_children == 0) {
	    continue;
	}
	if(alien_only && !(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF)) {
	    continue;
	}
	for(int j=0;j<leaf->num_children;j++) {
	    xparticle_t *ptr;
	    iris_real x, y, z, q;
	    if(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		if(leaf->flags & IRIS_FMM_CELL_ALIEN1) {
		    ptr = m_xparticles[0];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN2) {
		    ptr = m_xparticles[1];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN3) {
		    ptr = m_xparticles[2];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN4) {
		    ptr = m_xparticles[3];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN5) {
		    ptr = m_xparticles[4];
		}else if(leaf->flags & IRIS_FMM_CELL_ALIEN6) {
		    ptr = m_xparticles[5];
		}
		x = ptr[leaf->first_child+j].xyzq[0] - m_cell_meta[i].center[0];
		y = ptr[leaf->first_child+j].xyzq[1] - m_cell_meta[i].center[1];
		z = ptr[leaf->first_child+j].xyzq[2] - m_cell_meta[i].center[2];
		q = ptr[leaf->first_child+j].xyzq[3];
	    }else {
		x = m_particles[leaf->first_child+j].xyzq[0] - m_cell_meta[i].center[0];
		y = m_particles[leaf->first_child+j].xyzq[1] - m_cell_meta[i].center[1];
		z = m_particles[leaf->first_child+j].xyzq[2] - m_cell_meta[i].center[2];
		q = m_particles[leaf->first_child+j].xyzq[3];
	    }
	    p2m(m_order, x, y, z, q, m_M[i]);
	    in_cells[i].flags |= IRIS_FMM_CELL_VALID_M;
	    m_p2m_count++;
	}
    }
}

void fmm::eval_m2m(cell_t *in_cells, bool invalid_only)
{
    int last_level = invalid_only ? 0 : m_local_root_level;
    for(int level = max_level()-1;level>=last_level;level--) {
	int tcellID = cell_meta_t::offset_for_level(level);
	int scellID = cell_meta_t::offset_for_level(level+1);
	int ntcells = scellID - tcellID;
	for(int i = 0;i<ntcells;i++) {
	    if(invalid_only && (in_cells[tcellID].flags & IRIS_FMM_CELL_VALID_M)) {
		tcellID++;
		scellID+=8;
		continue;
	    }
	    
	    iris_real cx = m_cell_meta[tcellID].center[0];
	    iris_real cy = m_cell_meta[tcellID].center[1];
	    iris_real cz = m_cell_meta[tcellID].center[2];

	    bool valid_m = false;
	    for(int j=0;j<8;j++) {
		int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
		if(!(in_cells[tcellID].flags & mask)) {
		    scellID++;
		    continue;
		}
		iris_real x = m_cell_meta[scellID].center[0] - cx;
		iris_real y = m_cell_meta[scellID].center[1] - cy;
		iris_real z = m_cell_meta[scellID].center[2] - cz;
		memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
		m2m(m_order, x, y, z, m_M[scellID], m_M[tcellID], m_scratch);
		valid_m = true;
		scellID++;
		m_m2m_count++;
	    }
	    if(valid_m) {
		in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_M;
	    }
	    tcellID++;
	}
    }
}

void fmm::exchange_LET()
{
    timer tm;
    tm.start();
    
    memcpy(m_xcells, m_cells, m_tree_size * sizeof(cell_t));  // copy local tree to LET
    if(m_local_comm->m_size > 1) {
	exchange_p2p_halo();
	exchange_rest_of_LET();
	recalculate_LET();
    }
    //print_tree("Xcell", m_xcells, 0);
    
    tm.stop();
    m_logger->info("FMM: Exchange LET wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::recalculate_LET()
{
    relink_parents(m_xcells);
    eval_p2m(m_xcells, true);
    eval_m2m(m_xcells, true);
}

void fmm::print_tree(const char *label, cell_t *in_cells, int cellID)
{
    int level = cell_meta_t::level_of(cellID);
    if(level == max_level()) {
	m_logger->info("%*s%s %d (L%d) has %d particles starting from %d and flags 0x%x; M[0] = %f", level+1, " ", label, cellID, level, in_cells[cellID].num_children, in_cells[cellID].first_child, in_cells[cellID].flags, m_M[cellID][0]);
    }else {
	int num_children = 0;
	int mask = IRIS_FMM_CELL_HAS_CHILD1;
	for(int i=0;i<8;i++) {
	    if(in_cells[cellID].flags & mask) {
		num_children++;
	    }
	    mask <<= 1;
	}
	
	m_logger->info("%*s%s %d (L%d) has %d children and flags 0x%x; M[0] = %f", level+1, " ", label, cellID, level, num_children, in_cells[cellID].flags, m_M[cellID][0]);
    }
    if(level < max_level()) {
	int this_offset = cell_meta_t::offset_for_level(level);
	int children_offset = cell_meta_t::offset_for_level(level+1);
	for(int j=0;j<8;j++) {
	    int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
	    if(in_cells[cellID].flags & mask) {
		int childID = children_offset + 8*(cellID-this_offset) + j;
		print_tree(label, in_cells, childID);
	    }
	}
    }
}

void fmm::dual_tree_traversal()
{
    timer tm;
    tm.start();

    assert(m_queue.empty());
    
    for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
	    for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		pair_t root;
		root.sourceID = 0;
		root.targetID = 0;
		m_queue.push_back(root);
		traverse_queue(ix, iy, iz);
	    }
	}
    }

    eval_l2l(m_cells);
    eval_l2p(m_cells);
    
    tm.stop();
    m_logger->info("FMM: Dual Tree Traversal wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::traverse_queue(int ix, int iy, int iz)
{
    while(!m_queue.empty()) {
	pair_t pair = m_queue.front();
	m_queue.pop_front();
	if(m_cell_meta[pair.sourceID].radius > m_cell_meta[pair.targetID].radius) {
	    cell_t *src = m_xcells + pair.sourceID;
	    int level = cell_meta_t::level_of(pair.sourceID);
	    int this_offset = cell_meta_t::offset_for_level(level);
	    int children_offset = cell_meta_t::offset_for_level(level+1);
	    int mask = IRIS_FMM_CELL_HAS_CHILD1;
	    for(int i=0;i<8;i++) {
		if(m_xcells[pair.sourceID].flags & mask) {
		    int childID = children_offset + (pair.sourceID - this_offset)*8 + i;
		    interact(childID, pair.targetID, ix, iy, iz);
		}
		mask <<= 1;
	    }
	}else {
	    cell_t *target = m_cells + pair.targetID;
	    int level = cell_meta_t::level_of(pair.targetID);
	    int this_offset = cell_meta_t::offset_for_level(level);
	    int children_offset = cell_meta_t::offset_for_level(level+1);
	    int mask = IRIS_FMM_CELL_HAS_CHILD1;
	    for(int i=0;i<8;i++) {
		if(m_cells[pair.targetID].flags & mask) {
		    int childID = children_offset + (pair.targetID - this_offset)*8 + i;
		    interact(pair.sourceID, childID, ix, iy, iz);
		}
		mask <<= 1;
	    }
	}
    }
}

void fmm::interact(int srcID, int destID, int ix, int iy, int iz)
{
    iris_real src_cx = m_cell_meta[srcID].center[0] + ix * m_domain->m_global_box.xsize;
    iris_real src_cy = m_cell_meta[srcID].center[1] + iy * m_domain->m_global_box.ysize;
    iris_real src_cz = m_cell_meta[srcID].center[2] + iz * m_domain->m_global_box.zsize;

    iris_real dest_cx = m_cell_meta[destID].center[0];
    iris_real dest_cy = m_cell_meta[destID].center[1];
    iris_real dest_cz = m_cell_meta[destID].center[2];

    iris_real dx = dest_cx - src_cx;
    iris_real dy = dest_cy - src_cy;
    iris_real dz = dest_cz - src_cz;
    
    iris_real rn = sqrt(dx*dx + dy*dy + dz*dz);
    iris_real dn = m_cell_meta[srcID].radius + m_cell_meta[destID].radius;
    if(dn/rn < m_mac) {
	eval_m2l(srcID, destID, ix, iy, iz);
    }else if(cell_meta_t::level_of(srcID) == max_level() &&
	     cell_meta_t::level_of(destID) == max_level())
    {
	eval_p2p(srcID, destID, ix, iy, iz);
    }else {
	pair_t pair;
	pair.sourceID = srcID;
	pair.targetID = destID;
	m_queue.push_back(pair);
    }
}

void fmm::eval_p2p(int srcID, int destID, int ix, int iy, int iz)
{
    for(int i=0;i<m_cells[destID].num_children;i++) {
	iris_real tx = m_particles[m_cells[destID].first_child + i].xyzq[0];
	iris_real ty = m_particles[m_cells[destID].first_child + i].xyzq[1];
	iris_real tz = m_particles[m_cells[destID].first_child + i].xyzq[2];

	iris_real sum_phi = 0.0;
	iris_real sum_ex = 0.0;
	iris_real sum_ey = 0.0;
	iris_real sum_ez = 0.0;
	for(int j=0;j<m_xcells[srcID].num_children;j++) {
	    xparticle_t *ptr;
	    iris_real sx, sy, sz, sq;
	    if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN1) {
		    ptr = m_xparticles[0];
		}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN2) {
		    ptr = m_xparticles[1];
		}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN3) {
		    ptr = m_xparticles[2];
		}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN4) {
		    ptr = m_xparticles[3];
		}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN5) {
		    ptr = m_xparticles[4];
		}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN6) {
		    ptr = m_xparticles[5];
		}
		sx = ptr[m_xcells[srcID].first_child + j].xyzq[0] + ix * m_domain->m_global_box.xsize;
		sy = ptr[m_xcells[srcID].first_child + j].xyzq[1] + iy * m_domain->m_global_box.ysize;
		sz = ptr[m_xcells[srcID].first_child + j].xyzq[2] + iz * m_domain->m_global_box.zsize;
		sq = ptr[m_xcells[srcID].first_child + j].xyzq[3];
	    }else {
		sx = m_particles[m_xcells[srcID].first_child + j].xyzq[0] + ix * m_domain->m_global_box.xsize;
		sy = m_particles[m_xcells[srcID].first_child + j].xyzq[1] + iy * m_domain->m_global_box.ysize;
		sz = m_particles[m_xcells[srcID].first_child + j].xyzq[2] + iz * m_domain->m_global_box.zsize;
		sq = m_particles[m_xcells[srcID].first_child + j].xyzq[3];
	    }

	    iris_real dx = tx - sx;
	    iris_real dy = ty - sy;
	    iris_real dz = tz - sz;
	    iris_real r2 = dx*dx + dy*dy + dz*dz;
	    iris_real inv_r2;
	    if(r2 == 0) {
		inv_r2 = 0;
	    }else {
		inv_r2 = 1/r2;
	    }
	    iris_real phi = sq * sqrt(inv_r2);
	    iris_real phi_over_r2 = phi * inv_r2;
	    iris_real ex = dx * phi_over_r2;
	    iris_real ey = dy * phi_over_r2;
	    iris_real ez = dz * phi_over_r2;

	    sum_phi += phi;
	    sum_ex += ex;
	    sum_ey += ey;
	    sum_ez += ez;
	    
	    m_p2p_count++;
	}
	m_particles[m_cells[destID].first_child + i].tgt[0] += sum_phi;
	m_particles[m_cells[destID].first_child + i].tgt[1] += sum_ex;
	m_particles[m_cells[destID].first_child + i].tgt[2] += sum_ey;
	m_particles[m_cells[destID].first_child + i].tgt[3] += sum_ez;
    }
}

void fmm::eval_m2l(int srcID, int destID, int ix, int iy, int iz)
{
    assert((m_xcells[srcID].flags & IRIS_FMM_CELL_VALID_M));

    iris_real sx = m_cell_meta[srcID].center[0] + ix * m_domain->m_global_box.xsize;
    iris_real sy = m_cell_meta[srcID].center[1] + iy * m_domain->m_global_box.ysize;
    iris_real sz = m_cell_meta[srcID].center[2] + iz * m_domain->m_global_box.zsize;

    iris_real tx = m_cell_meta[destID].center[0];
    iris_real ty = m_cell_meta[destID].center[1];
    iris_real tz = m_cell_meta[destID].center[2];

    iris_real x = tx - sx;
    iris_real y = ty - sy;
    iris_real z = tz - sz;

    memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
    m2l(m_order, x, y, z, m_M[srcID], m_L[destID], m_scratch);
    m_cells[destID].flags |= IRIS_FMM_CELL_VALID_L;
    m_m2l_count++;
}

void fmm::eval_l2l(cell_t *in_cells)
{
    for(int level = 1; level < m_depth; level++) {
	int scellID = cell_meta_t::offset_for_level(level);
	int tcellID = cell_meta_t::offset_for_level(level+1);
	int nscells = tcellID - scellID;
	for(int i = 0;i<nscells;i++) {
	    if(!(m_cells[scellID].flags & IRIS_FMM_CELL_VALID_L)) {
		scellID++;
		tcellID += 8;
		continue;
	    }
	    
	    iris_real cx = m_cell_meta[scellID].center[0];
	    iris_real cy = m_cell_meta[scellID].center[1];
	    iris_real cz = m_cell_meta[scellID].center[2];

	    bool valid_l = false;
	    for(int j=0;j<8;j++) {
		int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
		if(!(in_cells[scellID].flags & mask)) {
		    tcellID++;
		    continue;
		}
		iris_real x = cx - m_cell_meta[tcellID].center[0];
		iris_real y = cy - m_cell_meta[tcellID].center[1];
		iris_real z = cz - m_cell_meta[tcellID].center[2];

		memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
		l2l(m_order, x, y, z, m_L[scellID], m_L[tcellID], m_scratch);
		valid_l = true;
		tcellID++;
		m_l2l_count++;
	    }
	    if(valid_l) {
		in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_L;
	    }
	    scellID++;
	}
    }
}

void fmm::eval_l2p(cell_t *in_cells)
{
    int offset = cell_meta_t::offset_for_level(max_level());
    for(int i=offset;i<m_tree_size;i++) {
    	cell_t *leaf = &in_cells[i];
    	if(leaf->num_children == 0 || !(leaf->flags & IRIS_FMM_CELL_VALID_L)) {
    	    continue;
    	}
    	for(int j=0;j<leaf->num_children;j++) {
	    iris_real x = m_cell_meta[i].center[0] - m_particles[leaf->first_child+j].xyzq[0];
	    iris_real y = m_cell_meta[i].center[1] - m_particles[leaf->first_child+j].xyzq[1];
	    iris_real z = m_cell_meta[i].center[2] - m_particles[leaf->first_child+j].xyzq[2];
	    iris_real q = m_particles[leaf->first_child+j].xyzq[3];
	    iris_real phi, Ex, Ey, Ez;
	    
	    memset(m_scratch, 0, 2*m_nterms*sizeof(iris_real));
	    l2p(m_order, x, y, z, q, m_L[i], m_scratch, &phi, &Ex, &Ey, &Ez);
	    
	    m_particles[leaf->first_child+j].tgt[0] += phi;
	    m_particles[leaf->first_child+j].tgt[1] += Ex;
	    m_particles[leaf->first_child+j].tgt[2] += Ey;
	    m_particles[leaf->first_child+j].tgt[3] += Ez;
	    
	    m_l2p_count++;
    	}
    }
}
