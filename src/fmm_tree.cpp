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

using namespace ORG_NCSA_IRIS;

#define _LN8 2.0794415416798357  // natural logarithm of 8

#define MIN_DEPTH 2   // minimum value for depth
#define MAX_DEPTH 16  // more than enough (e.g. 18 quadrillion particles)

fmm_tree::fmm_tree(iris *in_iris):
    state_accessor(in_iris), m_nparticles(0), m_particles(NULL),
    m_leaf_size{0.0, 0.0, 0.0}
{    
    m_nterms = (m_iris->m_order + 1) * (m_iris->m_order + 2) / 2;
    determine_depth();
    memory::create_1d(m_ncells, m_depth);
    memory::create_1d(m_cells, m_depth);
    // m_particles will be created in due time
}

fmm_tree::~fmm_tree()
{
    free_cells();
    memory::destroy_1d(m_ncells);
    memory::destroy_1d(m_particles);
}

void fmm_tree::free_cells()
{
    if(m_cells != NULL) {
	for(int i=0;i<m_depth;i++) {
	    memory::destroy_1d(m_cells[i]);
	}
	memory::destroy_1d(m_cells);
    }
}


//
// Leaf size is determined by dividing the length of the global box in each dimension
// by the number of leaf cells per dimension.
//
// In addition, this function calculates the global coordinates of the center of each
// leaf.
//
// TODO: this is overdoing it -- it calculates the centers of ALL the leafs, not only
//       those stored on this processor; might need to optimize it by taking the
//       extents of the local box in mind. But on the other hand, it MIGHT turn out
//       that we need them anyway, so let's see...
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
    if(m_depth < MIN_DEPTH) {
	m_depth = MIN_DEPTH;
    }
    if(m_depth > MAX_DEPTH) {
	m_depth = MAX_DEPTH;
    }
}



//
// Transform the list of charges on this processor to a list of 'particles'.
//
void fmm_tree::charges2particles()
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
	    n++;
	}
    }
    // sort the final list by cellID desending
    particle_t::sort(m_particles, m_nparticles, true);
}


//
// Based on the sorted list of particles, create the leaf structure of the tree
// A leaf cell has an ID, reference to the first particle in the sorted array,
// number of particles and a set of multipole and local expansions
//
void fmm_tree::particles2leafs()
{
#define FROB						\
    leafs[n].cellID = last;				\
    leafs[n].first_child = first_child; 		\
    leafs[n].num_children = num_children; 		\
    leafs[n].set_center(gbox, m_leaf_size);		\
    memory::create_1d(leafs[n].m, 2*m_nterms, true);	\
    memory::create_1d(leafs[n].l, 2*m_nterms, true)

    box_t<iris_real> *gbox = &m_domain->m_global_box;
    
    // theoretical maximum number of leafs here: number of leafs everywhere
    // divided by the # of server ranks (every rank has equal volume)
    // will be realloced at the end of this function
    int nleafs = int((1 << (3*(m_depth-1))) / m_iris->m_local_comm->m_size) + 1;

    cell_t *leafs;
    memory::create_1d(leafs, nleafs);

    int last = m_particles[0].cellID;
    int first_child = 0;
    int num_children = 0;
    int n = 0;
    for(int i=0;i<m_nparticles;i++) {
	if(m_particles[i].cellID != last) {
	    FROB;
	    first_child = i;
	    num_children = 0;
	    last = m_particles[i].cellID;
	    n++;
	}
	num_children++;
    }
    FROB;
    nleafs = ++n;
    memory::wrealloc(leafs, nleafs * sizeof(cell_t));
    m_cells[max_level()] = leafs;
    m_ncells[max_level()] = nleafs;
    
#undef FROB
}

// TODO: this is *almost* the same as particles2leafs.
// Figure out a way to avoid code duplication...
void fmm_tree::children2parent(int in_level)
{
#define FROB						\
    parents[n].cellID = last;				\
    parents[n].first_child = first_child;		\
    parents[n].num_children = num_children;		\
    parents[n].set_center(gbox, m_leaf_size);		\
    memory::create_1d(parents[n].m, 2*m_nterms, true);	\
    memory::create_1d(parents[n].l, 2*m_nterms, true)

    const box_t<iris_real> *gbox = &(m_domain->m_global_box);
    
    int nchildren = m_ncells[in_level];
    cell_t *children = m_cells[in_level];

    // theoretical maximum number of cells here: number of cells everywhere
    // divided by the # of server ranks (every rank has equal volume)
    // will be realloced at the end of this function
    int nparents = int((1 << (3*(in_level-1))) / m_iris->m_local_comm->m_size) + 1;
    
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
    memory::wrealloc(parents, nparents * sizeof(cell_t));
    m_cells[in_level-1] = parents;
    m_ncells[in_level-1] = nparents;
#undef FROB
}

//
// Perform P2M
//
void fmm_tree::eval_p2m()
{
    int nleafs = m_ncells[max_level()];
    cell_t *leafs = m_cells[max_level()];
    
    for(int i=0;i<nleafs;i++) {
	for(int j=0;j<leafs[i].num_children;j++) {
	    int rank = m_particles[leafs[i].first_child+j].rank;
	    int index = m_particles[leafs[i].first_child+j].index;
	    iris_real x = m_iris->m_charges[rank][index * 5 + 0] - leafs[i].center[0];
	    iris_real y = m_iris->m_charges[rank][index * 5 + 1] - leafs[i].center[1];
	    iris_real z = m_iris->m_charges[rank][index * 5 + 2] - leafs[i].center[2];
	    iris_real q = m_iris->m_charges[rank][index * 5 + 3];
	    p2m(m_iris->m_order, x, y, z, q, leafs[i].m);
	}
	//print_multipoles(leafs[i].cellID, leafs[i].m);
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
