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
#include <stdlib.h>
#include <complex>
#include "fmm.h"
#include "logger.h"
#include "math.h"
#include "real.h"
#include "domain.h"
#include "comm_rec.h"

#define _LN8 2.0794415416798357

#define MIN_MAX_LEVEL 2   // minimum value for m_max_level
#define MAX_MAX_LEVEL 16  // more than enough (e.g. 18 quadrillion particles)

#define MULTIPOLE_INDEX(l, m) (l * (l + 1) + (2*m))

using namespace ORG_NCSA_IRIS;

fmm::fmm(class iris *obj):
    solver(obj), m_natoms(0), m_max_level(0), m_ncrit(0), m_order(0)
{
}

fmm::~fmm()
{
}

void fmm::commit()
{
    if(m_dirty) {
	// assume these will not change; otherwise move them out of if(m_dirty)
	m_order = m_iris->m_order;
	m_nterms = (m_order + 1) * (m_order + 2) / 2;
	m_natoms = m_iris->m_natoms;
	solver_param_t t = m_iris->get_solver_param(IRIS_SOLVER_FMM_NCRIT);
	m_ncrit = t.i;
	set_max_level();
	m_dirty = false;
    }
}

// The maximum tree level is determined so as each leaf cell has around ncrit particles
// This means log_8 (natoms/ncrit)
void fmm::set_max_level()
{
    m_max_level = (m_natoms > m_ncrit) ? int(log(m_natoms / m_ncrit)/_LN8) + 1 : 0;
    if(m_max_level < MIN_MAX_LEVEL) {
	m_max_level = MIN_MAX_LEVEL;
    }
    if(m_max_level > MAX_MAX_LEVEL) {
	m_max_level = MAX_MAX_LEVEL;
    }

    m_num_leafs_1D = 1 << m_max_level;
    m_first_leafID = ((1 << 3 * m_max_level) - 1) / 7;

    set_leaf_size();
}

void fmm::set_leaf_size()
{
    m_leaf_size[0] = m_domain->m_global_box.xsize / m_num_leafs_1D;
    m_leaf_size[1] = m_domain->m_global_box.ysize / m_num_leafs_1D;
    m_leaf_size[2] = m_domain->m_global_box.zsize / m_num_leafs_1D;
}

void fmm::handle_box_resize()
{
    set_leaf_size();
}

particle_t *fmm::charges2particles()
{
    m_leaf_centers.clear();
    
    const box_t<iris_real> *gbox = &(m_domain->m_global_box);
    
    particle_t *particles = new particle_t[m_natoms_local];

    size_t n = 0;
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
	    size_t leaf_index = iz + m_num_leafs_1D * (iy + m_num_leafs_1D * ix);  
	    size_t cellID = m_first_leafID + leaf_index;

	    m_leaf_centers[cellID][0] = (ix + 0.5) * m_leaf_size[0];
	    m_leaf_centers[cellID][1] = (iy + 0.5) * m_leaf_size[1];
	    m_leaf_centers[cellID][2] = (iz + 0.5) * m_leaf_size[2];
	    
	    particles[n].rank = it->first;
	    particles[n].index = i;
	    particles[n].cellID = cellID;
	    n++;
	}
    }
    return particles;
}

int sort_by_cellID_descending(const void *aptr, const void *bptr)
{
    particle_t *a = (particle_t *)aptr;
    particle_t *b = (particle_t *)bptr;
    
    if(a->cellID < b->cellID) {
	return 1;
    }else if(a->cellID > b->cellID) {
	return -1;
    }else {
	return 0;
    }
}

void fmm::set_natoms_local()
{
    m_natoms_local = 0;
    for(auto it = m_iris->m_ncharges.begin(); it != m_iris->m_ncharges.end(); it++) {
	m_natoms_local += it->second;
    }
}

leaf_t *fmm::particles2leafs(particle_t *particles, size_t *out_nleafs)
{

#define FROB						\
    leafs[n].cellID = last_cellID; 			\
    leafs[n].first_particle = first_particle; 		\
    leafs[n].num_particles = num_particles; 		\
    memory::create_1d(leafs[n].m, 2*m_nterms, true);	\
    memory::create_1d(leafs[n].l, 2*m_nterms, true)
    
    // theoretical maximum number of leafs here: number of leafs everywhere divided by
    // the # of server ranks (every rank has equal volume)
    // will be realloced at the end of this function
    *out_nleafs = (1 << (3*m_max_level)) / m_iris->m_local_comm->m_size;

    leaf_t *leafs;
    memory::create_1d(leafs, *out_nleafs);

    size_t last_cellID = particles[0].cellID;
    int num_particles = 0;
    int first_particle = 0;
    int n = 0;
    for(int i=0;i<m_natoms_local;i++) {
	if(particles[i].cellID != last_cellID) {
	    FROB;
	    num_particles = 0;
	    first_particle = i;
	    last_cellID = particles[i].cellID;
	    n++;
	}
	num_particles++;
    }
    FROB;
    *out_nleafs = ++n;
    return (leaf_t *)memory::wrealloc(leafs, (*out_nleafs) * sizeof(leaf_t));
#undef FROB
}


void fmm::eval_p2m(leaf_t *leafs, particle_t *particles, size_t nleafs)
{
    for(int i=0;i<nleafs;i++) {
	for(int j=0;j<leafs[i].num_particles;j++) {
	    int rank = particles[leafs[i].first_particle+j].rank;
	    int index = particles[leafs[i].first_particle+j].index;
	    iris_real x = m_iris->m_charges[rank][index * 5 + 0] - m_leaf_centers[leafs[i].cellID][0];
	    iris_real y = m_iris->m_charges[rank][index * 5 + 1] - m_leaf_centers[leafs[i].cellID][1];
	    iris_real z = m_iris->m_charges[rank][index * 5 + 2] - m_leaf_centers[leafs[i].cellID][2];
	    iris_real q = m_iris->m_charges[rank][index * 5 + 3];
	    p2m(x, y, z, q, leafs[i].m);
	}
	print_multipoles(leafs[i].cellID, leafs[i].m);
    }
}

void fmm::print_multipoles(size_t cellID, iris_real *gamma)
{
    for(int l=0;l<=m_order;l++) {
	for(int m=0;m<=l;m++) {
	    int i = MULTIPOLE_INDEX(l, m);
	    m_logger->info("Cell %d M[%d][%d] = %f + i*%f", cellID, l, m, gamma[i], gamma[i+1]);
	}
    }
}


void fmm::p2m(iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_gamma)
{
    iris_real r2 = x * x + y * y + z * z;
    iris_real zz = z + z;
    std::complex<iris_real> x_plus_iy(x, y);
    std::complex<iris_real> t = 1;
    iris_real next = q;
    iris_real itz = z + zz;
    for(int m = 0; m < m_order; m++) {
	// 1. gamma_m^m
	iris_real gamma_m_m = next;
	int i = MULTIPOLE_INDEX(m, m);
	out_gamma[i] += t.real() * gamma_m_m;
	out_gamma[i+1] += t.imag() * gamma_m_m;

	next = gamma_m_m / (2*(m+1));

	// 2. gamma_m+1^m
	iris_real gamma_mplus1_m = z * gamma_m_m;
	i = MULTIPOLE_INDEX(m+1, m);
	out_gamma[i] += t.real() * gamma_mplus1_m;
	out_gamma[i+1] += t.imag() * gamma_mplus1_m;

	iris_real prev2 = gamma_m_m;
	iris_real prev1 = gamma_mplus1_m;
	iris_real itz1 = itz;
	for(int l = m+2; l <= m_order; l++) {

	    // 3. gamma_l_m
	    iris_real gamma_l_m = (itz1 * prev1 - r2 * prev2) / (l * l - m * m);
	    i = MULTIPOLE_INDEX(l, m);
	    out_gamma[i] += t.real() * gamma_l_m;
	    out_gamma[i+1] += t.imag() * gamma_l_m;

	    prev2 = prev1;
	    prev1 = gamma_l_m;
	    itz1 += zz;
	}
	t *= x_plus_iy;
	itz += zz;
    }
    int i = MULTIPOLE_INDEX(m_order, m_order);
    out_gamma[i] += t.real() * next;
    out_gamma[i+1] += t.imag() * next;
}

void fmm::solve()
{
    m_logger->trace("fmm::solve()");

    // find the number of particles stored on this processor
    set_natoms_local();

    // transform charges to particle structure and sort them by cellID
    particle_t *particles = charges2particles();
    qsort(particles, m_natoms_local, sizeof(particle_t), sort_by_cellID_descending);

    // create leaf-level cells, grouping particles with the same cellID into same cell
    size_t nleafs;
    leaf_t *leafs = particles2leafs(particles, &nleafs);
    
    // evaluate P2M for all the leafs
    eval_p2m(leafs, particles, nleafs);
    
    delete[] particles;
    memory::destroy_1d(leafs);
    exit(-1);
}
