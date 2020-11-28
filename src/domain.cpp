// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2018, the National Center for Supercomputing Applications
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
#include <stdexcept>
#include "iris.h"
#include "domain.h"
#include "logger.h"
#include "proc_grid.h"
#include "mesh.h"
#include "solver.h"
#include "comm_rec.h"
#include "utils.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

domain::domain(iris *obj)
    :state_accessor(obj), m_initialized(false), m_dirty(true)
{
}

domain::~domain()
{
}

void domain::set_global_box(iris_real x0, iris_real y0, iris_real z0,
			    iris_real x1, iris_real y1, iris_real z1)
{
    if(x0 >= x1 || y0 >= y1 || z0 >= z1)
    {
	throw std::domain_error("Invalid global bounding box!");
    }

    m_global_box.xlo = x0;
    m_global_box.ylo = y0;
    m_global_box.zlo = z0;

    m_global_box.xhi = x1;
    m_global_box.yhi = y1;
    m_global_box.zhi = z1;

    m_global_box.xsize = x1 - x0;
    m_global_box.ysize = y1 - y0;
    m_global_box.zsize = z1 - z0;

    m_initialized = true;
    m_dirty = true;

    if(m_mesh != NULL) {
	m_mesh->handle_box_resize();
    }
    
    if(m_solver != NULL) {
	m_solver->handle_box_resize();
    }

    m_logger->trace("Global box is %g x %g x %g: [%g:%g][%g:%g][%g:%g]",
		    m_global_box.xsize, m_global_box.ysize,
		    m_global_box.zsize,
		    m_global_box.xlo, m_global_box.xhi,
		    m_global_box.ylo, m_global_box.yhi,
		    m_global_box.zlo, m_global_box.zhi);
}

void domain::commit()
{
    if(!m_initialized) {
	throw std::logic_error("domain commit called without global box being initialized!");
    }

    if(m_dirty) {
	if(m_iris->m_which_solver != IRIS_SOLVER_FMM) {
	    iris_real *xsplit = m_proc_grid->m_xsplit;
	    iris_real *ysplit = m_proc_grid->m_ysplit;
	    iris_real *zsplit = m_proc_grid->m_zsplit;
	    int *c = m_proc_grid->m_coords;
	    int *size = m_proc_grid->m_size;
	    
	    // OAOO helper
#define CALC_LOCAL(ILO, IHI, ISIZE, ISPLIT, I)				\
	    m_local_box.ILO = m_global_box.ILO + m_global_box.ISIZE * ISPLIT[c[I]]; \
	    if(c[I] < size[I] - 1) {					\
		m_local_box.IHI = m_global_box.ILO + m_global_box.ISIZE * ISPLIT[c[I] + 1]; \
	    }else {							\
		m_local_box.IHI = m_global_box.IHI;			\
	    }
	    
	    CALC_LOCAL(xlo, xhi, xsize, xsplit, 0);
	    CALC_LOCAL(ylo, yhi, ysize, ysplit, 1);
	    CALC_LOCAL(zlo, zhi, zsize, zsplit, 2);
	    
#undef CALC_LOCAL
	    
	    m_local_box.xsize = m_local_box.xhi - m_local_box.xlo;
	    m_local_box.ysize = m_local_box.yhi - m_local_box.ylo;
	    m_local_box.zsize = m_local_box.zhi - m_local_box.zlo;
	}else {
	    octsect_dd();
	    octsect_update_hood();
	}
	m_logger->info("Local box is %g x %g x %g: [%g:%g][%g:%g][%g:%g]",
		       m_local_box.xsize, m_local_box.ysize,
		       m_local_box.zsize,
		       m_local_box.xlo, m_local_box.xhi,
		       m_local_box.ylo, m_local_box.yhi,
		       m_local_box.zlo, m_local_box.zhi);
	    
	m_dirty = false;
    }
}

// TODO: move this in grid, were it belongs...
void domain::octsect_dd()
{
    int sz = m_local_comm->m_size;
    if(!is_power_of_2(sz)) {
	throw std::domain_error("FMM implementation only supports server MPISIZEs which are power of 2!");
    }
    int bits = int(log(sz)/M_LN2);
    
    iris_real prev_min[3], curr_min[3];
    iris_real prev_max[3], curr_max[3];
    
    prev_min[0] = m_global_box.xlo;
    prev_min[1] = m_global_box.ylo;
    prev_min[2] = m_global_box.zlo;

    prev_max[0] = m_global_box.xhi;
    prev_max[1] = m_global_box.yhi;
    prev_max[2] = m_global_box.zhi;

    for(int i=0;i<bits;i++) {
	int d = 2 - i % 3;  // which dimension are we dividing ?
	for(int j=0;j<3;j++) {
	    curr_min[j] = prev_min[j];
	    curr_max[j] = prev_max[j];
	}
	if (m_local_comm->m_rank & (1 << (bits-1-i))) {
	    curr_min[d] = (prev_min[d] + prev_max[d]) / 2;
	}else {
	    curr_max[d] = (prev_min[d] + prev_max[d]) / 2;
	}
	for(int j=0;j<3;j++) {
	    prev_min[j] = curr_min[j];
	    prev_max[j] = curr_max[j];
	}
    }
    m_local_box.xlo = curr_min[0];
    m_local_box.ylo = curr_min[1];
    m_local_box.zlo = curr_min[2];

    m_local_box.xhi = curr_max[0];
    m_local_box.yhi = curr_max[1];
    m_local_box.zhi = curr_max[2];

    m_local_box.xsize = m_local_box.xhi - m_local_box.xlo;
    m_local_box.ysize = m_local_box.yhi - m_local_box.ylo;
    m_local_box.zsize = m_local_box.zhi - m_local_box.zlo;
}


// TODO: move this in grid, where it belons
void domain::octsect_update_hood()
{
    int sz = m_local_comm->m_size;
    int bits = int(log(sz)/M_LN2);

    int max[3];
    max[0] = 0;
    max[1] = 0;
    max[2] = 0;
    for(int p=0;p<sz;p++) {
	int coords[3];
	coords[0] = 0;
	coords[1] = 0;
	coords[2] = 0;
	for(int i=0;i<bits;i++) {
	    int d = 2 - i % 3;  // which dimension are we dividing ?
	    if (p & (1 << (bits-1-i))) {
		coords[d] += (1 << (bits-1-i)/3);
	    }
	}
	max[0] = MAX(max[0], coords[0]);
	max[1] = MAX(max[1], coords[1]);
	max[2] = MAX(max[2], coords[2]);
    }

    memory::destroy_3d(m_proc_grid->m_ranks);
    memory::create_3d(m_proc_grid->m_ranks, max[0]+1, max[1]+1, max[2]+1);
    for(int p=0;p<sz;p++) {
	int coords[3];
	coords[0] = 0;
	coords[1] = 0;
	coords[2] = 0;
	for(int i=0;i<bits;i++) {
	    int d = 2 - i % 3;  // which dimension are we dividing ?
	    if (p & (1 << (bits-1-i))) {
		coords[d] += (1 << (bits-1-i)/3);
	    }
	}
	m_proc_grid->m_ranks[coords[0]][coords[1]][coords[2]] = p;
	if(p == m_local_comm->m_rank) {
	    m_proc_grid->m_coords[0] = coords[0];
	    m_proc_grid->m_coords[1] = coords[1];
	    m_proc_grid->m_coords[2] = coords[2];
	}
    }

    // right
    if(m_proc_grid->m_coords[0] + 1 <= max[0]) {
	m_proc_grid->m_hood[0][0] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0] + 1][m_proc_grid->m_coords[1]][m_proc_grid->m_coords[2]];
    }else if (m_proc_grid->m_pbc[0] != 0) {
	m_proc_grid->m_hood[0][0] = m_proc_grid->m_ranks[0][m_proc_grid->m_coords[1]][m_proc_grid->m_coords[2]];
    }else {
	m_proc_grid->m_hood[0][0] = -1;
    }

    // left
    if(m_proc_grid->m_coords[0] - 1 >= 0) {
	m_proc_grid->m_hood[0][1] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0] - 1][m_proc_grid->m_coords[1]][m_proc_grid->m_coords[2]];
    }else if (m_proc_grid->m_pbc[0] != 0) {
	m_proc_grid->m_hood[0][1] = m_proc_grid->m_ranks[max[0]][m_proc_grid->m_coords[1]][m_proc_grid->m_coords[2]];
    }else {
	m_proc_grid->m_hood[0][1] = -1;
    }

    // up
    if(m_proc_grid->m_coords[1] + 1 <= max[1]) {
	m_proc_grid->m_hood[1][0] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][m_proc_grid->m_coords[1]+1][m_proc_grid->m_coords[2]];
    }else if (m_proc_grid->m_pbc[1] != 0) {
	m_proc_grid->m_hood[1][0] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][0][m_proc_grid->m_coords[2]];
    }else {
	m_proc_grid->m_hood[1][0] = -1;
    }

    // down
    if(m_proc_grid->m_coords[1] - 1 >= 0) {
	m_proc_grid->m_hood[1][1] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][m_proc_grid->m_coords[1]-1][m_proc_grid->m_coords[2]];
    }else if (m_proc_grid->m_pbc[1] != 0) {
	m_proc_grid->m_hood[1][1] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][max[1]][m_proc_grid->m_coords[2]];
    }else {
	m_proc_grid->m_hood[1][1] = -1;
    }

    // back
    if(m_proc_grid->m_coords[2] + 1 <= max[2]) {
	m_proc_grid->m_hood[2][0] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][m_proc_grid->m_coords[1]][m_proc_grid->m_coords[2]+1];
    }else if (m_proc_grid->m_pbc[2] != 0) {
	m_proc_grid->m_hood[2][0] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][m_proc_grid->m_coords[1]][0];
    }else {
	m_proc_grid->m_hood[2][0] = -1;
    }

    // front
    if(m_proc_grid->m_coords[2] - 1 >= 0) {
	m_proc_grid->m_hood[2][1] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][m_proc_grid->m_coords[1]][m_proc_grid->m_coords[2]-1];
    }else if (m_proc_grid->m_pbc[2] != 0) {
	m_proc_grid->m_hood[2][1] = m_proc_grid->m_ranks[m_proc_grid->m_coords[0]][m_proc_grid->m_coords[1]][max[2]];
    }else {
	m_proc_grid->m_hood[2][1] = -1;
    }
}
