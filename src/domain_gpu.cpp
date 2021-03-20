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
#include "iris_gpu.h"
#include "domain_gpu.h"
#include "logger_gpu.h"
#include "proc_grid_gpu.h"
#include "mesh_gpu.h"
#include "poisson_solver_gpu.h"
#include "comm_rec_gpu.h"

using namespace ORG_NCSA_IRIS;

domain_gpu::domain_gpu(iris_gpu *obj)
  :state_accessor_gpu(obj), m_initialized(false), m_dirty(true), m_local_boxes(NULL)
{
  int size = sizeof(box_t<iris_real>) * m_iris->m_server_size;
  m_local_boxes = (box_t<iris_real> *)memory::wmalloc(size);
}

domain_gpu::~domain_gpu()
{
  memory::wfree(m_local_boxes);
}

void domain_gpu::set_global_box(iris_real x0, iris_real y0, iris_real z0,
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

    m_mesh->handle_box_resize();
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

void domain_gpu::commit()
{
    if(!m_initialized) {
	throw std::logic_error("domain commit called without global box being initialized!");
    }

    if(m_dirty) {
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
	}else {								\
	    m_local_box.IHI = m_global_box.IHI;				\
	}
	
	CALC_LOCAL(xlo, xhi, xsize, xsplit, 0);
	CALC_LOCAL(ylo, yhi, ysize, ysplit, 1);
	CALC_LOCAL(zlo, zhi, zsize, zsplit, 2);

#undef CALC_LOCAL
	
	m_local_box.xsize = m_local_box.xhi - m_local_box.xlo;
	m_local_box.ysize = m_local_box.yhi - m_local_box.ylo;
	m_local_box.zsize = m_local_box.zhi - m_local_box.zlo;

	m_logger->trace("Local box is %g x %g x %g: [%g:%g][%g:%g][%g:%g]",
			m_local_box.xsize, m_local_box.ysize,
			m_local_box.zsize,
			m_local_box.xlo, m_local_box.xhi,
			m_local_box.ylo, m_local_box.yhi,
			m_local_box.zlo, m_local_box.zhi);

	MPI_Allgather(&m_local_box, sizeof(box_t<iris_real>), MPI_BYTE,
		      m_local_boxes, sizeof(box_t<iris_real>), MPI_BYTE,
		      m_local_comm->m_comm);

	m_dirty = false;
    }
}
