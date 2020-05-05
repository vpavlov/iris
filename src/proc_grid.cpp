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
#include <mpi.h>
#include "iris.h"
#include "proc_grid.h"
#include "comm_rec.h"
#include "logger.h"
#include "factorizer.h"

#include "domain.h"
#include "memory.h"
#include "mesh.h"

using namespace ORG_NCSA_IRIS;
using namespace std;


proc_grid::proc_grid(iris *obj)
    : grid(obj, "Processor")
{
}

proc_grid::~proc_grid()
{
}

// The proc grid, apart from doing whatever grid is doing in commit, must also
// check if mesh and domain are initialized and also set them dirty if it was
// re-configured.
void proc_grid::commit()
{    
    if(!m_mesh->m_initialized) {
	throw std::logic_error("proc_grid commit called, but mesh is not initialized!");
    }

    if(!m_domain->m_initialized) {
	throw std::logic_error("proc_grid commit called, but domain is not initialized!");
    }

    bool tmp_dirty = m_dirty;

    if(m_dirty) {
	figure_out_layout();
    }
        
    grid::commit();

    if(tmp_dirty) {

	// other configuration that depends on ours must be re-set
	if(m_mesh != NULL) {
	    m_mesh->m_dirty = true;
	}
	if(m_domain != NULL) {
	    m_domain->m_dirty = true;
	}
    }
}

void proc_grid::figure_out_layout()
{
    if(m_mesh->m_size[0] >= m_local_comm->m_size) {
	m_layout = IRIS_LAYOUT_PLANES_YZ;
	set_pref(0, 1, 1);
    }else if(m_mesh->m_size[1] >= m_local_comm->m_size) {
	m_layout = IRIS_LAYOUT_PLANES_ZX;
	set_pref(1, 0, 1);
    }else if(m_mesh->m_size[2] >= m_local_comm->m_size) {
	m_layout = IRIS_LAYOUT_PLANES_XY;
	set_pref(1, 1, 0);
    }else if(m_mesh->m_size[0]*m_mesh->m_size[1] >= m_local_comm->m_size) {
	m_layout = IRIS_LAYOUT_PENCILS_Z;
	set_pref(0, 0, 1);
    }else if(m_mesh->m_size[0]*m_mesh->m_size[2] >= m_local_comm->m_size) {
	m_layout = IRIS_LAYOUT_PENCILS_Y;
	set_pref(0, 1, 0);
    }else if(m_mesh->m_size[1]*m_mesh->m_size[2] >= m_local_comm->m_size) {
	m_layout = IRIS_LAYOUT_PENCILS_X;
	set_pref(1, 0, 0);
    }else {
	throw std::domain_error("Impossible to hold a whole dimension on a processor, so cannot do even 1D FFT!");
    }
}
