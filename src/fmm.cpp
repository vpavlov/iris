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
#include "fmm.h"
#include "fmm_tree.h"
#include "logger.h"
#include "domain.h"
#include "comm_rec.h"

using namespace ORG_NCSA_IRIS;

fmm::fmm(iris *obj):
    solver(obj), m_order(0),
    m_m2m_scratch(NULL), m_local_boxes(NULL),
    m_local_tree(NULL), m_LET(NULL)
{
}

fmm::~fmm()
{
    if(m_local_tree != NULL) {
	delete m_local_tree;
    }
    if(m_LET != NULL) {
	delete m_LET;
    }
    memory::destroy_1d(m_m2m_scratch);
    memory::destroy_1d(m_local_boxes);
}


//
// This gets called once all computation parameters, number of particles, global
// domain, etc. is known, so the solver can configure itself.
//
void fmm::commit()
{
    // assume these will not change after the first commit;
    // otherwise move them out of if(m_dirty)
    if(m_dirty) {
	
	m_order = m_iris->m_order;  // if p = 2, we expand multipoles until Y_2^2
	
	if(m_local_tree != NULL) {
	    delete m_local_tree;
	}
	m_local_tree = new fmm_tree(m_iris);
	
	// based on the global box and the max level, determine the size of the leaf cells
	handle_box_resize();

	memory::destroy_1d(m_m2m_scratch);
	memory::create_1d(m_m2m_scratch, 2*m_local_tree->nterms(), false);
	    
	m_dirty = false;
	m_logger->info("FMM: order = %d, depth = %d", m_order, m_local_tree->depth());
    }
}


//
// Get the local box of each rank. This will be needed for the construction of
// the Local Essential Tree (LET)
//
void fmm::get_local_boxes()
{
    memory::destroy_1d(m_local_boxes);
    memory::create_1d(m_local_boxes, m_iris->m_local_comm->m_size);
    
    MPI_Allgather(&m_domain->m_local_box, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_local_boxes, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_iris->m_local_comm->m_comm);
}


//
// This gets called when the global box changes dimensions. We need to re-calculate
// a few things ourselves, e.g. leaf size, etc.
//
void fmm::handle_box_resize()
{
    if(m_local_tree != NULL) {
	m_local_tree->set_leaf_size();
    }
    get_local_boxes();
}


void fmm::solve()
{
    m_logger->trace("fmm::solve()");

    if(m_LET != NULL) {
	delete m_LET;
    }
    
    m_local_tree->compute_local(m_m2m_scratch);
    m_LET = m_local_tree->compute_LET(m_local_boxes);
    
    MPI_Barrier(m_iris->server_comm());
    exit(-1);
}
