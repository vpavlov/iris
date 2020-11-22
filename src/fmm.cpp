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
#include "timer.h"

using namespace ORG_NCSA_IRIS;

fmm::fmm(iris *obj):
    solver(obj), m_order(0),
    m_m2m_scratch(NULL), m_local_boxes(NULL),
    m_local_tree(NULL)
{
}

fmm::~fmm()
{
    if(m_local_tree != NULL) {
	delete m_local_tree;
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
    memory::create_1d(m_local_boxes, m_local_comm->m_size);
    
    MPI_Allgather(&m_domain->m_local_box, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_local_boxes, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_local_comm->m_comm);
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
    
    timer tm;
    tm.start();

    m_local_tree->charges2particles();
    m_local_tree->particles2leafs();
				  
    m_local_tree->eval_p2m();                                  // evaluate P2M for all the leafs

    // NOTE: this bottom up tree construction creates the cells up the tree. It does not
    // depend on the p2m finished or not, so for the GPU impl this can be done on the CPU
    // while the GPU performs P2M
    m_local_tree->bottom_up();  // construct tree bottom up;

    // NOTE: for the GPU impl, bottom_up and eval_p2m has to be finished before this
    // can commence.
    m_local_tree->eval_m2m(m_m2m_scratch);  // evaluate M2M for all the cells that we have

    tm.stop();
    m_logger->info("FMM solve wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
    
    MPI_Barrier(m_iris->server_comm());
    exit(-1);
}
