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
#include "domain.h"
#include "comm.h"
#include "mpi_tags.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

iris::iris(MPI_Comm uber_comm, MPI_Comm iris_comm, int sim_master)
{
    the_comm = new comm(this, uber_comm, iris_comm, sim_master);
    the_domain = new domain(this);
}

iris::~iris()
{
    delete the_comm;
    delete the_domain;
}

void iris::domain_set_dimensions(int in_dimensions)
{
    the_domain->set_dimensions(in_dimensions);
}

void iris::domain_set_box(iris_real x0, iris_real y0, iris_real z0,
			  iris_real x1, iris_real y1, iris_real z1)
{
    the_domain->set_box(x0, y0, z0, x1, y1, z1);
}

void iris::comm_set_grid_pref(int x, int y, int z)
{
    the_comm->set_grid_pref(x, y, z);
}

// call this after all user-configuration is set so we can calculate whatever
// we need in order to start looping
void iris::apply_conf()
{
    the_comm->setup_grid();
    the_domain->setup_local_box();
    __announce_loc_box_info();
}

// This gathers the local boxes of all IRIS procs and sends them to proc 0
// of the uber comm (e.g. simulation master). It can then re-distribute this
// information to PP-only nodes so they know which atoms to send to which
// IRIS procs.
void iris::__announce_loc_box_info()
{
    iris_real *local_boxes;
    int sz = 6 * the_comm->iris_size;

    if(the_comm->iris_rank == 0) {
	memory::create_1d(local_boxes, sz);
    }
    
    MPI_Gather(the_domain->lbox_sides, 6, IRIS_REAL,
	       local_boxes, 6, IRIS_REAL,
	       0, the_comm->iris_comm);

    if(the_comm->iris_rank == 0) {
	MPI_Send(local_boxes, sz, IRIS_REAL, the_comm->sim_master,
		 IRIS_TAG_LOCAL_BOXES,
		 the_comm->uber_comm);
	memory::destroy_1d(local_boxes);
    }

}

// This must be called from simulation master only.
// Paired to the Isends in __announce_local_boxes
void iris::recv_local_boxes(MPI_Comm comm, int iris_comm_size,
			    iris_real *&out_local_boxes)
{
    int sz = 6 * iris_comm_size;
    memory::create_1d(out_local_boxes, sz);
    MPI_Recv(out_local_boxes, sz, IRIS_REAL, MPI_ANY_SOURCE,
	     IRIS_TAG_LOCAL_BOXES, comm, MPI_STATUS_IGNORE);
}
