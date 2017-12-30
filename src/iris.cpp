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

void iris::domain_set_box(iris_real in_box_min[3], iris_real in_box_max[3])
{
    the_domain->set_box(in_box_min, in_box_max);
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
    iris_real *local_boxes_min;
    iris_real *local_boxes_max;
    int sz = 3 * the_comm->iris_size;

    if(the_comm->iris_rank == 0) {
	memory::create_1d(local_boxes_min, sz);
	memory::create_1d(local_boxes_max, sz);
    }
    
    MPI_Request req1, req2;
    MPI_Status status1, status2;

    MPI_Gather(the_domain->loc_box_min, 3, IRIS_REAL,
	       local_boxes_min, 3, IRIS_REAL,
	       0, the_comm->iris_comm);

    if(the_comm->iris_rank == 0) {
	MPI_Isend(local_boxes_min, sz, IRIS_REAL, the_comm->sim_master,
		  IRIS_TAG_LOCAL_BOXES_MIN,
		  the_comm->uber_comm, &req1);
    }
	      
    MPI_Gather(the_domain->loc_box_max, 3, IRIS_REAL,
	       local_boxes_max, 3, IRIS_REAL,
	       0, the_comm->iris_comm);

    if(the_comm->iris_rank == 0) {
	MPI_Isend(local_boxes_max, sz, IRIS_REAL, the_comm->sim_master,
		  IRIS_TAG_LOCAL_BOXES_MAX,
		  the_comm->uber_comm, &req2);
    }

    if(the_comm->iris_rank == 0) {
	MPI_Wait(&req1, &status1);
	memory::destroy_1d(local_boxes_min);

	MPI_Wait(&req2, &status2);
	memory::destroy_1d(local_boxes_max);
    }
}

// This must be called from simulation master only.
// Paired to the Isends in __announce_local_boxes
void iris::recv_local_boxes(MPI_Comm comm, int iris_comm_size,
			    iris_real *&out_local_boxes_min,
			    iris_real *&out_local_boxes_max)
{
    int sz = iris_comm_size * 3;
    memory::create_1d(out_local_boxes_min, sz);
    memory::create_1d(out_local_boxes_max, sz);
    MPI_Recv(out_local_boxes_min, sz, IRIS_REAL, MPI_ANY_SOURCE,
	     IRIS_TAG_LOCAL_BOXES_MIN, comm, MPI_STATUS_IGNORE);
    MPI_Recv(out_local_boxes_max, sz, IRIS_REAL, MPI_ANY_SOURCE,
	     IRIS_TAG_LOCAL_BOXES_MAX, comm, MPI_STATUS_IGNORE);
}
