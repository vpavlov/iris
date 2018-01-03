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
#include <omp.h>
#include "iris.h"
#include "domain.h"
#include "comm.h"
#include "mpi_tags.h"
#include "memory.h"
#include "mesh.h"

using namespace ORG_NCSA_IRIS;

iris::iris(MPI_Comm uber_comm, MPI_Comm iris_comm, int sim_master)
{
    the_mesh = NULL;  // to prevent domain from notifying the mesh for box changed
    the_comm = new comm(this, uber_comm, iris_comm, sim_master);
    the_domain = new domain(this);
    the_mesh = new mesh(this);
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
    the_domain->setup_local();
    the_mesh->setup_local();
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

// Receive IRIS local boxes in all procs in pp_comm
// Paired to the Isends in __announce_local_boxes
// iris_comm_size: the size of the dedicated IRIS communicator (# of boxes)
// rank: this proc rank (in uber comm)
// pp_master: the proc rank (in uber_comm) that will receive the data from IRIS
// uber_comm: usually MPI_COMM_WORLD
// pp_comm: communicator for PP procs only (receivers of the data)
// out_local_boxes: the result
// 
// Paired to the send in __announce_local_boxes
void iris::recv_local_boxes(int iris_comm_size,
			    int rank,
			    int pp_master,
			    MPI_Comm uber_comm, 
			    MPI_Comm pp_comm,
			    iris_real *&out_local_boxes)
{
    int sz = 6 * iris_comm_size;
    memory::create_1d(out_local_boxes, sz);

    if(rank == pp_master) {
	MPI_Recv(out_local_boxes, sz, IRIS_REAL, MPI_ANY_SOURCE,
		 IRIS_TAG_LOCAL_BOXES, uber_comm, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(out_local_boxes, sz, IRIS_REAL, pp_master, pp_comm);
}

void iris::recv_atoms()
{
    MPI_Status status;

#pragma omp parallel
    {
#pragma omp single
	{
	    while(42) { // will break from within the loop
		MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG,
			  the_comm->uber_comm, &status);
		if(status.MPI_TAG == IRIS_TAG_ATOMS) {
		    int msg_size;
		    int natoms;
		    MPI_Get_count(&status, IRIS_REAL, &msg_size);
		    if(msg_size % 4 != 0) {
			throw std::length_error("Unexpected message size while receiving atoms!");
		    }
		    
		    natoms = msg_size / 4;
		    if(natoms != 0) {
			int src = status.MPI_SOURCE;
			iris_real **atoms;
			memory::create_2d(atoms, natoms, 4);
			MPI_Recv(&(atoms[0][0]), msg_size, IRIS_REAL, 
				 status.MPI_SOURCE, status.MPI_TAG,
				 the_comm->uber_comm,
				 MPI_STATUS_IGNORE);
#pragma omp task firstprivate(atoms, natoms, src)
			{
			    printf("%d[%d]: Received %d atoms from %d\n",
			           the_comm->uber_rank, omp_get_thread_num(), natoms, src);
			    the_mesh->assign_charges(atoms, natoms);
			    printf("%d[%d]: charge assignment done\n",
				   the_comm->uber_rank, omp_get_thread_num());
			    memory::destroy_2d(atoms);
			}
		    }else {
			// printf("%d: Received 0 atoms from %d\n",
			//        the_comm->uber_rank, status.MPI_SOURCE);
			MPI_Recv(NULL, msg_size, IRIS_REAL,
				 status.MPI_SOURCE, status.MPI_TAG,
				 the_comm->uber_comm,
				 MPI_STATUS_IGNORE);
		    }
		}else if(status.MPI_TAG == IRIS_TAG_ATOMS_EOF) {
		    int dummy;
		    MPI_Recv(&dummy, 1, MPI_INT,
			     status.MPI_SOURCE, status.MPI_TAG,
			     the_comm->uber_comm,
			     MPI_STATUS_IGNORE);
		    break;
		}else {
		    throw std::logic_error("Unexpected MPI message while receiving atoms!");
		    
		}
	    }
	}
    }
    the_mesh->dump_rho("NaCl-rho-2");
}

void iris::mesh_set_size(int nx, int ny, int nz)
{
    the_mesh->set_size(nx, ny, nz);
}
