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
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include "iris.h"
#include "domain.h"
#include "comm.h"
#include "event_codes.h"
#include "memory.h"
#include "mesh.h"
#include "event.h"
#include "debug.h"

using namespace ORG_NCSA_IRIS;

iris::iris(MPI_Comm uber_comm, MPI_Comm iris_comm, int sim_master)
{
    the_mesh = NULL;  // to prevent domain from notifying the mesh for box changed
    the_debug = new debug(this);
    the_comm = new comm(this, uber_comm, iris_comm, sim_master);
    the_domain = new domain(this);
    the_mesh = new mesh(this);

    __event_handlers[IRIS_EVENT_ATOMS] = &iris::__handle_atoms;
    __event_handlers[IRIS_EVENT_ATOMS_EOF] = &iris::__handle_atoms_eof;
    __event_handlers[IRIS_EVENT_BARRIER] = &iris::__handle_barrier;

    rest_time = 100;  // sleep for 100 microseconds if there's nothing to do
    set_state(IRIS_STATE_INITIALIZED);
    __barrier_posted = false;
}

iris::~iris()
{
    delete the_comm;
    delete the_domain;
    delete the_mesh;
    delete the_debug;
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

void iris::mesh_set_size(int nx, int ny, int nz)
{
    the_mesh->set_size(nx, ny, nz);
}

// call this after all user-configuration is set so we can calculate whatever
// we need in order to start looping
void iris::apply_conf()
{
    the_comm->setup_grid();
    the_domain->setup_local();
    the_mesh->setup_local();
    the_mesh->reset_rho();

    set_state(IRIS_STATE_WAITING_FOR_ATOMS);
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
		 IRIS_EVENT_LOCAL_BOXES,
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
		 IRIS_EVENT_LOCAL_BOXES, uber_comm, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(out_local_boxes, sz, IRIS_REAL, pp_master, pp_comm);
}

typedef void (iris::*event_handler_t)(event_t);

void iris::__handle_event(event_t event)
{
    event_handler_t fun = __event_handlers[event.code];
    if(fun) {
	(this->*fun)(event);
    }else {
	__handle_unimplemented(event);
    }
    memory::wfree(event.data);
}

event_t iris::poke_mpi_event(MPI_Comm comm, bool &out_has_event)
{
    MPI_Status status;
    int nbytes;
    void *msg;
    event_t retval;
    int has_event;

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &has_event, &status);
    if(has_event) {
	MPI_Get_count(&status, MPI_BYTE, &nbytes);
	msg = memory::wmalloc(nbytes);
	MPI_Recv(msg, nbytes, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG,
		 comm, MPI_STATUS_IGNORE);
	retval.comm = comm;
	retval.peer = status.MPI_SOURCE;
	retval.code = status.MPI_TAG;
	retval.size = nbytes;
	retval.data = msg;
	out_has_event = true;
    }else {
	out_has_event = false;
    }
    return retval;
}

void iris::send_event(event_t event)
{
    MPI_Send(event.data, event.size, MPI_BYTE, event.peer, event.code, event.comm);
}

event_t iris::poke_uber_event(bool &out_has_event)
{
    return poke_mpi_event(the_comm->uber_comm, out_has_event);
}

event_t iris::poke_iris_event(bool &out_has_event)
{
    return poke_mpi_event(the_comm->iris_comm, out_has_event);
}

event_t iris::poke_barrier_event(bool &out_has_event)
{
    event_t ev;

    out_has_event = false;
    if(__barrier_posted) {
	int has_event;
	MPI_Status status;
	MPI_Test(&__barrier_req, &has_event, &status);
	if(has_event) {
	    MPI_Status status;
	    MPI_Wait(&__barrier_req, MPI_STATUS_IGNORE);
	    MPI_Barrier(the_comm->iris_comm);
	    the_debug->trace("----------");

	    ev.comm = the_comm->iris_comm;
	    ev.peer = -1;
	    ev.code = IRIS_EVENT_BARRIER;
	    ev.size = 0;
	    ev.data = NULL;
	    out_has_event = true;
	    __barrier_posted = false;
	}
    }
    return ev;
}

event_t iris::poke_event(bool &out_has_event)
{
    event_t event;

    event = poke_uber_event(out_has_event);
    if(out_has_event) {
	return event;
    }

    event = poke_iris_event(out_has_event);
    if(out_has_event) {
	return event;
    }

    event = poke_barrier_event(out_has_event);

    return event;
}

void iris::run()
{
#pragma omp parallel
#pragma omp single
    {
	__quit_event_loop = false;
	suspend_event_loop = false;
	while(!__quit_event_loop) {
	    if(!suspend_event_loop) {
		bool has_event;
		event_t event = poke_event(has_event);
#pragma omp task default(none) firstprivate(has_event, event)
		if(has_event) {
		    __handle_event(event);
		}else {
		    usleep(rest_time);  // suspend for some time so others can work
		}
	    }
	}
    }
}

void iris::set_state(int in_state)
{
    if(in_state != IRIS_STATE_INITIALIZED) {
	the_debug->trace("Changing state %d -> %d", state, in_state);
    }else {
	the_debug->trace("Initializing state to %d", in_state);
    }
    state = in_state;

    switch(state) {

    case IRIS_STATE_HAS_RHO:
	__quit_event_loop = true;
	break;
    }
}

void iris::post_barrier()
{
    MPI_Ibarrier(the_comm->iris_comm, &__barrier_req);
    __barrier_posted = true;
}

////////////////////////////////////////////////////////////////////////////////
// Event handlers
////////////////////////////////////////////////////////////////////////////////

void iris::__handle_unimplemented(event_t event)
{
    the_debug->trace("Unimplemented event: %d", event.code);
}

void iris::__handle_atoms(event_t event)
{
    if(state != IRIS_STATE_WAITING_FOR_ATOMS) {
	throw std::logic_error("Receiving atoms while in un-configured state!");
    }

    int unit = 4 * sizeof(iris_real);
    if(event.size % unit != 0) {
	throw std::length_error("Unexpected message size while receiving atoms!");
    }

    int natoms = event.size / unit;
    if(natoms != 0) {
	the_debug->trace("Received %d atoms from %d", natoms, event.peer);
	the_mesh->assign_charges((iris_real *)event.data, natoms);
	the_debug->trace("Charge assignment from %d done", event.peer);
	MPI_Request req;
	MPI_Isend(NULL, 0, MPI_INT, event.peer, IRIS_EVENT_ATOMS_ACK, event.comm, &req);
    }

}

void iris::__handle_atoms_eof(event_t event)
{
    if(state != IRIS_STATE_WAITING_FOR_ATOMS) {
	throw std::logic_error("Receiving atoms EOF while in un-configured state!");
    }

    the_debug->trace("All atoms received");
    post_barrier();
}

void iris::__handle_barrier(event_t ev)
{
    if(state == IRIS_STATE_WAITING_FOR_ATOMS) {
	the_mesh->exchange_halo();

	char fname[256];
	sprintf(fname, "NaCl-rho-%d-%d-%d-%d", the_comm->uber_size, omp_get_num_threads(), the_comm->uber_rank, omp_get_thread_num());
	the_mesh->dump_rho(fname);

	set_state(IRIS_STATE_HAS_RHO);
    }
}
