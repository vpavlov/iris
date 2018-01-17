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
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include "iris.h"
#include "logger.h"
#include "comm_rec.h"
#include "domain.h"
#include "mesh.h"
#include "charge_assigner.h"
#include "proc_grid.h"
#include "memory.h"
#include "tags.h"

using namespace ORG_NCSA_IRIS;

iris::iris(MPI_Comm in_uber_comm)
    :m_role(IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER), m_local_leader(0),
     m_remote_leader(0), m_client_size(0), m_server_size(0)
{
    init(in_uber_comm, in_uber_comm);
}

iris::iris(MPI_Comm in_uber_comm, int in_leader)
    :m_role(IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER), m_local_leader(in_leader),
     m_remote_leader(in_leader), m_client_size(0), m_server_size(0)
{
    init(in_uber_comm, in_uber_comm);
}

iris::iris(int in_client_size, int in_server_size,
	   int in_role, MPI_Comm in_local_comm,
	   MPI_Comm in_uber_comm, int in_remote_leader)
    :m_role(in_role), m_local_leader(0), m_remote_leader(in_remote_leader),
     m_client_size(in_client_size), m_server_size(in_server_size)
{
    init(in_local_comm, in_uber_comm);
}

iris::iris(int in_client_size, int in_server_size,
	   int in_role, MPI_Comm in_local_comm, int in_local_leader,
	   MPI_Comm in_uber_comm, int in_remote_leader)
    :m_role(in_role), m_local_leader(in_local_leader),
     m_remote_leader(in_remote_leader), m_client_size(in_client_size),
     m_server_size(in_server_size)
{
    init(in_local_comm, in_uber_comm);
}

void iris::init(MPI_Comm in_local_comm, MPI_Comm in_uber_comm)
{
    m_quit = false;
    m_inter_comm = NULL;
    m_local_comm = new comm_rec(this, in_local_comm);
    m_uber_comm = new comm_rec(this, in_uber_comm);

    if(!is_both()) {
	// For the intercomm to be created, the two groups must be disjoint, and
	// this is not the case when nodes are client/server.
	MPI_Comm inter_comm;
	MPI_Intercomm_create(m_local_comm->m_comm,
			     m_local_leader,
			     m_uber_comm->m_comm,
			     m_remote_leader,
			     IRIS_TAG_INTERCOMM_CREATE,
			     &inter_comm);
	m_inter_comm = new comm_rec(this, inter_comm);
	MPI_Comm_free(&inter_comm);
    }else {
	m_client_size = m_server_size = m_local_comm->m_size;
	m_mixed_mode_pending = new int[m_server_size];
	for(int i=0;i<m_server_size;i++) {
	    m_mixed_mode_pending[i] = 0;
	}
	MPI_Win_create(m_mixed_mode_pending, m_server_size, sizeof(int),
		       MPI_INFO_NULL, m_local_comm->m_comm,
		       &m_mixed_mode_pending_win);
    }

    if(is_server() && m_server_size != m_local_comm->m_size) {
	throw std::invalid_argument("Inconsistent server size!");
    }

    if(is_client() && m_client_size != m_local_comm->m_size) {
	throw std::invalid_argument("Inconsistent client size!");
    }

    m_logger = new logger(this);

    m_mesh = NULL;
    m_domain = NULL;
    m_chass = NULL;
    m_proc_grid = NULL;

    if(is_server()) {
	m_domain = new domain(this);
	m_mesh = new mesh(this);
	m_chass = new charge_assigner(this);
	m_proc_grid = new proc_grid(this);
    }

    m_state = IRIS_STATE_INITIALIZED;
    m_logger->trace("Node initialized as %s %d %s",
		    is_server()?(is_client()?"client/server":"server"):"client",
		   m_local_comm->m_rank,
		   is_leader()?"(leader)":"");
}

iris::~iris()
{
    if(m_proc_grid != NULL) {
	delete m_proc_grid;
    }

    if(m_chass != NULL) {
	delete m_chass;
    }

    if(m_mesh != NULL) {
	delete m_mesh;
    }

    if(m_domain != NULL) {
	delete m_domain;
    }


    if(is_both()) {
	delete [] m_mixed_mode_pending;
	MPI_Win_free(&m_mixed_mode_pending_win);
    }else {
	if(m_inter_comm != NULL) {
	    delete m_inter_comm;
	}
    }

    delete m_local_comm;

    m_logger->trace("Shutting down node");  // before m_uber_comm
    delete m_logger;

    delete m_uber_comm;
}

bool iris::is_leader()
{
    return m_local_comm->m_rank == m_local_leader;
};

void iris::set_global_box(iris_real x0, iris_real y0, iris_real z0,
			  iris_real x1, iris_real y1, iris_real z1)
{
    if(is_server()) {
	m_domain->set_global_box(x0, y0, z0, x1, y1, z1);
    }
}

void iris::set_mesh_size(int nx, int ny, int nz)
{
    if(is_server()) {
	m_mesh->set_size(nx, ny, nz);
    }
}

void iris::set_order(int in_order)
{
    if(is_server()) {
	m_chass->set_order(in_order);
    }
}

void iris::set_grid_pref(int x, int y, int z)
{
    m_proc_grid->set_pref(x, y, z);
}


void iris::commit()
{
    if(is_server()) {
	// Beware: order is important. Some configurations depend on other
	// being already performed
	m_chass->commit();      // does not depend on anything being commited
	m_proc_grid->commit();  // does not depend on anything being comitted
	m_domain->commit();     // depends on m_proc_grid->commit()
	m_mesh->commit();       // depends on m_proc_grid->commit()
    }
    m_state = IRIS_STATE_COMMITED;
}

// run the main server loop
void iris::run()
{
    if(is_server() && !is_client()) {
	while(!m_quit) {
	    event_t event;
	    if(m_uber_comm->peek_event(event) ||
	       m_local_comm->peek_event(event) ||
	       m_inter_comm->peek_event(event))
	    {
		process_event(&event);
		memory::wfree(event.data);
	    }
	}
    }
}

void iris::process_event(event_t *event)
{
    //m_logger->trace_event(event);
    switch(event->tag) {
    case IRIS_TAG_QUIT:
	m_logger->trace("Quit event received");
	m_quit = true;
	break;

    case IRIS_TAG_CHARGES:
	m_logger->trace_event(event);
	break;
    }
}


void iris::bcast_charges_to_servers(int *in_counts, iris_real *in_charges)
{
    if(!is_client()) {
	throw std::logic_error("bcast_charges_to_servers may only be called from client nodes!");
    }

    // in shared mode we only have m_local_comm; in SOC mode we use intercomm
    MPI_Comm comm = is_server()?m_local_comm->m_comm:m_inter_comm->m_comm;

    int offset = 0;
    MPI_Request *req = new MPI_Request[m_server_size];

    if(is_server()) {
	MPI_Win_fence(MPI_MODE_NOPRECEDE, m_mixed_mode_pending_win);
    }

    for(int i=0;i<m_server_size;i++) {
	req[i] = MPI_REQUEST_NULL;
	if(in_counts[i] != 0) {
	    if(is_server() && i == m_local_comm->m_rank) {
		event_t ev;
		ev.comm = comm;
		ev.peer = i;
		ev.tag = IRIS_TAG_CHARGES;
		ev.size = 4*in_counts[i]*sizeof(iris_real);
		ev.data = in_charges + offset;
		process_event(&ev);
	    }else {
		MPI_Isend(in_charges + offset,  4*in_counts[i], IRIS_REAL, i,
			  IRIS_TAG_CHARGES, comm, &req[i]);
		if(is_server()) {
		    int one = 1;
		    MPI_Put(&one, 1, MPI_INT, i, m_local_comm->m_rank,
			    1, MPI_INT, m_mixed_mode_pending_win);
		}
	    }
	}
	offset += 4*in_counts[i];
    }


    if(is_server()) {
	MPI_Barrier(m_local_comm->m_comm);
	MPI_Win_fence(MPI_MODE_NOSUCCEED | MPI_MODE_NOSTORE,
		      m_mixed_mode_pending_win);
	for(int i=0;i<m_server_size;i++) {
	    if(m_mixed_mode_pending[i] == 1) {
		m_mixed_mode_pending[i] = 0;
		event_t ev;
		m_local_comm->get_event(ev);
		process_event(&ev);
	    }
	}
    }

    MPI_Waitall(m_server_size, req, MPI_STATUSES_IGNORE);

}

void iris::bcast_charges_eof_to_servers()
{
    if(is_client() && !is_server()) {
	// Make sure all clients have already sent their atoms before notifying
	// the server that there are no more atoms.
	MPI_Barrier(m_local_comm->m_comm);
	if(is_leader()) {
	    MPI_Request *req = new MPI_Request[m_server_size];
	    for(int i=0;i<m_server_size;i++) {
		MPI_Isend(NULL, 0, MPI_BYTE, i, IRIS_TAG_CHARGES_EOF,
			  m_inter_comm->m_comm,
			  &req[i]);
	    }
	    MPI_Waitall(m_server_size, req, MPI_STATUSES_IGNORE);
	}
    }	
}

void iris::bcast_quit_to_servers()
{
    MPI_Request *req = new MPI_Request[m_server_size];

    if(is_client() && !is_server() && is_leader()) {
	for(int i=0;i<m_server_size;i++) {
	    MPI_Isend(NULL, 0, MPI_BYTE, i, IRIS_TAG_QUIT,
		      m_inter_comm->m_comm,
		      &req[i]);
	}
	MPI_Waitall(m_server_size, req, MPI_STATUSES_IGNORE);
    }	
}



box_t<iris_real> *iris::get_local_boxes()
{
    box_t<iris_real> *local_boxes = NULL;
    int size = sizeof(box_t<iris_real>) * m_server_size;

    // Output need to be allocated by everybody except pure-server non-leaders,
    // which doesn't need it.
    if(!is_server() || is_client() || is_leader()) {
	local_boxes = (box_t<iris_real> *)memory::wmalloc(size);
    }

    if(is_server()) {
	MPI_Gather(&(m_domain->m_local_box),
		   sizeof(box_t<iris_real>)/sizeof(iris_real), IRIS_REAL,
		   local_boxes,
		   sizeof(box_t<iris_real>)/sizeof(iris_real), IRIS_REAL,
		   m_local_leader,
		   m_local_comm->m_comm);

	if(!is_client()) {
	    if(is_leader()) {
		MPI_Send(local_boxes, size, MPI_BYTE, m_remote_leader,
			 IRIS_TAG_LOCAL_BOXES, m_uber_comm->m_comm);
		memory::wfree(local_boxes);
	    }

	    return NULL;
	}
    }

    // intentional fall-through (no else if) to handle mixed mode cases

    if(is_client()) {
	if(!is_server() && is_leader()) {
	    MPI_Recv(local_boxes, size, MPI_BYTE, m_remote_leader,
		     IRIS_TAG_LOCAL_BOXES, m_uber_comm->m_comm,
		     MPI_STATUS_IGNORE);
	}
	MPI_Bcast(local_boxes, size, MPI_BYTE, m_local_leader,
		  m_local_comm->m_comm);
	return local_boxes;
    }
}



// iris::iris(MPI_Comm uber_comm, MPI_Comm iris_comm, int sim_master)
// {
//     m_mesh = NULL;  // to prevent domain from notifying the mesh for box changed
//     m_logger = new logger(this);
//     m_proc_grid = new proc_grid(this, uber_comm, iris_comm, sim_master);
//     m_domain = new domain(this);
//     m_mesh = new mesh(this);

//     __event_handlers[IRIS_EVENT_ATOMS] = &iris::__handle_atoms;
//     __event_handlers[IRIS_EVENT_ATOMS_EOF] = &iris::__handle_atoms_eof;
//     __event_handlers[IRIS_EVENT_BARRIER] = &iris::__handle_barrier;

//     rest_time = 100;  // sleep for 100 microseconds if there's nothing to do
//     set_state(IRIS_STATE_INITIALIZED);
//     __barrier_posted = false;
// }

// call this after all user-configuration is set so we can calculate whatever
// we need in order to start looping
// void iris::apply_conf()
// {
//     m_proc_grid->setup_grid();
//     m_domain->set_local_box();
//     m_mesh->setup_local();
//     m_mesh->reset_rho();

//     set_state(IRIS_STATE_WAITING_FOR_ATOMS);
//     __announce_loc_box_info();
// }

// This gathers the local boxes of all IRIS procs and sends them to proc 0
// of the uber comm (e.g. simulation master). It can then re-distribute this
// information to PP-only nodes so they know which atoms to send to which
// IRIS procs.
// void iris::__announce_loc_box_info()
// {
//     // TODO: redo this with m_inter_comm

//     // iris_real *local_boxes;
//     // int sz = 6 * m_local_comm->m_size;

//     // if(m_local_comm->m_rank == 0) {
//     // 	memory::create_1d(local_boxes, sz);
//     // }
    
//     // MPI_Gather(&(m_domain->m_local_box), 6, IRIS_REAL,
//     // 	       local_boxes, 6, IRIS_REAL,
//     // 	       0, m_local_comm->m_comm);

//     // if(m_local_comm->m_rank == 0) {
//     // 	MPI_Send(local_boxes, sz, IRIS_REAL, m_proc_grid->sim_master,
//     // 		 IRIS_EVENT_LOCAL_BOXES,
//     // 		 m_uber_comm->comm);
//     // 	memory::destroy_1d(local_boxes);
//     // }
// }

// // Receive IRIS local boxes in all procs in pp_comm
// // Paired to the Isends in __announce_local_boxes
// // iris_comm_size: the size of the dedicated IRIS communicator (# of boxes)
// // rank: this proc rank (in uber comm)
// // pp_master: the proc rank (in uber_comm) that will receive the data from IRIS
// // uber_comm: usually MPI_COMM_WORLD
// // pp_comm: communicator for PP procs only (receivers of the data)
// // out_local_boxes: the result
// // 
// // Paired to the send in __announce_local_boxes
// void iris::recv_local_boxes(int iris_comm_size,
// 			    int rank,
// 			    int pp_master,
// 			    MPI_Comm uber_comm, 
// 			    MPI_Comm pp_comm,
// 			    iris_real *&out_local_boxes)
// {
//     int sz = 6 * iris_comm_size;
//     memory::create_1d(out_local_boxes, sz);

//     if(rank == pp_master) {
// 	MPI_Recv(out_local_boxes, sz, IRIS_REAL, MPI_ANY_SOURCE,
// 		 IRIS_EVENT_LOCAL_BOXES, uber_comm, MPI_STATUS_IGNORE);
//     }

//     MPI_Bcast(out_local_boxes, sz, IRIS_REAL, pp_master, pp_comm);
// }

// typedef void (iris::*event_handler_t)(event_t);

// void iris::__handle_event(event_t event)
// {
//     event_handler_t fun = __event_handlers[event.code];
//     if(fun) {
// 	(this->*fun)(event);
//     }else {
// 	__handle_unimplemented(event);
//     }
//     memory::wfree(event.data);
// }

// event_t iris::poke_mpi_event(MPI_Comm comm, bool &out_has_event)
// {
//     MPI_Status status;
//     int nbytes;
//     void *msg;
//     event_t retval;
//     int has_event;

//     MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &has_event, &status);
//     if(has_event) {
// 	MPI_Get_count(&status, MPI_BYTE, &nbytes);
// 	msg = memory::wmalloc(nbytes);
// 	MPI_Recv(msg, nbytes, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG,
// 		 comm, MPI_STATUS_IGNORE);
// 	retval.comm = comm;
// 	retval.peer = status.MPI_SOURCE;
// 	retval.code = status.MPI_TAG;
// 	retval.size = nbytes;
// 	retval.data = msg;
// 	out_has_event = true;
//     }else {
// 	out_has_event = false;
//     }
//     return retval;
// }

// void iris::send_event(event_t event)
// {
//     MPI_Send(event.data, event.size, MPI_BYTE, event.peer, event.code, event.comm);
// }

// event_t iris::poke_uber_event(bool &out_has_event)
// {
//     //    return poke_mpi_event(m_proc_grid->uber_comm, out_has_event);
// }

// event_t iris::poke_iris_event(bool &out_has_event)
// {
//     //    return poke_mpi_event(m_proc_grid->iris_comm, out_has_event);
// }

// event_t iris::poke_barrier_event(bool &out_has_event)
// {
//     event_t ev;

//     out_has_event = false;
//     if(__barrier_posted) {
// 	int has_event;
// 	MPI_Status status;
// 	MPI_Test(&__barrier_req, &has_event, &status);
// 	if(has_event) {
// 	    MPI_Status status;
// 	    MPI_Wait(&__barrier_req, MPI_STATUS_IGNORE);
// 	    //MPI_Barrier(m_proc_grid->iris_comm);
// 	    m_logger->trace("----------");

// 	    //ev.comm = m_proc_grid->iris_comm;
// 	    ev.peer = -1;
// 	    ev.code = IRIS_EVENT_BARRIER;
// 	    ev.size = 0;
// 	    ev.data = NULL;
// 	    out_has_event = true;
// 	    __barrier_posted = false;
// 	}
//     }
//     return ev;
// }

// event_t iris::poke_event(bool &out_has_event)
// {
//     event_t event;

//     event = poke_uber_event(out_has_event);
//     if(out_has_event) {
// 	return event;
//     }

//     event = poke_iris_event(out_has_event);
//     if(out_has_event) {
// 	return event;
//     }

//     event = poke_barrier_event(out_has_event);

//     return event;
// }

// void iris::run()
// {
// #pragma omp parallel
// #pragma omp single
//     {
// 	__quit_event_loop = false;
// 	suspend_event_loop = false;
// 	while(!__quit_event_loop) {
// 	    if(!suspend_event_loop) {
// 		bool has_event;
// 		event_t event = poke_event(has_event);
// #pragma omp task default(none) firstprivate(has_event, event)
// 		if(has_event) {
// 		    __handle_event(event);
// 		}else {
// 		    usleep(rest_time);  // suspend for some time so others can work
// 		}
// 	    }
// 	}
//     }
// }

// void iris::set_state(int in_state)
// {
//     if(in_state != IRIS_STATE_INITIALIZED) {
// 	m_logger->trace("Changing state %d -> %d", state, in_state);
//     }else {
// 	m_logger->trace("Initializing state to %d", in_state);
//     }
//     state = in_state;

//     switch(state) {

//     case IRIS_STATE_HAS_RHO:
// 	__quit_event_loop = true;
// 	break;
//     }
// }

// void iris::post_barrier()
// {
//     //MPI_Ibarrier(m_proc_grid->iris_comm, &__barrier_req);
//     __barrier_posted = true;
// }

////////////////////////////////////////////////////////////////////////////////
// Event handlers
////////////////////////////////////////////////////////////////////////////////

// void iris::__handle_unimplemented(event_t event)
// {
//     m_logger->trace("Unimplemented event: %d", event.code);
// }

// void iris::__handle_atoms(event_t event)
// {
//     if(state != IRIS_STATE_WAITING_FOR_ATOMS) {
// 	throw std::logic_error("Receiving atoms while in un-configured state!");
//     }

//     int unit = 4 * sizeof(iris_real);
//     if(event.size % unit != 0) {
// 	throw std::length_error("Unexpected message size while receiving atoms!");
//     }

//     int natoms = event.size / unit;
//     if(natoms != 0) {
// 	m_logger->trace("Received %d atoms from %d", natoms, event.peer);
// 	m_chass->assign_charges((iris_real *)event.data, natoms);
// 	m_logger->trace("Charge assignment from %d done", event.peer);
// 	MPI_Request req;
// 	MPI_Isend(NULL, 0, MPI_INT, event.peer, IRIS_EVENT_ATOMS_ACK, event.comm, &req);
//     }

// }

// void iris::__handle_atoms_eof(event_t event)
// {
//     if(state != IRIS_STATE_WAITING_FOR_ATOMS) {
// 	throw std::logic_error("Receiving atoms EOF while in un-configured state!");
//     }

//     m_logger->trace("All atoms received");
//     post_barrier();
// }

// void iris::__handle_barrier(event_t ev)
// {
//     if(state == IRIS_STATE_WAITING_FOR_ATOMS) {
// 	//m_chass->exchange_halo();

// 	// char fname[256];
// 	// sprintf(fname, "NaCl-rho-%d-%d-%d-%d", m_proc_grid->uber_size, omp_get_num_threads(), m_proc_grid->uber_rank, omp_get_thread_num());
// 	// m_mesh->dump_rho(fname);

// 	set_state(IRIS_STATE_HAS_RHO);
//     }
// }
