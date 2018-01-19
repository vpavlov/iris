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
#include "taylor_stencil.h"
#include "poisson_solver_psm.h"

using namespace ORG_NCSA_IRIS;

// OAOO: helper macro to assert that client-to-server sending routines are
//       only called from client nodes
#define ASSERT_CLIENT(routine)						  \
    if(!is_client()) {							  \
	throw std::logic_error(routine					  \
			       " may only be called from client nodes!"); \
    }

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
    m_solver = NULL;
    m_stencil = NULL;
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
    m_logger->info("Node initialized as %s %d %s",
		    is_server()?(is_client()?"client/server":"server"):"client",
		   m_local_comm->m_rank,
		   is_leader()?"(leader)":"");
}

iris::~iris()
{
    if(m_stencil != NULL) {
	delete m_stencil;
    }

    if(m_solver != NULL) {
	delete m_solver;
    }

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

    if(m_inter_comm != NULL) {
	delete m_inter_comm;
    }

    delete m_local_comm;

    m_logger->info("Shutting down node");  // before m_uber_comm
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

void iris::set_poisson_solver(int in_solver)
{
    if(m_solver != NULL) {
	delete m_solver;
    }

    switch(in_solver) {
    case IRIS_POISSON_SOLVER_PSM:
	m_solver = new poisson_solver_psm(this);
	break;

    default:
	throw std::logic_error("Unknown poisson solver selected!");
    }
}

void iris::set_taylor_stencil(int in_order)
{
    if(in_order < 2 || in_order > 12 || in_order % 2) {
	throw std::logic_error("Taylor stencil only supports even orders between 2 and 12!");
    }

    if(m_stencil != NULL) {
	delete m_stencil;
    }
    m_stencil = new taylor_stencil(this, in_order);
}

void iris::commit()
{
    if(is_server()) {
	// select defaults
	if(m_stencil == NULL) {
	    set_taylor_stencil(2);
	}

	if(m_solver == NULL) {
	    set_poisson_solver(IRIS_POISSON_SOLVER_PSM);
	}

	// Beware: order is important. Some configurations depend on other
	// being already performed
	m_chass->commit();      // does not depend on anything
	m_proc_grid->commit();  // does not depend on anything
	m_domain->commit();     // depends on m_proc_grid
	m_mesh->commit();       // depends on m_proc_grid
	m_stencil->commit();    // depends on m_mesh
	m_solver->commit();     // depends on m_stencil
    }

    m_state = IRIS_STATE_COMMITED;
}


box_t<iris_real> *iris::get_local_boxes()
{
    box_t<iris_real> *local_boxes = NULL;
    int size = sizeof(box_t<iris_real>) * m_server_size;


    // TODO: figure out the exact states in which it is permissable to call
    // get_local_boxes and place a verification here


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
	handle_charges(event);
	break;

    case IRIS_TAG_COMMIT_CHARGES:
	handle_commit_charges();
	break;

    case IRIS_TAG_RHO_HALO:
	handle_rho_halo(event);
	break;
    }
}

inline MPI_Comm iris::server_comm()
{
    // in shared mode we only have m_local_comm; in SOC mode we use intercomm
    return is_server()?m_local_comm->m_comm:m_inter_comm->m_comm;
}

int *iris::stos_fence_pending(MPI_Win *out_win)
{
    if(!is_server()) {
	return NULL;
    }
    
    int *pending = new int[m_server_size];
    for(int i=0;i<m_server_size;i++) {
	pending[i] = 0;
    }

    MPI_Win_create(pending, m_server_size, sizeof(int),
		   MPI_INFO_NULL, m_local_comm->m_comm,
		   out_win);
    MPI_Win_fence(MPI_MODE_NOPRECEDE, *out_win);
    return pending;
}

void iris::stos_process_pending(int *in_pending, MPI_Win in_pending_win)
{
    if(!is_server()) {
	return;
    }
    
    //MPI_Barrier(m_local_comm->m_comm);  // is this needed? I think not.
    MPI_Win_fence(MPI_MODE_NOSUCCEED | MPI_MODE_NOSTORE, in_pending_win);
    for(int i=0;i<m_server_size;i++) {
	if(in_pending[i] == 1) {
	    event_t ev;
	    m_local_comm->get_event(ev);
	    process_event(&ev);
	}
    }
    delete [] in_pending;
    MPI_Win_free(&in_pending_win);
}

void iris::send_event(MPI_Comm in_comm, int in_peer, int in_tag,
		      int in_size, void *in_data, MPI_Request *req,
		      MPI_Win in_pending_win)
{
    MPI_Isend(in_data, in_size, MPI_BYTE, in_peer, in_tag, in_comm, req);
    if(is_server()) {
	int one = 1;
	MPI_Put(&one, 1, MPI_INT, in_peer, m_local_comm->m_rank, 1, MPI_INT,
		in_pending_win);
    }
}

void iris::broadcast_charges(int in_peer, iris_real *in_charges, int in_count)
{
    ASSERT_CLIENT("broadcast_charges");

    MPI_Comm comm = server_comm();
    MPI_Win win;
    int *pending = stos_fence_pending(&win);

    int offset = 0;
    MPI_Request req;
    req = MPI_REQUEST_NULL;
    if(in_count != 0) {
	send_event(comm, in_peer, IRIS_TAG_CHARGES,
		   4*in_count*sizeof(iris_real),
		   in_charges, &req, win);
    }

    stos_process_pending(pending, win);

    MPI_Wait(&req, MPI_STATUS_IGNORE);
}

void iris::commit_charges()
{
    ASSERT_CLIENT("commit_charges");

    // Make sure all clients have already sent their atoms before notifying
    // the server that there are no more charges.
    MPI_Barrier(m_local_comm->m_comm);

    MPI_Comm comm = server_comm();
    MPI_Win win;
    int *pending = stos_fence_pending(&win);

    if(is_leader()) {
	for(int i=0;i<m_server_size;i++) {
	    MPI_Request req;
	    send_event(comm, i, IRIS_TAG_COMMIT_CHARGES, 0, NULL, &req, win);
	    if(req != MPI_REQUEST_NULL) {
		MPI_Request_free(&req);
	    }
	}
    }

    stos_process_pending(pending, win);
}

void iris::quit()
{
    ASSERT_CLIENT("quit");

    MPI_Comm comm = server_comm();
    MPI_Win win;
    int *pending = stos_fence_pending(&win);

    if(is_leader()) {
	for(int i=0;i<m_server_size;i++) {
	    MPI_Request req;
	    send_event(comm, i, IRIS_TAG_QUIT, 0, NULL, &req, win);
	    if(req != MPI_REQUEST_NULL) {
		MPI_Request_free(&req);
	    }
	}
    }

    stos_process_pending(pending, win);
}







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


// void iris::post_barrier()
// {
//     //MPI_Ibarrier(m_proc_grid->iris_comm, &__barrier_req);
//     __barrier_posted = true;
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

void iris::handle_charges(event_t *event)
{
    int unit = 4 * sizeof(iris_real);
    if(event->size % unit != 0) {
	throw std::length_error("Unexpected message size while receiving charges!");
    }

    int natoms = event->size / unit;
    if(natoms != 0) {
	m_logger->trace("Received %d atoms from %d: initiating charge assignment", natoms, event->peer);
	m_mesh->assign_charges((iris_real *)event->data, natoms);
	m_logger->trace("Charge assignment from %d done", event->peer);
    }
}

void iris::handle_commit_charges()
{
    m_logger->trace("Commit charges received: initiating halo exchange");
    m_mesh->exchange_halo();
    m_logger->trace("Halo exchange done");
}

void iris::handle_rho_halo(event_t *event)
{
    int unit = sizeof(halo_item_t);
    if(event->size % unit != 0) {
	throw std::length_error("Unexpected message size while receiving rho halo!");
    }

    int nitems = event->size / unit;
    if(nitems != 0) {
	m_logger->trace("Received %d halo items from %d: adding them to our ρ",
			nitems, event->peer);
	m_mesh->add_halo_items((halo_item_t *)event->data, nitems);
	m_logger->trace("Adding halo to ρ done");
    }
}
