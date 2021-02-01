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
#include <stdexcept>
#include <cmath>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
#include <string.h>
#include "iris.h"
#include "logger.h"
#include "comm_rec.h"
#include "domain.h"
#include "mesh.h"
#include "charge_assigner.h"
#include "proc_grid.h"
#include "memory.h"
#include "tags.h"
#include "solver.h"
#include "poisson_solver_p3m.h"
#include "poisson_solver_cg.h"
#include "timer.h"
#include "utils.h"
#include "factorizer.h"
#include "openmp.h"
#include "fmm.h"

#ifdef IRIS_CUDA
#include "cuda_runtime_api.h"
#include "cuda.h"
#endif

using namespace ORG_NCSA_IRIS;

#define _SQRT_PI  1.772453850905516027298167483341
#define _SQRT_2PI 2.506628274631000502415765284811
#define _2PI      6.283185307179586476925286766559
#define _PI2      1.570796326794896619231321691639
#define _SQRT3_2  0.8660254037844386    // sqrt(3)/2

#define FACTOR_QUALITY_THRESHOLD 3

// OAOO: helper macro to assert that client-to-server sending routines are
//       only called from client nodes
#define ASSERT_CLIENT(routine)						  \
    if(!is_client()) {							  \
	throw std::logic_error(routine					  \
			       " may only be called from client nodes!"); \
    }

// OAOO: helper macro to assert that server-only routines are only called
// from server nodes
#define ASSERT_SERVER(routine)						  \
    if(!is_server()) {							  \
	throw std::logic_error(routine					  \
			       " may only be called from server nodes!"); \
    }

iris::iris(int in_which_solver, MPI_Comm in_uber_comm, bool in_cuda):
    m_which_solver(in_which_solver), m_role(IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER), m_local_leader(0),
    m_remote_leader(0), m_cuda(in_cuda)
{
    init(in_uber_comm, in_uber_comm);
}

iris::iris(int in_which_solver, MPI_Comm in_uber_comm, int in_leader, bool in_cuda):
    m_which_solver(in_which_solver), m_role(IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER), m_local_leader(in_leader),
    m_remote_leader(in_leader), m_cuda(in_cuda)
{
    init(in_uber_comm, in_uber_comm);
}

iris::iris(int in_which_solver, int in_client_size, int in_server_size,
	   int in_role, MPI_Comm in_local_comm,
	   MPI_Comm in_uber_comm, int in_remote_leader, bool in_cuda):
    m_which_solver(in_which_solver), m_client_size(in_client_size), m_server_size(in_server_size),
    m_role(in_role), m_local_leader(0), m_remote_leader(in_remote_leader), m_cuda(in_cuda)
     
{
    init(in_local_comm, in_uber_comm);
}

iris::iris(int in_which_solver, int in_client_size, int in_server_size,
	   int in_role, MPI_Comm in_local_comm, int in_local_leader,
	   MPI_Comm in_uber_comm, int in_remote_leader, bool in_cuda):
    m_which_solver(in_which_solver), m_client_size(in_client_size), m_server_size(in_server_size),
    m_role(in_role), m_local_leader(in_local_leader),
    m_remote_leader(in_remote_leader), m_cuda(in_cuda)
{
    init(in_local_comm, in_uber_comm);
}

void iris::init(MPI_Comm in_local_comm, MPI_Comm in_uber_comm)
{
    srand(time(NULL));

#ifndef IRIS_CUDA
    if(m_cuda == true) {
	throw std::logic_error("CUDA version required, but CUDA support not compiled in!");
    }
#endif
    
#if defined _OPENMP
#pragma omp parallel default(none)
    m_nthreads = omp_get_num_threads();
#else
    m_nthreads = 1;
#endif

    // clear solver parameters
    memset(&(m_solver_params[0]), 0,
	   IRIS_SOLVER_PARAM_CNT * sizeof(solver_param_t));
 
    // initially, all calculation parameters are un-set (thus - free)
    m_qtot2 = 0.0;
    m_cutoff = 0.0;
    m_natoms = 0;
    m_accuracy_free = true;
    m_alpha_free = true;
    m_order_free = true;
    m_hx_free = true;
    m_hy_free = true;
    m_hz_free = true;
    m_dirty = true;

    m_compute_global_energy = true;
    m_compute_global_virial = true;
    m_units = new units(real);

    m_wff = NULL;

    m_uber_comm = new comm_rec(this, in_uber_comm);
    m_local_comm = new comm_rec(this, in_local_comm);
    m_inter_comm = NULL;

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

    if(is_client()) {
	m_wff = new int[m_server_size];
	clear_wff();
    }

    m_logger = new logger(this);

    m_domain = NULL;
    m_proc_grid = NULL;
    m_mesh = NULL;
    m_chass = NULL;
    m_solver = NULL;
    
    if(is_server()) {
	m_domain = new domain(this);
	m_proc_grid = new proc_grid(this);
	if(m_which_solver == IRIS_SOLVER_P3M || m_which_solver == IRIS_SOLVER_CG) {
	    m_mesh = new mesh(this);
	    m_chass = new charge_assigner(this);
	}
    }

    m_quit = false;

    m_logger->info("Node initialized as %s %d %s",
		    is_server()?(is_client()?"client/server":"server"):"client",
		   m_local_comm->m_rank,
		   is_leader()?"(leader)":"");

    if(is_leader()) {
	MPI_Request req;
	MPI_Irecv(&m_other_leader, 1, MPI_INT, m_remote_leader, IRIS_TAG_LEADER_EXCHANGE,
		  m_uber_comm->m_comm, &req);
	MPI_Send(&m_local_comm->m_rank, 1, MPI_INT, m_remote_leader, IRIS_TAG_LEADER_EXCHANGE,
		 m_uber_comm->m_comm);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	m_logger->info("This node is a leader; other leader's local rank = %d", m_other_leader);
    }

    memory::create_1d(m_ncharges, m_client_size, true);

    // default value for P3M FFT3D remap -- use collective comm
    solver_param_t def_param;
    def_param.i = 1;
    set_solver_param(IRIS_SOLVER_P3M_USE_COLLECTIVE, def_param);

    // default value for FMM NCRIT - 64
    def_param.i = 64;
    set_solver_param(IRIS_SOLVER_FMM_NCRIT, def_param);

    // // // default value for FMM MAC (θ) - 0.5
    // def_param.r = 0.5;
    // set_solver_param(IRIS_SOLVER_FMM_MAC, def_param);

    // // default value for FMM MAC LET correction parameter
    // def_param.r = 2;
    // set_solver_param(IRIS_SOLVER_FMM_MAC_CORR, def_param);
    
    def_param.r = _SQRT3_2;
    set_solver_param(IRIS_SOLVER_FMM_MAC, def_param);

    def_param.r = 1.5;
    set_solver_param(IRIS_SOLVER_FMM_MAC_CORR, def_param);

    def_param.i = -1;
    set_solver_param(IRIS_SOLVER_FMM_DEPTH, def_param);
}

iris::~iris()
{
    if(m_wff != NULL) {
	delete [] m_wff;
    }

    if(m_chass != NULL) {
	delete m_chass;
    }

    for(auto it = m_charges.begin(); it != m_charges.end(); it++) {
#ifdef IRIS_CUDA
	if(m_cuda) {
	    memory::wfree_gpu(it->second, true); // this is pinned memory
	}else
#endif
	{
	    memory::wfree(it->second);
	}
    }
    
    for(auto it = m_forces.begin(); it != m_forces.end(); it++) {
	memory::wfree(it->second);
    }

    if(m_mesh != NULL) {
	delete m_mesh;
    }

    if(m_proc_grid != NULL) {
	delete m_proc_grid;
    }

    if(m_domain != NULL) {
	delete m_domain;
    }

    if(m_solver != NULL) {
	delete m_solver;
    }

    m_logger->trace("Shutting down node");  // before m_uber_comm
    delete m_logger;

    if(m_inter_comm != NULL) {
	delete m_inter_comm;
    }

    delete m_local_comm;
    delete m_uber_comm;

    memory::destroy_1d(m_ncharges);
}

void iris::config_auto_tune(int in_natoms, iris_real in_qtot2,
			    iris_real in_cutoff)
{

    // float buff[3] = {(float)in_natoms,in_qtot2,in_cutoff};

    // MPI_Bcast(buff, 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(is_server()) {
	m_qtot2 = fabs_fn(in_qtot2) * m_units->ecf;
	m_cutoff = fabs_fn(in_cutoff);
	m_natoms = abs(in_natoms);
	m_dirty = true;
    }
}

void iris::set_accuracy(iris_real in_accuracy, bool is_relative)
{
    if(is_server()) {
	iris_real acc = fabs_fn(in_accuracy);
	if(is_relative) {
	    acc *= ((m_units->ecf * m_units->e * m_units->e) /
		    (m_units->ang * m_units->ang));
	}
	m_accuracy = acc;
	
	m_accuracy_free = false;
	m_dirty = true;
    }
}

void iris::set_alpha(iris_real in_alpha)
{
    m_alpha = in_alpha;
    m_alpha_free = false;
    m_dirty = true;
}

void iris::set_order(int in_order)
{
    if(is_server()) {
	m_order_free = false;
	m_dirty = true;
	m_order = in_order;
	if(m_which_solver == IRIS_SOLVER_P3M || m_which_solver == IRIS_SOLVER_CG) {
	    m_chass->set_order(in_order);
	}
    }
}

void iris::set_solver_param(int in_idx, solver_param_t in_value)
{
    if(is_server()) {
	if(in_idx < IRIS_SOLVER_PARAM_CNT) {
	    m_solver_params[in_idx] = in_value;
	    if(m_solver != NULL) {
		m_solver->set_dirty(true);
	    }
	}else {
	    throw std::invalid_argument("Invalid solver parameter!");
	}
    }
}

void iris::set_mesh_size(int nx, int ny, int nz)
{
    if(is_server()) {
	if(nx > 0) {
	    m_hx_free = false;
	    m_nx_user = nx;
	    m_dirty = true;
	}
	if(ny > 0) {
	    m_hy_free = false;
	    m_ny_user = ny;
	    m_dirty = true;
	}
	if(nz > 0) {
	    m_hz_free = false;
	    m_nz_user = nz;
	    m_dirty = true;
	}
    }
}

bool iris::is_leader()
{
    return m_local_comm->m_rank == m_local_leader;
};

void iris::set_units(EUnits in_units)
{
    delete m_units;
    m_units = new units(in_units);
}

// this must be called by the client leader.
// there is no harm if other clients call it as well
// (but no servers, please!)
void iris::set_global_box(box_t<iris_real> *in_box)
{
    ASSERT_CLIENT("set_global_box");

    if(is_both()) {
	m_domain->set_global_box(in_box->xlo, in_box->ylo, in_box->zlo,
				 in_box->xhi, in_box->yhi, in_box->zhi);
	return;
    }

    if(is_leader()) {
	// the client leader sends to the server leader the global box
	MPI_Comm comm = server_comm();
	MPI_Request req = MPI_REQUEST_NULL;
	send_event(comm, m_other_leader, IRIS_TAG_SET_GBOX_FANOUT, sizeof(box_t<iris_real>), in_box, &req, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(NULL, 0, MPI_BYTE, m_other_leader, IRIS_TAG_SET_GBOX_DONE, comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(m_local_comm->m_comm);
}


void iris::set_grid_pref(int x, int y, int z)
{
    if(is_server()) {
	m_proc_grid->set_pref(x, y, z);
    }
}

void iris::set_pbc(bool x, bool y, bool z)
{
    if(is_server()) {
	m_proc_grid->set_pbc(x, y, z);
    }
}

void iris::perform_commit()
{
    if(is_server()) {
	if(m_dirty) {
	    if(m_which_solver == IRIS_SOLVER_P3M) {
		auto_tune_parameters();
	    }
	    
	    if(m_solver != NULL) {
		delete m_solver;
	    }
	    m_solver = get_solver();
	    m_dirty = false;
	}

	// Beware: order is important. Some configurations depend on other
	// being already performed
	if(m_chass != NULL) {
	    m_chass->commit();      // does not depend on anything
	}
	if(m_proc_grid != NULL) {
	    m_proc_grid->commit();  // does not depend on anything
	}
	if(m_domain != NULL) {
	    m_domain->commit();     // depends on m_proc_grid
	}

	for(auto it = m_charges.begin(); it != m_charges.end(); it++) {
#ifdef IRIS_CUDA
	    if(m_cuda) {
		memory::wfree_gpu(it->second, true); // this is pinned memory
	    }else
#endif
	    {
		memory::wfree(it->second);
	    }
	}
	
	for(auto it = m_forces.begin(); it != m_forces.end(); it++) {
	    memory::wfree(it->second);
	}

	memset(m_ncharges, 0, m_client_size*sizeof(int));
	
	m_charges.clear();
	m_forces.clear();	
	
	if(m_mesh != NULL) {
	    m_mesh->commit();       // depends on m_proc_grid and m_chass and alpha
	}
	if(m_solver != NULL) {
	    m_solver->commit();     // depends on m_mesh
	}
	m_quit = false;
    }
}

// this one is called by all clients
void iris::get_local_boxes(box_t<iris_real> *out_local_boxes)
{
    ASSERT_CLIENT("get_local_boxes");

    int size = sizeof(box_t<iris_real>) * m_server_size;

    if(is_both()) {
	memcpy(out_local_boxes, m_domain->m_local_boxes, size);
	return;
    }
 
    if(is_leader()) {
	// the client leader sends to the server leader request to get the local boxes
	MPI_Comm comm = server_comm();
	MPI_Request req = MPI_REQUEST_NULL;
	send_event(comm, m_other_leader, IRIS_TAG_GET_LBOXES, 0, NULL, &req, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(out_local_boxes, size, MPI_BYTE, m_other_leader, IRIS_TAG_GET_LBOXES_DONE, comm, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(out_local_boxes, size, MPI_BYTE, m_local_leader, m_local_comm->m_comm);
}

// this one is called by all clients
void iris::get_ext_boxes(box_t<iris_real> *out_ext_boxes)
{
    ASSERT_CLIENT("get_ext_boxes");

    int size = sizeof(box_t<iris_real>) * m_server_size;

    if(is_both()) {
	memcpy(out_ext_boxes, m_solver->get_ext_boxes(), size);
	return;
    }
 
    if(is_leader()) {
	// the client leader sends to the server leader request to get the local boxes
	MPI_Comm comm = server_comm();
	MPI_Request req = MPI_REQUEST_NULL;
	send_event(comm, m_other_leader, IRIS_TAG_GET_EBOXES, 0, NULL, &req, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(out_ext_boxes, size, MPI_BYTE, m_other_leader, IRIS_TAG_GET_EBOXES_DONE, comm, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(out_ext_boxes, size, MPI_BYTE, m_local_leader, m_local_comm->m_comm);
}

void iris::commit()
{
    ASSERT_CLIENT("commit");

    if(is_both()) {
	perform_commit();
	return;
    }

    if(is_leader()) {
	MPI_Comm comm = server_comm();
	MPI_Request req = MPI_REQUEST_NULL;
	send_event(comm, m_other_leader, IRIS_TAG_COMMIT_FANOUT, 0, NULL, &req, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(NULL, 0, MPI_BYTE, m_other_leader, IRIS_TAG_COMMIT_DONE, comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(m_local_comm->m_comm);
}

void iris::quit()
{
    ASSERT_CLIENT("quit");

    if(is_both()) {
	m_quit = true;
	return;
    }

    if(is_leader()) {
	MPI_Comm comm = server_comm();
	MPI_Request req = MPI_REQUEST_NULL;
	send_event(comm, m_other_leader, IRIS_TAG_QUIT_FANOUT, 0, NULL, &req, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(NULL, 0, MPI_BYTE, m_other_leader, IRIS_TAG_QUIT_DONE, comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(m_local_comm->m_comm);
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
	    }
	}
    }
}


void iris::process_event(event_t *event)
{
    //m_logger->trace_event(event);
    bool hodl = false;
    switch(event->tag) {

    case IRIS_TAG_SET_GBOX_FANOUT:
    case IRIS_TAG_COMMIT_FANOUT:
    case IRIS_TAG_QUIT_FANOUT:
    case IRIS_TAG_GGE_FANOUT:
	hodl = fanout_event(event);
	break;
	
    case IRIS_TAG_CHARGES:
	hodl = handle_charges(event);
	break;

    case IRIS_TAG_COMMIT_CHARGES:
	hodl = handle_commit_charges();
	break;

    case IRIS_TAG_RHO_HALO:
	hodl = handle_rho_halo(event);
	break;

    case IRIS_TAG_SET_GBOX:
	hodl = handle_set_gbox(event);
	break;

    case IRIS_TAG_GET_LBOXES:
	hodl = handle_get_lboxes(event);
	break;

    case IRIS_TAG_GET_EBOXES:
	hodl = handle_get_eboxes(event);
	break;
	
    case IRIS_TAG_COMMIT:
	hodl= handle_commit(event);
	break;

    case IRIS_TAG_QUIT:
	hodl = handle_quit(event);
	break;

    case IRIS_TAG_GGE:
	hodl = handle_get_global_energy(event);
	break;
	
    default:
	m_logger->warn("Unhandled event %d", event->tag);
	break;
    }

    if(!hodl) {
	memory::wfree(event->data);
    }
}

MPI_Comm iris::server_comm()
{
    // in shared mode we only have m_local_comm; in SOC mode we use intercomm
    return is_server()?m_local_comm->m_comm:m_inter_comm->m_comm;
}

MPI_Comm iris::client_comm()
{
    // in shared mode we only have m_local_comm; in SOC mode we use intercomm
    return is_client()?m_local_comm->m_comm:m_inter_comm->m_comm;
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

    MPI_Win_create(pending, m_server_size*sizeof(int), sizeof(int),
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
    if(is_server() && in_pending_win) {
	int one = 1;
	MPI_Put(&one, 1, MPI_INT, in_peer, m_local_comm->m_rank, 1, MPI_INT, in_pending_win);
    }
}

void iris::send_charges(int in_peer, iris_real *in_charges, int in_count)
{
    ASSERT_CLIENT("send_charges");

    MPI_Comm comm = server_comm();
    MPI_Win win = NULL;
    int *pending = stos_fence_pending(&win);

    MPI_Request req;
    req = MPI_REQUEST_NULL;
    if(in_count != 0) {
	m_wff[in_peer] = in_count;
	send_event(comm, in_peer, IRIS_TAG_CHARGES,
		   5*in_count*sizeof(iris_real),
		   in_charges, &req, win);
	if(!is_server()) {
	    MPI_Recv(NULL, 0, MPI_BYTE, in_peer, IRIS_TAG_CHARGES_ACK, comm, MPI_STATUS_IGNORE);
	}
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
    MPI_Win win = NULL;
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

bool iris::handle_charges(event_t *event)
{
    int unit = 5 * sizeof(iris_real);
    if(event->size % unit != 0) {
	throw std::length_error("Unexpected message size while receiving charges!");
    }

    int ncharges = event->size / unit;

    m_ncharges[event->peer] = ncharges;
#ifdef IRIS_CUDA
    if(m_cuda) {
	m_charges[event->peer] = (iris_real *)memory::wmalloc_gpu(ncharges * unit, false, true);  // copy to pinned memory
	memcpy(m_charges[event->peer], event->data, ncharges * unit);
    }else
#endif
    {
	m_charges[event->peer] = (iris_real *)event->data;
    }

    if(!is_client()) {
	MPI_Request req;
	MPI_Isend(NULL, 0, MPI_BYTE, event->peer, IRIS_TAG_CHARGES_ACK,
		  event->comm, &req);
	MPI_Request_free(&req);
    }

#ifdef IRIS_CUDA
    if(m_cuda) {
	return false;  // in the cuda version, we already copied it to pinned memory
    }
#endif
	
    return true;  // hold on to dear life; we need the charges for later
}

bool iris::handle_commit_charges()
{
    if(m_mesh != NULL) {
	m_mesh->assign_charges();
	m_mesh->exchange_rho_halo();
    }
    
    m_solver->solve();
    
    bool ad = false;
    if(m_which_solver == IRIS_SOLVER_P3M) {
	m_mesh->exchange_field_halo();
    }else if(m_which_solver == IRIS_SOLVER_CG) {
	m_mesh->exchange_phi_halo();
	ad = true;
    }else if(m_which_solver == IRIS_SOLVER_FMM) {
	// do nothing -- the FMM solver sends back the forces itself
    }else {
       	throw std::logic_error("Don't know how to handle forces for this solver!");
    }

    if(m_mesh != NULL) {
	m_mesh->assign_forces(ad);
    }
    return false;  // no need to hodl
}

bool iris::handle_rho_halo(event_t *event)
{
    m_logger->trace_event(event);
    return false;  // no need to hodl
}

void iris::set_rhs(rhs_fn_t fn)
{
    for(int i=0;i<m_mesh->m_own_size[0];i++) {
    	for(int j=0;j<m_mesh->m_own_size[1];j++) {
    	    for(int k=0;k<m_mesh->m_own_size[2];k++) {
    		m_mesh->m_rho[i][j][k] = fn(this, i, j, k);
    	    }
    	}
    }
    
}

void iris::clear_wff()
{
    if(!is_client()) {
	return;
    }

    for(int i=0;i<m_server_size;i++) {
	m_wff[i] = 0;
    }
}

iris_real *iris::receive_forces(int **out_counts, iris_real *out_Ek, iris_real *out_virial)
{
    timer tm, tm_get_event,tm_alloc_copy;
    tm.start();
    *out_Ek = 0.0;
    *(out_virial + 0) = 0.0;
    *(out_virial + 1) = 0.0;
    *(out_virial + 2) = 0.0;
    *(out_virial + 3) = 0.0;
    *(out_virial + 4) = 0.0;
    *(out_virial + 5) = 0.0;
    
    int unit = 4 * sizeof(iris_real);
    if(!is_client()) {
	*out_counts = NULL;
	return NULL;
    }

    size_t hwm = 0;  // high water mark (in bytes)
    iris_real *retval = NULL;

    *out_counts = new int[m_server_size];

    comm_rec *server_comm = is_server()?m_local_comm:m_inter_comm;

    for(int i=0;i<m_server_size;i++) {
	(*out_counts)[i] = 0;
	if(m_wff[i]) {
	    event_t ev;
	    tm_get_event.start();
	    server_comm->get_event(i, IRIS_TAG_FORCES, ev);
	    tm_get_event.stop();
	    if((ev.size - 7*sizeof(iris_real)) % unit != 0) {
		throw std::length_error("Unexpected message size while receiving forces!");
	    }
	    (*out_counts)[i] = (ev.size - 7*sizeof(iris_real)) / unit;
	    
	    m_logger->trace("Received %d forces from server #%d (this is not rank!)", (*out_counts)[i], i);
	    tm_alloc_copy.start();
	    retval = (iris_real *)memory::wrealloc(retval, hwm + ev.size - 7*sizeof(iris_real));
	    memcpy(((unsigned char *)retval) + hwm, (unsigned char *)ev.data + 7*sizeof(iris_real), ev.size - 7*sizeof(iris_real));

	    hwm += ev.size - 7*sizeof(iris_real);

	    *out_Ek +=         *((iris_real *)ev.data + 0);
	    *(out_virial+0) += *((iris_real *)ev.data + 1);
	    *(out_virial+1) += *((iris_real *)ev.data + 2);
	    *(out_virial+2) += *((iris_real *)ev.data + 3);
	    *(out_virial+3) += *((iris_real *)ev.data + 4);
	    *(out_virial+4) += *((iris_real *)ev.data + 5);
	    *(out_virial+5) += *((iris_real *)ev.data + 6);

	    memory::wfree(ev.data);
	    tm_alloc_copy.stop();
	}
    }

    hwm=0;
    for(int i=0;i<m_server_size;i++) {
	if(m_wff[i]) {
	    hwm += (*out_counts)[i]*unit;
	}
	
    }
    clear_wff();
    tm.stop();
    m_logger->trace("receive_forces total %f s get_enent %f s allocate and copy data %f s",tm.read_wall(),tm_get_event.read_wall(),tm_alloc_copy.read_wall());

    return retval;
}

void iris::perform_get_global_energy(iris_real *out_Ek, iris_real *out_Es, iris_real *out_Ecorr)
{
    if(m_which_solver != IRIS_SOLVER_FMM) {
	iris_real volume =
	    m_domain->m_global_box.xsize * 
	    m_domain->m_global_box.ysize *
	    m_domain->m_global_box.zsize;
	
	MPI_Allreduce(&m_Ek, out_Ek, 1, IRIS_REAL, MPI_SUM, server_comm());
	*out_Es = -m_alpha * m_mesh->m_q2tot * m_units->ecf / _SQRT_PI;
	*out_Ecorr = _PI2 * m_mesh->m_qtot * m_mesh->m_qtot * m_units->ecf / (m_alpha * m_alpha * volume);
    }else {
	MPI_Allreduce(&m_Ek, out_Ek, 1, IRIS_REAL, MPI_SUM, server_comm());
	*out_Es = 0;
	*out_Ecorr = 0;
    }
}

void iris::get_global_energy(iris_real *out_Ek, iris_real *out_Es, iris_real *out_Ecorr)
{
    ASSERT_CLIENT("get_global_energy");

    if(is_both()) {
	// clients are also servers, can do the calculations directly
	return perform_get_global_energy(out_Ek, out_Es, out_Ecorr);
    }

    iris_real tmp[3];
    if(is_leader()) {  // client leader asks server leader to do whatever needs to be done
	MPI_Comm comm = server_comm();
	MPI_Request req = MPI_REQUEST_NULL;
	send_event(comm, m_other_leader, IRIS_TAG_GGE_FANOUT, 0, NULL, &req, NULL);
	MPI_Wait(&req, MPI_STATUS_IGNORE);
	MPI_Recv(tmp, 3, IRIS_REAL, m_other_leader, IRIS_TAG_GGE_DONE, comm, MPI_STATUS_IGNORE);
    }

    // client leader broadcasts to the other clients the result
    MPI_Bcast(tmp, 3, IRIS_REAL, m_local_leader, m_local_comm->m_comm);
    *out_Ek = tmp[0];
    *out_Es = tmp[1];
    *out_Ecorr = tmp[2];
}

// We have several parameters:
//   - accuracy ε
//   - order P
//   - mesh step hx, hy, hz
//   - splitting parameter α
//
// Additionally, we need Q2 (sum of all charges squared) and rc (real-space
// cutoff)
//
// Based on some of them, we can calculate the others.
//
// Reference:
// [1] Deserno M., Hold C. "How to mesh up Ewald sums (II): An accurate error estimate for the P3M algorithm" arXiv:cond-mat/9807100v1
//
void iris::auto_tune_parameters()
{
    // Scenario 1
    // ----------
    //
    // The user supplies desired accuracy and order of interpolation. The code
    // calculates the splitting parameter and mesh step so that the accuracy
    // can be satisfied.
    //
    // Given: ε and P
    // 
    // Calculate: α, hx, hy, hz
    if(!m_accuracy_free &&
       !m_order_free &&
       m_alpha_free /* &&
		       (m_hx_free || m_hy_free || m_hz_free) */)
    {
	atp_scenario1();
    }else if(!m_alpha_free && !m_hx_free && !m_hy_free && !m_hz_free) {
	m_mesh->set_size(m_nx_user, m_ny_user, m_nz_user);
    }
}

iris_real opt_acc_fn(iris_real alpha, void *obj)
{
    iris *p = (iris *)obj;
    iris_real Q2 = p->m_qtot2;
    iris_real rc = p->m_cutoff;
    int N = p->m_natoms;
    iris_real Lx = p->m_domain->m_global_box.xsize;
    iris_real Ly = p->m_domain->m_global_box.ysize;
    iris_real Lz = p->m_domain->m_global_box.zsize;
    int nx = p->m_mesh->m_size[0];
    int ny = p->m_mesh->m_size[1];
    int nz = p->m_mesh->m_size[2];

    iris_real rerr = 2.0 * Q2 * exp_fn(-alpha*alpha * rc*rc) /
	sqrt_fn(N*rc*Lx*Ly*Lz);

    iris_real kx = p->kspace_error(Lx / nx, Lx, alpha);
    iris_real ky = p->kspace_error(Ly / ny, Ly, alpha);
    iris_real kz = p->kspace_error(Lz / nz, Lz, alpha);

    iris_real kerr = sqrt(kx * kx + ky * ky + kz * kz) / sqrt(3.0);

    p->m_accuracy = kerr;
    return rerr - kerr;
}

void iris::atp_scenario1()
{
    m_logger->trace("Auto-tuning parameters:");
    m_logger->trace("  Desired accuracy: %g", m_accuracy);
    m_logger->trace("  Desired order: %d", m_chass->m_order);
    
    if(m_qtot2 == 0.0 || m_cutoff == 0.0 || m_natoms == 0 ||
       m_accuracy == 0.0 || !m_domain->m_initialized)
    {
	const char *err = "  Cannot proceed with auto-tuning because Q2, rc, N or domain are not initialized!";
	m_logger->error(err);
	throw std::invalid_argument(err);
    }

    iris_real alpha, eps;
    initial_alpha_estimate(&alpha, &eps);
    m_logger->trace("  Initial α = %f; ε = %g", alpha, eps);
    int nx = h_estimate(0, alpha, eps);
    int ny = h_estimate(1, alpha, eps);
    int nz = h_estimate(2, alpha, eps);
    m_mesh->set_size(nx, ny, nz);
    m_alpha = root_of(opt_acc_fn, alpha, this);
    m_logger->info("  Final   α = %f; ε = %g", m_alpha, m_accuracy);
}

// Based on [1], equation (23) for the real-space contribution to the error
// 
// ε ~= 2*Q2 / sqrt(N*rc*L^3) exp(-α^2 rc^2)
//
// Solving for α (everything else is given), we have:
//
// exp(-(α*rc)^2) =   ε * sqrt(N*rc*L^3) / 2*Q2
// -(α*rc)^2 =   log(0.5 * ε * sqrt(N*rc*L^3) / Q2)
//  α = sqrt(- log(0.5 * ε * sqrt(N*rc*L^3) / Q2)) / rc
// 
// Naturally, this has real solutions for α only in case the expression
// in the last square root is positive. For it to be positive, the log
// expression must be negative. If it is not, and since N, rc and L^3 are
// fixed by the problem solved (cannot change), the only way to make this
// work is to actually *lower* the ε. This problem can only occur if the
// user sets ε to a large value, e.g. 1.0 This means that he/she has
// very very low requirements for the accuracy, and by lowering ε, we are
// actually meeting these requirements...
void iris::initial_alpha_estimate(iris_real *out_alpha, iris_real *out_eps)
{
    iris_real eps = m_accuracy;
    iris_real rc = m_cutoff;
    iris_real N = m_natoms;
    iris_real Q2 = m_qtot2;
    iris_real L3 = 
	m_domain->m_global_box.xsize * 
	m_domain->m_global_box.ysize *
	m_domain->m_global_box.zsize;
    iris_real alpha;

    do {
	alpha = sqrt_fn(-log_fn(0.5 * eps * sqrt_fn(N*rc*L3) / Q2)) / rc;
	if(!std::isnan(alpha) && alpha > 0.0) {
	    break;
	}
	eps /= 10.0;
    }while(42);

    *out_alpha = alpha;
    *out_eps = eps;
}

// Eqn (38) of [1]
static iris_real am[][7] = 
    {{2.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0 / 50.0, 5.0 / 294.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {1.0 / 588.0, 7.0 / 1440.0, 21.0 / 3872.0, 0.0, 0.0, 0.0, 0.0},
     {1.0 / 4320.0, 3.0 / 1936.0, 7601.0 / 2271360.0, 143.0 / 28800.0, 0.0, 0.0, 0.0},
     {1.0 / 23232.0, 7601.0 / 13628160.0, 143.0 / 69120.0, 517231.0 / 106536960.0, 106640677.0 / 11737571328.0, 0.0, 0.0},
     { 691.0 / 68140800.0, 13.0 / 57600.0, 47021.0 / 35512320.0, 9694607.0 / 2095994880.0, 733191589.0 / 59609088000.0, 326190917.0 / 11700633600.0, 0.0},
     { 1.0 / 345600.0, 3617.0 / 35512320.0, 745739.0 / 838397952.0, 56399353.0 / 12773376000.0, 25091609.0 / 1560084480.0, 1755948832039.0 / 36229939200000.0, 4887769399.0 / 37838389248.0}};

iris_real iris::kspace_error(iris_real h, iris_real L, iris_real alpha)
{
    int N = m_natoms;
    iris_real Q2 = m_qtot2;
    int P = (double) m_chass->m_order;
    iris_real ha = h * alpha;
    iris_real *a = am[P-1];

    iris_real s = 0.0;
    for(int m=0;m<P;m++) {
	s += a[m]*pow_fn(ha, 2*m);
    }

    return Q2 * pow_fn(ha, P) * sqrt_fn(alpha*L*_SQRT_2PI*s/N) / (L*L);
}

int iris::h_estimate(int dim, iris_real alpha, iris_real eps)
{
    bool h_free;
    iris_real L;
    int n;
    switch(dim) {
    case 0:
	h_free = m_hx_free;
	L = m_domain->m_global_box.xsize;
	n = m_nx_user;
	break;

    case 1:
	h_free = m_hy_free;
	L = m_domain->m_global_box.ysize;
	n = m_ny_user;
	break;

    case 2:
	h_free = m_hz_free;
	L = m_domain->m_global_box.zsize;
	n = m_nz_user;
	break;
    }

    if(h_free) {
	iris_real h = 1 / alpha;
	n = static_cast<int>(L / h) + 1;
	iris_real kerr = kspace_error(h, L, alpha);
	while(kerr > eps || !good_factor_quality(n)) {
	    n++;
	    h = L / n;
	    kerr = kspace_error(h, L, alpha);
	}
    }

    return n;
}

// Return true if N has at least K integer factors.
bool iris::good_factor_quality(int n)
{
//	if(n % m_server_size != 0) {
//		return false;
//	}

    int *factors;
    int *powers;
    int nfactors = factorize(n, &factors, &powers);
    int nfact_tot = 0;
    for(int i=0;i<nfactors;i++) {
	nfact_tot += powers[i];
    }
    delete factors;
    delete powers;
    if(nfact_tot >= FACTOR_QUALITY_THRESHOLD) {
	return true;
    }else {
	return false;
    }
}

solver *iris::get_solver()
{
    switch(m_which_solver) {
    case IRIS_SOLVER_P3M:
	return new poisson_solver_p3m(this);

    case IRIS_SOLVER_CG:
	return new poisson_solver_cg(this);

    case IRIS_SOLVER_FMM:
	return new fmm(this);
	
    default:
	throw new std::logic_error("Unimplemented solver!");
    }
}

// an event is received only by the server leader
// we need to distribute it to the rest of the server nodes
bool iris::fanout_event(struct event_t *event)
{
    MPI_Comm comm = server_comm();
    MPI_Request *req = new MPI_Request[m_server_size];
    for(int i=0;i<m_server_size;i++) {
	req[i] = MPI_REQUEST_NULL;
    }

    for(int i=0;i<m_server_size;i++) {
	send_event(comm, i, event->tag + 1, event->size, event->data, req + i, NULL);
    }
    
    MPI_Waitall(m_server_size, req, MPI_STATUS_IGNORE);
    delete req;
    return false;
}

bool iris::handle_set_gbox(struct event_t *event)
{
    box_t<iris_real> *box = (box_t<iris_real> *)event->data;
    m_domain->set_global_box(box->xlo, box->ylo, box->zlo,
			     box->xhi, box->yhi, box->zhi);
    if(is_leader()) {
	MPI_Send(NULL, 0, MPI_BYTE, m_other_leader, IRIS_TAG_SET_GBOX_DONE, client_comm());
    }
    return false;
}

bool iris::handle_get_lboxes(event_t *in_event)
{
    MPI_Send(m_domain->m_local_boxes, sizeof(box_t<iris_real>) * m_server_size, MPI_BYTE,
	     m_other_leader, IRIS_TAG_GET_LBOXES_DONE, client_comm());
    return false;
}

bool iris::handle_get_eboxes(event_t *in_event)
{
    MPI_Send(m_solver->get_ext_boxes(), sizeof(box_t<iris_real>) * m_server_size, MPI_BYTE,
	     m_other_leader, IRIS_TAG_GET_EBOXES_DONE, client_comm());
    return false;
}

bool iris::handle_commit(event_t *in_event)
{
    perform_commit();
    if(is_leader()) {
	MPI_Send(NULL, 0, MPI_BYTE, m_other_leader, IRIS_TAG_COMMIT_DONE, client_comm());
    }
    return false;
}

bool iris::handle_quit(event_t *in_event)
{
    m_quit = true;
    if(is_leader()) {
	MPI_Send(NULL, 0, MPI_BYTE, m_other_leader, IRIS_TAG_QUIT_DONE, client_comm());
    }
    return false;
}

bool iris::handle_get_global_energy(event_t *in_event)
{
    iris_real tmp[3];
    perform_get_global_energy(tmp, tmp+1, tmp+2);
    
    if(is_leader()) {
	MPI_Send(tmp, 3, IRIS_REAL, m_other_leader, IRIS_TAG_GGE_DONE, client_comm());
    }
    return false;
}

int iris::num_local_atoms()
{
    int retval = 0;
    for(int i=0;i<m_client_size;i++) {
	int ncharges = m_ncharges[i];
	iris_real *charges = m_charges[i];
	for(int i=0;i<ncharges;i++) {
	    if(charges[i*5+4] > 0) {
		retval++;
	    }
	}
    }
    return retval;
}

int iris::num_halo_atoms()
{
    int retval = 0;
    for(int i=0;i<m_client_size;i++) {
	int ncharges = m_ncharges[i];
	iris_real *charges = m_charges[i];
	for(int i=0;i<ncharges;i++) {
	    if(charges[i*5+4] < 0) {
		retval++;
	    }
	}
    }
    return retval;
}
