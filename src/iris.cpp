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
#include "poisson_solver.h"
#include "poisson_solver_p3m.h"
#include "poisson_solver_cg.h"
#include "timer.h"
#include "utils.h"
#include "factorizer.h"
#include "openmp.h"

using namespace ORG_NCSA_IRIS;

#define _SQRT_PI  1.772453850905516027298167483341
#define _SQRT_2PI 2.506628274631000502415765284811
#define _2PI      6.283185307179586476925286766559
#define _PI2      1.570796326794896619231321691639

#define FACTOR_QUALITY_THRESHOLD 3

// OAOO: helper macro to assert that client-to-server sending routines are
//       only called from client nodes
#define ASSERT_CLIENT(routine)						  \
    if(!is_client()) {							  \
	throw std::logic_error(routine					  \
			       " may only be called from client nodes!"); \
    }

iris::iris(MPI_Comm in_uber_comm)
    : m_role(IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER), m_local_leader(0),
      m_remote_leader(0)
{
    init(in_uber_comm, in_uber_comm);
}

iris::iris(MPI_Comm in_uber_comm, int in_leader)
    : m_role(IRIS_ROLE_CLIENT | IRIS_ROLE_SERVER), m_local_leader(in_leader),
      m_remote_leader(in_leader)
{
    init(in_uber_comm, in_uber_comm);
}

iris::iris(int in_client_size, int in_server_size,
	   int in_role, MPI_Comm in_local_comm,
	   MPI_Comm in_uber_comm, int in_remote_leader)
    : m_client_size(in_client_size), m_server_size(in_server_size),
      m_role(in_role), m_local_leader(0), m_remote_leader(in_remote_leader)
     
{
    init(in_local_comm, in_uber_comm);
}

iris::iris(int in_client_size, int in_server_size,
	   int in_role, MPI_Comm in_local_comm, int in_local_leader,
	   MPI_Comm in_uber_comm, int in_remote_leader)
    : m_client_size(in_client_size), m_server_size(in_server_size),
      m_role(in_role), m_local_leader(in_local_leader),
      m_remote_leader(in_remote_leader)
{
    init(in_local_comm, in_uber_comm);
}

void iris::init(MPI_Comm in_local_comm, MPI_Comm in_uber_comm)
{

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
    m_which_solver = IRIS_SOLVER_P3M;
    m_dirty = true;

    m_compute_global_energy = true;
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
	m_wff = new bool[m_server_size];
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
	m_mesh = new mesh(this);
	m_chass = new charge_assigner(this);
    }

    m_quit = false;

    m_logger->trace("Node initialized as %s %d %s",
		    is_server()?(is_client()?"client/server":"server"):"client",
		   m_local_comm->m_rank,
		   is_leader()?"(leader)":"");
}

iris::~iris()
{
    if(m_wff != NULL) {
	delete [] m_wff;
    }

    if(m_chass != NULL) {
	delete m_chass;
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
}

void iris::config_auto_tune(int in_natoms, iris_real in_qtot2,
			    iris_real in_cutoff)
{
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
	m_chass->set_order(in_order);
	m_order_free = false;
	m_dirty = true;
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

void iris::set_global_box(iris_real x0, iris_real y0, iris_real z0,
			  iris_real x1, iris_real y1, iris_real z1)
{
    if(is_server()) {
	m_domain->set_global_box(x0, y0, z0, x1, y1, z1);
    }
}

void iris::set_grid_pref(int x, int y, int z)
{
    if(is_server()) {
	m_proc_grid->set_pref(x, y, z);
    }
}

void iris::set_solver(int in_which_solver)
{
    m_which_solver = in_which_solver;
    m_dirty = true;
}

void iris::commit()
{
    if(is_server()) {
	if(m_dirty) {
	    auto_tune_parameters();

	    if(m_solver != NULL) {
		delete m_solver;
	    }
	    m_solver = get_solver();
	    m_dirty = false;
	}

	// Beware: order is important. Some configurations depend on other
	// being already performed
	m_chass->commit();      // does not depend on anything
	m_proc_grid->commit();  // does not depend on anything
	m_domain->commit();     // depends on m_proc_grid
	m_mesh->commit();       // depends on m_proc_grid and m_chass and alpha
	m_solver->commit();     // depends on m_mesh
    }
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
	    }
	}
    }
}


void iris::process_event(event_t *event)
{
    //m_logger->trace_event(event);
    bool hodl = false;
    switch(event->tag) {
    case IRIS_TAG_QUIT:
	m_logger->trace("Quit event received");
	m_quit = true;
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

    case IRIS_TAG_GET_GLOBAL_ENERGY:
	hodl = handle_get_global_energy(event);
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
    if(is_server() && in_pending_win) {
	int one = 1;
	MPI_Put(&one, 1, MPI_INT, in_peer, m_local_comm->m_rank, 1, MPI_INT,
		in_pending_win);
    }
}

void iris::send_charges(int in_peer, iris_real *in_charges, int in_count)
{
    ASSERT_CLIENT("send_charges");

    MPI_Comm comm = server_comm();
    MPI_Win win;
    int *pending = stos_fence_pending(&win);

    MPI_Request req;
    req = MPI_REQUEST_NULL;
    if(in_count != 0) {
	m_wff[in_peer] = true;
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

bool iris::handle_charges(event_t *event)
{
    int unit = 5 * sizeof(iris_real);
    if(event->size % unit != 0) {
	throw std::length_error("Unexpected message size while receiving charges!");
    }

    int ncharges = event->size / unit;
    m_logger->trace("Received %d atoms from %d", ncharges, event->peer);

    m_mesh->m_ncharges[event->peer] = ncharges;
    m_mesh->m_charges[event->peer] = (iris_real *)event->data;

    if(!is_client()) {
	MPI_Request req;
	MPI_Isend(NULL, 0, MPI_BYTE, event->peer, IRIS_TAG_CHARGES_ACK,
		  event->comm, &req);
	MPI_Request_free(&req);
    }

    return true;  // hold on to dear life; we need the charges for later
}

bool iris::handle_commit_charges()
{
    m_logger->trace("Client called 'commit_charges'. Initiating computation...");
    m_mesh->assign_charges();
    m_mesh->exchange_rho_halo();
    solve();
    m_mesh->exchange_field_halo();
    m_mesh->assign_forces();
    
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

void iris::solve()
{
    if(m_compute_global_energy) {
	m_global_energy = 0.0;
    }

    m_solver->solve();

    if(m_compute_global_energy) {
	iris_real etot;
	iris_real volume =
	    m_domain->m_global_box.xsize * 
	    m_domain->m_global_box.ysize *
	    m_domain->m_global_box.zsize;

	MPI_Allreduce(&m_global_energy, &etot, 1, IRIS_REAL, MPI_SUM, server_comm());
	m_global_energy = etot;
	m_logger->info("etot1 = %f", m_global_energy);
	m_global_energy *= 0.5 * volume;
	m_logger->info("etot2 = %f", m_global_energy);
	m_global_energy -= m_alpha * m_mesh->m_q2tot / _SQRT_PI + 
	    _PI2 * m_mesh->m_qtot * m_mesh->m_qtot / (m_alpha * m_alpha * volume);
	m_logger->info("etot3 = %f", m_global_energy);
	m_global_energy *= m_units->ecf;
	m_logger->info("etot4 = %f", m_global_energy);
    }
}

void iris::clear_wff()
{
    if(!is_client()) {
	return;
    }

    for(int i=0;i<m_server_size;i++) {
	m_wff[i] = false;
    }
}

iris_real *iris::receive_forces(int **out_counts)
{
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
	    server_comm->get_event(i, IRIS_TAG_FORCES, ev);

	    if(ev.size % unit != 0) {
		throw std::length_error("Unexpected message size while receiving forces!");
	    }
	    (*out_counts)[i] = ev.size / unit;

	    m_logger->trace("Received %d forces", (*out_counts)[i]);

	    retval = (iris_real *)memory::wrealloc(retval, hwm + ev.size);
	    memcpy(((unsigned char *)retval) + hwm, ev.data, ev.size);
	    hwm += ev.size;

	    memory::wfree(ev.data);
	}
    }
    clear_wff();

    return retval;
}

iris_real iris::global_energy()
{
    // all servers have the global energy anyway, just return it
    if(is_server()) {
	return m_global_energy;
    }

    // we are definitely a pure client; ask one of the servers for the value
    iris_real eng;
    MPI_Comm comm = server_comm();
    MPI_Request req;
    send_event(comm, 0, IRIS_TAG_GET_GLOBAL_ENERGY, 0, NULL, &req, NULL);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    MPI_Recv(&eng, 1, IRIS_REAL, 0, IRIS_TAG_GLOBAL_ENERGY, comm, MPI_STATUS_IGNORE);
    return eng;
}

bool iris::handle_get_global_energy(struct event_t *event)
{
    MPI_Send(&m_global_energy, 1, IRIS_REAL, event->peer, IRIS_TAG_GLOBAL_ENERGY, event->comm);
    return false;  // no need to hodl
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
	if(!isnan(alpha) && alpha > 0.0) {
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

poisson_solver *iris::get_solver()
{
    switch(m_which_solver) {
    case IRIS_SOLVER_P3M:
	return new poisson_solver_p3m(this);

    case IRIS_SOLVER_CG:
	return new poisson_solver_cg(this);
    }
}
