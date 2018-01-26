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
#ifndef __IRIS_IRIS_H__
#define __IRIS_IRIS_H__

#include <mpi.h>
#include <pthread.h>
#include <map>
#include "real.h"
#include "box.h"
#include "event.h"

namespace ORG_NCSA_IRIS {

    // Role of this IRIS node.
    // Internal note: coded in binary to underline the fact that it is a bitmask
#define IRIS_ROLE_CLIENT 0b01
#define IRIS_ROLE_SERVER 0b10

#define IRIS_STATE_INITIALIZED  1
#define IRIS_STATE_COMMITED     2

    // type of function called to set pieces of the right-hand side
    typedef iris_real (*rhs_fn_t)(class iris *obj, int i, int j, int k);

    class iris {

    public:

	// IRIS works in two modes:
	//   - shared mode: every node is both client and server
	//   - distributed mode: every node is either client XOR server

	// Use this constructor when for shared mode and rank 0 is the leader
	iris(MPI_Comm in_uber_comm);

	// Use this constructor when for shared mode and you want to specify
	// a different leader
	iris(MPI_Comm in_uber_comm, int in_leader);

	// Use this constructor when in distributed mode and the local leader
	// of each group is its respective rank 0
	iris(int in_client_size, int in_server_size,
	     int in_role, MPI_Comm in_local_comm,
	     MPI_Comm in_uber_comm, int in_remote_leader);

	// Use this constructor when in distributed  mode and you want to
	// specify a local leader != 0
	iris(int in_client_size, int in_server_size,
	     int in_role, MPI_Comm in_local_comm, int in_local_leader,
	     MPI_Comm in_uber_comm, int in_remote_leader);

	~iris();

	bool is_client() { return m_role & IRIS_ROLE_CLIENT; };
	bool is_server() { return m_role & IRIS_ROLE_SERVER; };
	bool is_both() { return is_client() && is_server(); };
	bool is_leader();

	// Sets or resets the simulation box extents. Has no effect if called
	// from a purely client node.
	void set_global_box(iris_real x0, iris_real y0, iris_real z0,
			    iris_real x1, iris_real y1, iris_real z1);

	// Sets or resets the mesh size (how many points in each direction
	// the discretization grid will have). Has no effect if called from a
	// purely client node.
	void set_mesh_size(int nx, int ny, int nz);

	// Sets or resets the interpolation/stencil order
	void set_order(int order);

	// Sets preferences about domain decomposition (e.g. 3x4x5 procs)
	void set_grid_pref(int x, int y, int z);

	// Sets the stencil of the calculation to a Taylor-derived approximation
	// accurate to order in_order.
	void set_taylor_stencil(int in_order);

	// Set the poisson solver to be used
	void set_poisson_solver(int in_solver);

	// Set the right hand side directly (skips all charge assignment and
	// whatnot, usful for testing)
	void set_rhs(rhs_fn_t fn);

	// Call this when all configuration is done. When called, it will
	// signal all internal components to:
	//   - set default configuration not explicitly set by the user;
	//   - verify the configuration for any missing mandatory items;
	//   - perform any preliminary calculations necessary for the solving;
	void commit();

	// Call this to run the event loop
	void run();

	// The client nodes receive an array of xlo,ylo,zlo,xhi,yhi,zhi
	// for each of the server's local boxes, in the rank order of the
	// server's local_comm.
	box_t<iris_real> *get_local_boxes();

	// Use this to broadcast charges from a client node to a server node
	void broadcast_charges(int in_peer, iris_real *in_charges, int in_count);
	void commit_charges();
	void solve();
	void quit();


	// Helpers used in internode communication
	MPI_Comm server_comm();
	int *stos_fence_pending(MPI_Win *out_win);
	void stos_process_pending(int *in_pending, MPI_Win in_win);
	void send_event(MPI_Comm in_comm, int in_peer, int in_tag,
			int in_size, void *in_data, MPI_Request *req,
			MPI_Win in_pending_win);

    private:
	void init(MPI_Comm in_local_comm, MPI_Comm in_uber_comm);
	void process_event(struct event_t *in_event);

	void handle_charges(struct event_t *in_event);
	void handle_commit_charges();
	void handle_rho_halo(struct event_t *in_event);

    public:
	int m_client_size;             // # of client nodes
	int m_server_size;             // # of server nodes
	int m_role;                    // is this node client or server or both
	int m_local_leader;            // rank in local_comm of local leader
	int m_remote_leader;           // rank in uber_comm of remote leader
	int m_state;                   // State of the solver (FSM)

	class event_queue *m_queue;       // IRIS event queue
	class comm_rec    *m_uber_comm;   // to facilitate comm with world
	class comm_rec    *m_local_comm;  // ...within group (client OR server)
	class comm_rec    *m_inter_comm;  // ...between groups
	class logger      *m_logger;      // Logger
	class domain      *m_domain;      // Domain of the simulation
	class proc_grid   *m_proc_grid;   // MPI Comm related stuff
	class mesh        *m_mesh;        // Computational mesh
	class charge_assigner *m_chass;   // Charge assignmen machinery
	class stencil        *m_stencil;  // Which stencil to use
	class poisson_solver *m_solver;   // Which solver to use

    private:
	volatile bool m_quit;  // quit the main loop
    };
}
#endif
