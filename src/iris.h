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

namespace ORG_NCSA_IRIS {

    // Role of this IRIS node.
    // Internal note: coded in binary to underline the fact that it is a bitmask
#define IRIS_ROLE_CLIENT 0b01
#define IRIS_ROLE_SERVER 0b10

#define IRIS_STATE_INITIALIZED         0  // intial state when constructed
#define IRIS_STATE_WAITING_FOR_ATOMS   1  // waiting to receive atoms
#define IRIS_STATE_HAS_RHO             2  // right-hand side built

    class iris {

    public:

	// Use this constructor when in dual (client+server) mode and
	// the master process is rank 0
	iris(MPI_Comm in_uber_comm);

	// Use this constructor when in dual (client+server) mode and
	// you want to specify a different master process
	iris(MPI_Comm in_uber_comm, int in_leader);

	// Use this constructor when in separation (client/server) mode
	// and the local leader of each group is its respective rank 0
	iris(int in_role, MPI_Comm in_local_comm,
	     MPI_Comm in_uber_comm, int in_remote_leader);

	// Use this constructor when in separation (client/server) mode
	// and you want to specify a local leader != 0
	iris(int in_role, MPI_Comm in_local_comm, int in_local_leader,
	     MPI_Comm in_uber_comm, int in_remote_leader);

	// API: create the iris object through wich all further API calls 
	// are made
	// Example: iris *hiris = new iris(MPI_COMM_WORLD, mycomm, 0)
	iris(MPI_Comm uber_comm, MPI_Comm iris_comm, int sim_master);

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

	// Call this when all configuration is done. When called, it will
	// signal all internal components to:
	//   - set default configuration not explicitly set by the user;
	//   - verify the configuration for any missing mandatory items;
	//   - perform any preliminary calculations necessary for the solving;
	void commit();

	// Call this to run the event loop (in a separate thread)
	void run();

	// This is not part of the API, but needs to be public, so the
	// static thread function can call it
	void *event_loop();





	void set_state(int state);  // set new FSM state

	static void send_event(struct event_t);
        void post_barrier();


	struct event_t poke_event(bool &out_has_event);
	struct event_t poke_mpi_event(MPI_Comm comm, bool &out_has_event);
	struct event_t poke_uber_event(bool &out_has_event);
	struct event_t poke_iris_event(bool &out_has_event);
	struct event_t poke_barrier_event(bool &out_has_event);

	static void recv_local_boxes(int iris_comm_size,
				     int rank,
				     int pp_master,
				     MPI_Comm uber_comm, 
				     MPI_Comm pp_comm,
				     iris_real *&out_local_boxes);

    private:
	void init(MPI_Comm in_local_comm, MPI_Comm in_uber_comm);
	void start_event_sink();
	void stop_event_sink();


	void __announce_loc_box_info();

    public:
	int m_role;                    // is this node client or server or both
	int m_local_leader;            // rank in local_comm of local leader
	int m_remote_leader;           // rank in uber_comm of remote leader

	class event_queue *m_queue;       // IRIS event queue
	class comm_rec    *m_uber_comm;   // to facilitate comm with world
	class comm_rec    *m_local_comm;  // ...within group (client OR server)
	class comm_rec    *m_inter_comm;  // ...between groups
	class logger      *m_logger;      // Logger

	class domain *m_domain;            // Domain of the simulation
	class proc_grid *m_proc_grid;      // MPI Comm related stuff
	class mesh *m_mesh;                // Computational mesh
	class charge_assigner *m_chass;    // Charge assignmen machinery

	int state;  // the state of the state machine that IRIS is
	int rest_time;  // amount ot useconds to sleep while nothing to do
	bool suspend_event_loop;  // temporarily suspend the event loop

    private:
	pthread_t m_main_thread;
	bool m_main_thread_running;




	// event handlers
	bool volatile __quit_event_loop;  // when to break the event loop

	MPI_Request __barrier_req;  // to facilitate barrier events
	bool __barrier_posted;      // has a posted barrier event

	std::map<int, void (iris::*)(event_t)> __event_handlers;
	void __handle_event(event_t event);
	void __handle_unimplemented(event_t event);
	void __handle_atoms(event_t event);
	void __handle_atoms_eof(event_t event);
	void __handle_barrier(event_t event);
    };
}
#endif
