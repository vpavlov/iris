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
#include <set>
#include "real.h"

namespace ORG_NCSA_IRIS {

#define IRIS_STATE_INITIALIZED         0  // intial state when constructed
#define IRIS_STATE_WAITING_FOR_ATOMS   1  // waiting to receive atoms
#define IRIS_STATE_HAS_RHO             2  // right-hand side built

    class iris {

    public:

	// API: create the iris object through wich all further API calls 
	// are made
	// Example: iris *hiris = new iris(MPI_COMM_WORLD, mycomm, 0)
	iris(MPI_Comm uber_comm, MPI_Comm iris_comm, int sim_master);

	// API: release resources
	~iris();

	// API: set the number of dimentions of the domain of the problem
	// Example: hiris->domain_set_exceptions(3)
	void domain_set_dimensions(int in_dimensions);
	void domain_set_box(iris_real x0, iris_real y0, iris_real z0,
			    iris_real x1, iris_real y1, iris_real z1);

	void mesh_set_size(int nx, int ny, int nz);

	// API: set preferences about domain decomposition (e.g. 3x4x5 procs)
	void comm_set_grid_pref(int x, int y, int z);

	// API: call this after all user settings has been set in order to
	// apply the configuration and prepare for the actual calculations
	void apply_conf();

	void run();

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
	void __announce_loc_box_info();

    public:
	class domain *the_domain;  // Domain of the simulation (box, etc.)
	class comm *the_comm;      // MPI Comm related stuff
	class mesh *the_mesh;      // Computational mesh
	class debug *the_debug;    // Debug helper

	int state;  // the state of the state machine that IRIS is
	int rest_time;  // amount ot useconds to sleep while nothing to do
	bool suspend_event_loop;  // temporarily suspend the event loop

    private:

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
