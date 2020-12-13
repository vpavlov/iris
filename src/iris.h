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
#include "units.h"
#include "solver_param.h"

namespace ORG_NCSA_IRIS {

    // Role of this IRIS node.
    // Internal note: coded in binary to underline the fact that it is a bitmask
#define IRIS_ROLE_CLIENT 0b01
#define IRIS_ROLE_SERVER 0b10
    
#define IRIS_SOLVER_P3M  0x01
#define IRIS_SOLVER_CG   0x02
    
    static const iris_real _4PI = 12.566370614359172;

    // type of function called to set pieces of the right-hand side
    typedef iris_real (*rhs_fn_t)(class iris *obj, int i, int j, int k);

    class iris {

    public:

	// IRIS works in two modes:
	//   - shared mode: every node is both client and server
	//   - distributed mode: every node is either client XOR server

	// Use this constructor when for shared mode and rank 0 is the leader
	iris(int in_which_solver, MPI_Comm in_uber_comm);

	// Use this constructor when for shared mode and you want to specify
	// a different leader
	iris(int in_which_solver, MPI_Comm in_uber_comm, int in_leader);

	// Use this constructor when in distributed mode and the local leader
	// of each group is its respective rank 0
	iris(int in_which_solver, int in_client_size, int in_server_size,
	     int in_role, MPI_Comm in_local_comm,
	     MPI_Comm in_uber_comm, int in_remote_leader);

	// Use this constructor when in distributed  mode and you want to
	// specify a local leader != 0
	iris(int in_which_solver, int in_client_size, int in_server_size,
	     int in_role, MPI_Comm in_local_comm, int in_local_leader,
	     MPI_Comm in_uber_comm, int in_remote_leader);

	~iris();

	bool is_client() { return m_role & IRIS_ROLE_CLIENT; };
	bool is_server() { return m_role & IRIS_ROLE_SERVER; };
	bool is_both() { return is_client() && is_server(); };
	bool is_leader();

	// Pass the sum of all charges squared and the real-space cutoff.
	// This is needed in order to perform automatic calculation of free
	// parameters.
	void config_auto_tune(int in_natoms, iris_real in_qtot2,
			      iris_real in_cutoff);

	// Set desired accuracy (absolute or relative accuracy in the selected
	// units of force). Relative accuracy is relative in regards to the
	// force between two elementary charges 1 angstrom apart.
	void set_accuracy(iris_real in_accuracy, bool is_relative);

	// Sets charge assignment/force interpolation order
	void set_order(int order);

	void set_solver_param(int in_idx, solver_param_t in_value);
	solver_param_t get_solver_param(int in_idx)
	{
	    if (in_idx < IRIS_SOLVER_PARAM_CNT) {
		return m_solver_params[in_idx];
	    }else {
		throw std::invalid_argument("Invalid solver param!");
	    }
	};


	void set_solver(int in_solver);

	// client-only call
	void set_global_box(box_t<iris_real> *in_box);
	
	// Sets or resets the mesh size (how many points in each direction
	// the discretization grid will have). Has no effect if called from a
	// purely client node.
	void set_mesh_size(int nx, int ny, int nz);


	// Sets or resets the Ewald splitting parameter (1/distance)
	void set_alpha(iris_real in_alpha);

	// Set the poisson solver to be used
	void set_poisson_solver(int in_solver);

	void set_laplacian(int in_style, int in_arg1, int in_arg2);

	void set_units(EUnits in_units);

	// Set the right hand side directly (skips all charge assignment and
	// whatnot, usful for testing)
	void set_rhs(rhs_fn_t fn);

	void set_compute_global_energy(bool in_flag) { m_compute_global_energy = in_flag; };
	void set_compute_global_virial(bool in_flag) { m_compute_global_virial = in_flag; };


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

	// Use this on a client node to send charges to a server node
	void send_charges(int in_peer, iris_real *in_charges, int in_count);

	void commit_charges();

	// Use this on a client node to receive forces from server nodes
	iris_real *receive_forces(int **out_count, iris_real *out_Ek, iris_real *out_virial);
	void get_global_energy(iris_real *out_Ek, iris_real *out_Es, iris_real *out_Ecorr);

	void solve();

	void quit();

	// Helpers used in internode communication
	MPI_Comm server_comm();
	MPI_Comm client_comm();
	int *stos_fence_pending(MPI_Win *out_win);
	void stos_process_pending(int *in_pending, MPI_Win in_win);
	void send_event(MPI_Comm in_comm, int in_peer, int in_tag,
			int in_size, void *in_data, MPI_Request *req,
			MPI_Win in_pending_win);

	iris_real kspace_error(iris_real h, iris_real L, iris_real alpha);

	iris_real alpha() { return m_alpha; };

	int num_local_atoms();
	
    private:
	void init(MPI_Comm in_local_comm, MPI_Comm in_uber_comm);
	void process_event(struct event_t *in_event);

	void perform_commit();
	
	bool handle_charges(struct event_t *in_event);
	bool handle_commit_charges();
	bool handle_rho_halo(struct event_t *in_event);
	bool handle_get_global_energy(struct event_t *in_event);
	bool fanout_event(struct event_t *in_event);
	bool handle_set_gbox(struct event_t *in_event);
	bool handle_get_lboxes(struct event_t *in_event);
	bool handle_commit(struct event_t *in_event);
	bool handle_quit(struct event_t *in_event);

	void perform_get_global_energy(iris_real *out_Ek, iris_real *out_Es, iris_real *out_Ecorr);
	
	// initialize m_waiting_forces_from to sensible initial state
	void clear_wff();

	void calculate_etot();  // calculate Hartree energy, for verification

	void auto_tune_parameters();
	void atp_scenario1();
	void initial_alpha_estimate(iris_real *out_alpha, iris_real *out_eps);
	int  h_estimate(int dim, iris_real alpha, iris_real eps);
	bool good_factor_quality(int n);
	class solver *get_solver();

	
    public:
	int m_which_solver;            // P3M, CG, ...
	int m_order;                   // approximation order (different for different solvers)
	int m_client_size;             // # of client nodes
	int m_server_size;             // # of server nodes
	int m_role;                    // is this node client or server or both
	int m_local_leader;            // rank in local_comm of local leader
	int m_remote_leader;           // rank in uber_comm of remote leader
	int m_other_leader;            // the leader of the other group
	int m_nthreads;                // # of threads to use
	// which server peers this client is waiting to receive forces from
	bool *m_wff;

	std::map<int, int> m_ncharges;           // per sending rank
	std::map<int, iris_real *> m_charges;    // per sending rank
	std::map<int, iris_real *> m_forces;     // per recv rank
	
	bool                   m_compute_global_energy;  // whether to compute global long-range energy
	bool                   m_compute_global_virial;  // whether to compute global long-range virial
	iris_real              m_Ek;  // E(k) contribution to the global energy (from this process only)
	iris_real              m_virial[6];  // global virial, 6-element tensor

	iris_real              m_qtot2;       // Q^2 (sum of all charges squared)
	iris_real              m_cutoff;      // real-space cutoff
	int                    m_natoms;      // total number of atoms

	bool m_accuracy_free;
	bool m_alpha_free;
	bool m_order_free;
	bool m_hx_free;
	bool m_hy_free;
	bool m_hz_free;

	iris_real m_alpha;             // Ewald splitting parameter (1/distance)
	iris_real m_accuracy;          // Desired accuracy
	int       m_nx_user;           // Desired mesh size
	int       m_ny_user;
	int       m_nz_user;
	bool      m_dirty;

	solver_param_t m_solver_params[IRIS_SOLVER_PARAM_CNT];

	class comm_rec        *m_uber_comm;   // to facilitate comm with world
	class comm_rec        *m_local_comm;  // ...within group (client/server)
	class comm_rec        *m_inter_comm;  // ...between groups
	class logger          *m_logger;      // Logger
	class domain          *m_domain;      // Domain of the simulation
	class proc_grid       *m_proc_grid;   // MPI Comm related stuff
	class mesh            *m_mesh;        // Computational mesh
	class charge_assigner *m_chass;       // Charge assignmen machinery
	class solver          *m_solver;      // The Poisson equation solver itself
	class units           *m_units;       // Units system to use
    private:
	volatile bool m_quit;  // quit the main loop
    };
}
#endif
