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
#include "real.h"

namespace ORG_NCSA_IRIS {

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
	void domain_set_box(iris_real in_box_min[3], iris_real in_box_max[3]);

	// API: set preferences about domain decomposition (e.g. 3x4x5 procs)
	void comm_set_grid_pref(int x, int y, int z);

	// API: call this after all user settings has been set in order to
	// apply the configuration and prepare for the actual calculations
	void apply_conf();

	static void recv_local_boxes(MPI_Comm comm, int iris_comm_size,
				     iris_real *&out_local_boxes_min,
				     iris_real *&out_local_boxes_max);

    private:
	void __announce_loc_box_info();

    public:
	class domain *the_domain;  // Domain of the simulation (box, etc.)
	class comm *the_comm;      // MPI Comm related stuff
    };
}
#endif
