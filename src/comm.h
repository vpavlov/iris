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
#ifndef __IRIS_COMM_H__
#define __IRIS_COMM_H__

#include "global_state.h"

namespace ORG_NCSA_IRIS {

    class comm : protected global_state {

    public:
	comm(class iris *obj, MPI_Comm in_uber_comm, MPI_Comm in_iris_comm,
	     int sim_master);
	~comm();

	void set_grid_pref(int x, int y, int z);
	void setup_grid();

    private:
	static int __factorize(int n, int **factors, int **powers);
	static int **__all_grid_factorizations_of(int n, int *count);
	static void __next_fcomb(int fcnt, int *factors, int *powers,
				 int tx, int ty, int tz,
				 int **output, int *out_count);
	int __filter_factors_per_pref(int n, int **factors);
	int __filter_factors_per_mesh(int n, int **factors);
	int __select_best_factor(int n, int **factors, int *out_best);
	void __select_grid_size();
	void __setup_grid_details();
	void __setup_splits();

    public:
	// Details about the simulation communicator (e.g. MPI_COMM_WORLD)
	MPI_Comm uber_comm;
	int uber_size;
	int uber_rank;
	int sim_master;      // Rank of simulation master (in uber_comm)

	// Details about the communicator, dedicated to IRIS
	MPI_Comm iris_comm;
	int iris_size;
	int iris_rank;

	int grid_pref[3];     // User preference about procs in each dir
	int grid_size[3];     // MxNxK procs in each direction
	int grid_coords[3];   // This process' coords in the grid
	int grid_hood[27];    // Ranks of this process' neighbours (0 - me; 1 - left, 2 - right, 3 - bottom...)
	int ***grid_rank;     // = rank of the proc at [i][j][k] point in grid
	iris_real *xsplit;    // M ranges (rel 0 - 1) for each proc in X dir
	iris_real *ysplit;    // N ranges (rel 0 - 1) for each proc in Y dir
	iris_real *zsplit;    // K ranges (rel 0 - 1) for each proc in Z dir
    };
}

#endif
