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
#include <set>
#include <mpi.h>
#include "comm.h"
#include "iris.h"
#include "domain.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;
using namespace std;

comm::comm(iris *obj, MPI_Comm in_uber_comm, MPI_Comm in_iris_comm,
	   int in_sim_master)
    : global_state(obj)
{
    uber_comm = in_uber_comm;
    MPI_Comm_size(uber_comm, &uber_size);
    MPI_Comm_rank(uber_comm, &uber_rank);

    iris_comm = in_iris_comm;
    MPI_Comm_size(iris_comm, &iris_size);
    MPI_Comm_rank(iris_comm, &iris_rank);

    sim_master = in_sim_master;

    grid_pref[0] = grid_pref[1] = grid_pref[2] = 0;

    grid_rank = NULL;
    xsplit = ysplit = zsplit = NULL;
}

comm::~comm()
{
    memory::destroy_3d(grid_rank);
    memory::destroy_1d(xsplit);
    memory::destroy_1d(ysplit);
    memory::destroy_1d(zsplit);
}


//------------------------------------------------------------------------------
// Factorization
//------------------------------------------------------------------------------
int comm::__factorize(int n, int **factors, int **powers)
{
    if(n <= 0) {
	throw std::invalid_argument("Can only factorize positive integers!");
    }

    *factors = new int[n/2];
    *powers = new int[n/2];
    int count = 0;
    for(int p = 2; n != 1; p++) {
	while(n % p == 0) {
	    if(count == 0 || (*factors)[count-1] != p) {
		(*factors)[count] = p;
		(*powers)[count++] = 1;
	    }else {
		(*powers)[count-1]++;
	    }
	    n /= p;
	}
    }

    return count;
}

void comm::__next_fcomb(int fcnt, int *factors, int *powers,
			int tx, int ty, int tz,
			int **output, int *out_count)
{
    if(fcnt == 0) {
	output[*out_count][0] = tx;
	output[*out_count][1] = ty;
	output[*out_count][2] = tz;
	(*out_count)++;
	return;
    }

    for(int x = powers[0]; x>= 0; x--) {
	for(int i = 0; i < x; i++) {
	    tx *= factors[0];
	}

	for(int y = powers[0] - x; y >= 0; y--) {
	    for(int i = 0; i < y; i++) {
		ty *= factors[0];
	    }

	    for(int i = 0; i < powers[0]-x-y; i++) {
		tz *= factors[0];
	    }

	    __next_fcomb(fcnt-1, factors+1, powers+1, tx, ty, tz,
			 output, out_count);

	    for(int i = 0; i < powers[0]-x-y; i++) {
		tz /= factors[0];
	    }

	    for(int i = 0; i < y; i++) {
		ty /= factors[0];
	    }
	}
        for (int i = 0; i < x; i++)
        {
	    tx /= factors[0];
        }
    }
}

int **comm::__all_grid_factorizations_of(int n, int *count)
{
    int *factors;
    int *powers;
    int fcnt = __factorize(n, &factors, &powers);

    int prod = 1;
    for(int i = 0; i < fcnt; i++) {
	prod *= (powers[i]+2)*(powers[i]+1)/2;
    }

    int **retval;
    memory::create_2d(retval, prod, 3);

    *count = 0;
    __next_fcomb(fcnt, factors, powers, 1, 1, 1, retval, count);

    delete [] factors;
    delete [] powers;
    return retval;
}

int comm::__filter_factors_for_1d(int n, int **factors)
{
    for(int i=0;i<n;i++) {
	if(factors[i][2] != 1 && factors[i][1] != 1) {
	    for(int j = 0; j < 3; j++) {
		factors[i][j] = factors[n-1][j];
	    }
	    n--;
	    i--;
	}
    }
    return n;
}

int comm::__filter_factors_for_2d(int n, int **factors)
{
    for(int i=0;i<n;i++) {
	if(factors[i][2] != 1) {
	    for(int j = 0; j < 3; j++) {
		factors[i][j] = factors[n-1][j];
	    }
	    n--;
	    i--;
	}
    }
    return n;
}

int comm::__filter_factors_per_pref(int n, int **factors)
{
    for(int i=0;i<n;i++) {
	if((grid_pref[0] != 0 && factors[i][0] != grid_pref[0]) ||
	   (grid_pref[1] != 0 && factors[i][1] != grid_pref[1]) ||
	   (grid_pref[2] != 0 && factors[i][2] != grid_pref[2]))
	{
	    for(int j = 0; j < 3; j++) {
		factors[i][j] = factors[n-1][j];
	    }
	    n--;
	    i--;
	}
    }
    return n;
}

// Select best factor based on minimizing the surface area of the subdomains
int comm::__select_best_factor(int n, int **factors, int *out_best)
{
    double area[3];
    area[0] = the_domain->box_size[0] * the_domain->box_size[1];
    area[1] = the_domain->box_size[0] * the_domain->box_size[2];
    area[2] = the_domain->box_size[1] * the_domain->box_size[2];

    int index;
    double best = 2.0 * (area[0] + area[1] + area[2]);
    for(int i=0;i<n;i++) {
	double surf =
	    area[0] / factors[i][0] / factors[i][1] +
	    area[1] / factors[i][0] / factors[i][2] +
	    area[2] / factors[i][1] / factors[i][2];
	if(surf < best) {
	    best = surf;
	    out_best[0] = factors[i][0];
	    out_best[1] = factors[i][1];
	    out_best[2] = factors[i][2];
	    index = i;
	}
    }

    return index;
}

void comm::set_grid_pref(int x, int y, int z)
{
    if(x < 0 || y < 0 || z < 0) {
	throw domain_error("Invalid gred preference, negative number of processors given!");
    }

    int t = x*y*z;
    if(t && t != iris_size) {
	throw domain_error("Invalid grid preference, does not match size of IRIS communicator!");
    }

    grid_pref[0] = x;
    grid_pref[1] = y;
    grid_pref[2] = z;
}

// Figure out the processor grid size (e.g. 4x2x2 procs in X, Y, Z)
// Sets this->grid_size
void comm::__select_grid_size()
{
    int num_factors;
    int **factors = __all_grid_factorizations_of(iris_size, &num_factors);

    if(the_domain->dimensions == 2) {
	num_factors = __filter_factors_for_2d(num_factors, factors);
    }else if(the_domain->dimensions == 1) {
	num_factors = __filter_factors_for_1d(num_factors, factors);
    }

    num_factors = __filter_factors_per_pref(num_factors, factors);

    if(num_factors == 0) {
	throw domain_error("Impossible grid processor assignment!");
    }

    __select_best_factor(num_factors, factors, grid_size);

    memory::destroy_2d(factors);
}

// Once we have the size, we can setup the rest of the grid details
// this->grid_coords (e.g. rank X is at coords I, J, K)
// this->grid_rank (e.g. coords I, J, K has rank X)
// this->grid_hood (e.g. rank X has ranks Y and Z to left and right, etc.)
void comm::__setup_grid_details()
{
    MPI_Comm cart_comm;
    int periods[3];
    periods[0] = periods[1] = periods[2] = 1;
    MPI_Cart_create(iris_comm, 3, grid_size, periods, 0, &cart_comm);

    // This call fills grid_coords with the coordinates of the calling
    // process inside the grid (e.g. this proc is 3,1,0)
    MPI_Cart_get(cart_comm, 3, grid_size, periods, grid_coords);

    // Fill in process neighbourhood -- the ranks of the neighbours in each
    // direction.
    MPI_Cart_shift(cart_comm, 0, 1, &grid_hood[0][0], &grid_hood[0][1]);
    MPI_Cart_shift(cart_comm, 1, 1, &grid_hood[1][0], &grid_hood[1][1]);
    MPI_Cart_shift(cart_comm, 2, 1, &grid_hood[2][0], &grid_hood[2][1]);

    memory::destroy_3d(grid_rank);
    memory::create_3d(grid_rank, grid_size[0], grid_size[1], grid_size[2]);

    for (int i = 0; i < grid_size[0]; i++) {
	for (int j = 0; j < grid_size[1]; j++) {
	    for (int k = 0; k < grid_size[2]; k++) {
		int coords[] = {i, j, k};
		MPI_Cart_rank(cart_comm, coords, &grid_rank[i][j][k]);
	    }
	}
    }

    MPI_Comm_free(&cart_comm);
}

// Setup the range of the global box that each proc is responsible for
// Processor with grid coords I, J, K is responsible for the part of the
// global box between xsplit[I] and xsplit[I+1], ysplit[J] and ysplit[J+1] and
// zsplit[K] and zsplit[K+1]
void comm::__setup_splits()
{
    memory::destroy_1d(xsplit);
    memory::destroy_1d(ysplit);
    memory::destroy_1d(zsplit);

    memory::create_1d(xsplit, grid_size[0]+1);
    memory::create_1d(ysplit, grid_size[1]+1);
    memory::create_1d(zsplit, grid_size[2]+1);

    for(int i=0;i<grid_size[0];i++) {
	xsplit[i] = i * 1.0 / grid_size[0];
    }

    for(int i=0;i<grid_size[1];i++) {
	ysplit[i] = i * 1.0 / grid_size[1];
    }

    for(int i=0;i<grid_size[2];i++) {
	zsplit[i] = i * 1.0 / grid_size[2];
    }

    xsplit[grid_size[0]] = ysplit[grid_size[1]] = zsplit[grid_size[2]] = 1.0;
}

void comm::setup_grid()
{
    __select_grid_size();
    __setup_grid_details();
    __setup_splits();
}
