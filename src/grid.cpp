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
#include <mpi.h>
#include <string.h>
#include "iris.h"
#include "grid.h"
#include "comm_rec.h"
#include "logger.h"
#include "factorizer.h"

#include "domain.h"
#include "memory.h"
#include "mesh.h"

using namespace ORG_NCSA_IRIS;
using namespace std;

grid::grid(iris *obj, const char *in_name)
    : state_accessor(obj), m_size{0, 0, 0}, m_coords{0, 0, 0},
    m_xsplit(NULL), m_ysplit(NULL), m_zsplit(NULL), m_pref{0, 0, 0},
    m_ranks(NULL), m_dirty(true)
{
    m_name = strdup(in_name);
}

grid::~grid()
{
    free(m_name);
    memory::destroy_3d(m_ranks);
    memory::destroy_1d(m_xsplit);
    memory::destroy_1d(m_ysplit);
    memory::destroy_1d(m_zsplit);
}

void grid::set_pref(int x, int y, int z)
{
    if(x < 0 || y < 0 || z < 0) {
	throw std::domain_error("Invalid gred preference, negative number of processors given!");
    }

    int t = x*y*z;
    if(t && t != m_local_comm->m_size) {
	throw std::domain_error("Invalid grid preference, does not match size of server communicator!");
    }

    m_pref[0] = x;
    m_pref[1] = y;
    m_pref[2] = z;
    m_dirty = true;
}

void grid::commit()
{
    if(m_dirty) {
	select_grid_size();      // set m_size
	setup_grid_details();    // based on this, setup grid details
	setup_splits();
	m_dirty = false;
    }
}

// Figure out the processor grid size (e.g. 4x2x2 procs in X, Y, Z)
// This function is called for its side-effect: it sets m_size
void grid::select_grid_size()
{
    int num_factors;
    int **factors = grid_factorizations(m_local_comm->m_size, &num_factors);
    num_factors = filter_factors_mod(num_factors, factors, m_mesh->m_size);
    num_factors = filter_factors_exact(num_factors, factors, m_pref);
    if(num_factors == 0) {
	throw domain_error("Impossible grid processor assignment!");
    }
    
    select_best_factor(num_factors, factors, m_size);
    m_logger->trace("%s grid is %d x %d x %d", m_name,
		   m_size[0], m_size[1], m_size[2]);
    memory::destroy_2d(factors);
}

// Select best factor based on minimizing the surface area of the subdomains
int grid::select_best_factor(int n, int **factors, int *out_best)
{
    double area[3];
    box_t<iris_real> *gbox = &m_domain->m_global_box;
    area[0] = gbox->xsize * gbox->ysize;
    area[1] = gbox->xsize * gbox->zsize;
    area[2] = gbox->ysize * gbox->zsize;

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

// Once we have the size, we can setup the rest of the grid details
// this->m_coords (e.g. rank X is at coords I, J, K)
// this->m_ranks (e.g. coords I, J, K has rank X)
// this->m_hood (e.g. rank X has ranks Y and Z to left and right, etc.)
void grid::setup_grid_details()
{
    MPI_Comm cart_comm;
    int pbc[] = { 1, 1, 1 };
    MPI_Cart_create(m_local_comm->m_comm, 3, m_size, pbc, 0, &cart_comm);

    // This call fills m_coords with the coordinates of the calling
    // process inside the grid (e.g. this proc is 3,1,0)
    MPI_Cart_get(cart_comm, 3, m_size, pbc, m_coords);
    MPI_Cart_shift(cart_comm, 0, -1, &m_hood[0][0], &m_hood[0][1]);
    MPI_Cart_shift(cart_comm, 1, -1, &m_hood[1][0], &m_hood[1][1]);
    MPI_Cart_shift(cart_comm, 2, -1, &m_hood[2][0], &m_hood[2][1]);

    memory::destroy_3d(m_ranks);
    memory::create_3d(m_ranks, m_size[0], m_size[1], m_size[2]);

    for (int i = 0; i < m_size[0]; i++) {
	for (int j = 0; j < m_size[1]; j++) {
	    for (int k = 0; k < m_size[2]; k++) {
		int coords[] = {i, j, k};
		MPI_Cart_rank(cart_comm, coords, &m_ranks[i][j][k]);
	    }
	}
    }

    MPI_Comm_free(&cart_comm);
}

// Setup the range of the global box that each proc is responsible for
// Processor with grid coords I, J, K is responsible for the part of the
// global box between m_xsplit[I] and m_xsplit[I+1],
// m_ysplit[J] and m_ysplit[J+1] and m_zsplit[K] and m_zsplit[K+1]
void grid::setup_splits()
{
    memory::destroy_1d(m_xsplit);
    memory::destroy_1d(m_ysplit);
    memory::destroy_1d(m_zsplit);

    memory::create_1d(m_xsplit, m_size[0]+1);
    memory::create_1d(m_ysplit, m_size[1]+1);
    memory::create_1d(m_zsplit, m_size[2]+1);

    for(int i=0;i<m_size[0];i++) {
	m_xsplit[i] = i * 1.0 / m_size[0];
    }

    for(int i=0;i<m_size[1];i++) {
	m_ysplit[i] = i * 1.0 / m_size[1];
    }

    for(int i=0;i<m_size[2];i++) {
	m_zsplit[i] = i * 1.0 / m_size[2];
    }

    m_xsplit[m_size[0]] = m_ysplit[m_size[1]] = m_zsplit[m_size[2]] = 1.0;
}
