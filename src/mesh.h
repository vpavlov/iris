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
#ifndef __IRIS_MESH_H__
#define __IRIS_MESH_H__

#include "global_state.h"

namespace ORG_NCSA_IRIS {

    struct rho_halo_item_t {
	int x;
	int y;
	int z;
	iris_real q;
    };

    class mesh : protected global_state {

    public:
	mesh(class iris *obj);
	~mesh();

	void set_size(int nx, int ny, int nz);
	void set_order(int order);
	void setup_local();
	void reset_rho();
	void assign_charges(iris_real *atoms, int natoms);
	void exchange_halo();

	void box_changed();  // called by the_domain when the box is changed
	void dump_rho(char *fname);  // dump right-hand-side to a BOV file
    private:
	void __compute_ca_coeff(iris_real dx, iris_real dy, iris_real dz);
	void *__pack_rho_halo(std::map<std::tuple<int, int, int>, iris_real> halo, int &nbytes);
    public:

	int order;    // charge assignment/interpolation order (from 2 to 7)

	int size[3];  // global mesh size: MxNxP mesh points in each dir
	int lsize[3];  // this proc's mesh size: M1xN1xP1 mesh points
	int loffset[3];  // lower/left/front of local mesh
	iris_real hinv[3];  // 1/h for each direction
	iris_real hinv3;    // 1/dV

	iris_real ***rho;  // values of rho (rhight-hand side) [local]
	MPI_Win rho_win;
	std::map<int, std::map<std::tuple<int, int, int>, iris_real>> outer_rho;  // rho halo (for neighbours)

    private:

	iris_real __center;  // for even orders: 0.5; odd order: 0
	iris_real __shift1;   // for even orders: 0.0; odd orders: 0.5
	iris_real **__ca_coeff;  // temporary array to hold charge assignment coefficients
	iris_real *__rho_coeff;  // pointer to one of the below
	int __left;              // -(order-1)/2
	int __right;             // order/2

	static iris_real __2order[2][2];
	static iris_real __3order[3][3];
	static iris_real __4order[4][4];
	static iris_real __5order[5][5];
	static iris_real __6order[6][6];
	static iris_real __7order[7][7];

    };
}
#endif
