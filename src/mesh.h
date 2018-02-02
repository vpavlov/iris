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

#include <tuple>
#include "state_accessor.h"

namespace ORG_NCSA_IRIS {

    // An item of halo: x, y, z coords in the local mesh and v to contribute
    // to the rho at that point
    struct halo_item_t {
	iris_real v;
	int x;
	int y;
	int z;
    };

    class mesh : protected state_accessor {

    public:
	mesh(class iris *obj);
	~mesh();

	void set_size(int nx, int ny, int nz);

	// commit configuration. Perform all preliminary calculations based on
	// configuration and prepare all that is needed in order to
	// start solving
	void commit();

	void assign_charges(iris_real *in_charges, int ncharges);
	void exchange_rho_halo();
	void exchange_field_halo();

	void dump_bov(const char *in_fname, iris_real ***data);
	void dump_ascii(const char *in_fname, iris_real ***data);
	void dump_log(const char *in_name, iris_real ***data);

	void dump_exyz(const char *in_fname);

	void ijk_to_xyz(int i, int j, int k,
			iris_real &x, iris_real &y, iris_real &z);

	void check_exyz();

    private:
	void send_rho_halo(int in_dim, int in_dir,
			   iris_real **out_sendbuf, MPI_Request *out_req);
	void recv_rho_halo(int in_dim, int in_dir);
	void extract_rho();

    public:
	bool      m_dirty;  // if we need to re-calculate upon commit
	bool      m_initialized;
	int       m_size[3];  // global mesh size: MxNxP mesh points in each dir
	iris_real m_h[3];     // step of the mesh (h) in each direction
	iris_real m_hinv[3];  // 1/h in each direction
	iris_real m_h3inv;    // 1/dV
	int       m_own_size[3];    // local mesh size: my portion only
	int       m_own_offset[3];  // where does my mesh start from 
	int       m_ext_size[3];    // local mesh + halo items

	iris_real ***m_rho;  // own charge density (ρ), part of RHS
	iris_real ***m_rho_plus;  // ρ, own + halo items
	iris_real ***m_phi;  // potential φ (unknown in the LHS)
	iris_real ***m_Ex;   // Electical field x component
	iris_real ***m_Ey;   // Electical field y component
	iris_real ***m_Ez;   // Electical field z component

    };
}
#endif
