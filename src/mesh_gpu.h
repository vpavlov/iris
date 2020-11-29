// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2019, the National Center for Supercomputing Applications
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
#ifndef __IRIS_GPU_MESH_H__
#define __IRIS_GPU_MESH_H__

#include <tuple>
#include "state_accessor_gpu.h"

namespace ORG_NCSA_IRIS {

    class mesh_gpu : protected state_accessor_gpu {

    public:
	mesh_gpu(class iris_gpu *obj);
	~mesh_gpu();

	void set_size(int nx, int ny, int nz);

	// commit configuration. Perform all preliminary calculations based on
	// configuration and prepare all that is needed in order to
	// start solving
	void commit();

	void assign_charges();
	void exchange_rho_halo();
	void exchange_phi_halo();
	void exchange_field_halo();
	void assign_forces(bool ad);

	void dump_bov(const char *in_fname, iris_real ***data);
	void dump_ascii(const char *in_fname, iris_real ***data);
	void dump_log(const char *in_name, iris_real ***data);

	void dump_exyz(const char *in_fname);

	void ijk_to_xyz(int i, int j, int k,
			iris_real &x, iris_real &y, iris_real &z);
	void handle_box_resize();
      
    private:


	void assign_charges1(int in_ncharges, iris_real *in_charges, iris_real *sendbuff_gpu);
	void assign_charges_gpu(iris_real* sendbuff_gpu);
	void assign_forces1(int in_ncharges, iris_real *in_charges,
			    iris_real *out_forces);
	void assign_forces1_ad(int in_ncharges, iris_real *in_charges,
			       iris_real *out_forces);

	// void send_rho_halo(int in_dim, int in_dir,
	// iris_real **out_sendbuf, MPI_Request *out_req);
	// void recv_rho_halo(int in_dim, int in_dir);
	void extract_rho();

	// void send_field_halo(int in_dim, int in_dir,
	// 		   iris_real **out_sendbuf, MPI_Request *out_req);
	// void recv_field_halo(int in_dim, int in_dir);
	void imtract_field();
	void imtract_phi();
	void assign_energy_virial_data(iris_real *forces, bool include_energy_virial);

    public:
	bool      m_dirty;  // if we need to re-calculate upon commit
	bool      m_initialized;
	int       m_size[3];  // global mesh_gpu size: MxNxP mesh_gpu points in each dir

	iris_real m_h[3];     // step of the mesh_gpu (h) in each direction
	iris_real m_h3;       // dV

	iris_real m_hinv[3];  // 1/h in each direction
	iris_real m_h3inv;    // 1/dV

	int       m_own_size[3];    // local mesh_gpu size: my portion only
	int       m_own_offset[3];  // where does my mesh_gpu start from 
	int       m_ext_size[3];    // local mesh_gpu + ass/interpol halo items

	class haloex_gpu *m_rho_haloex;
	class haloex_gpu *m_Ex_haloex;
	class haloex_gpu *m_Ey_haloex;
	class haloex_gpu *m_Ez_haloex;
	class haloex_gpu *m_phi_haloex;
	
	iris_real m_qtot;  // total charge (must be 0)
	iris_real m_q2tot; // total charge squared (must be != 0)

	std::map<int, int> m_ncharges;           // per sending rank
	std::map<int, iris_real *> m_charges;    // per sending rank
	std::map<int, iris_real *> m_forces;     // per recv rank

	// cpu buffers when cuda aware mpi is disabled
		std::map<int, iris_real *> m_charges_cpu;    // per sending rank
		std::map<int, iris_real *> m_forces_cpu;     // per recv rank

	iris_real ***m_rho;  // own charge density (ρ), part of RHS
	iris_real ***m_rho_plus;  // ρ, own + halo items

	iris_real ***m_phi;  // potential φ (unknown in the LHS)
	iris_real ***m_phi_plus;  // potential φ (unknown in the LHS)
	iris_real ***m_Ex;   // Electical field x component
	iris_real ***m_Ex_plus;  // Ex, own + halo items
	iris_real ***m_Ey;   // Electical field y component
	iris_real ***m_Ey_plus;  // Ex, own + halo items
	iris_real ***m_Ez;   // Electical field z component
	iris_real ***m_Ez_plus;  // Ex, own + halo items

    };
}
#endif
