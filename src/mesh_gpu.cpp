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
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "iris_gpu.h"
#include "mesh_gpu.h"
#include "proc_grid_gpu.h"
#include "memory.h"
#include "domain_gpu.h"
#include "logger_gpu.h"
#include "event.h"
#include "charge_assigner_gpu.h"
#include "comm_rec_gpu.h"
#include "tags.h"
#include "poisson_solver_gpu.h"
#include "openmp.h"
#include "haloex_gpu.h"
#include "timer.h"

using namespace ORG_NCSA_IRIS;

mesh_gpu::mesh_gpu(iris_gpu *obj)
    :state_accessor_gpu(obj), m_size{0, 0, 0}, m_rho(NULL), m_rho_plus(NULL),
     m_dirty(true), m_initialized(false), m_phi(NULL), m_phi_plus(NULL),
     m_Ex(NULL), m_Ey(NULL), m_Ez(NULL), m_Ex_plus(NULL), m_Ey_plus(NULL),
     m_Ez_plus(NULL), m_rho_haloex(NULL), m_Ex_haloex(NULL), m_Ey_haloex(NULL), m_Ez_haloex(NULL),
     m_phi_haloex(NULL)
{
}

mesh_gpu::~mesh_gpu()
{
    memory_gpu::destroy_3d(m_rho);
    memory_gpu::destroy_3d(m_rho_plus);
    memory_gpu::destroy_3d(m_phi);
    memory_gpu::destroy_3d(m_phi_plus);
    memory_gpu::destroy_3d(m_Ex);
    memory_gpu::destroy_3d(m_Ex_plus);
    memory_gpu::destroy_3d(m_Ey);
    memory_gpu::destroy_3d(m_Ey_plus);
    memory_gpu::destroy_3d(m_Ez);
    memory_gpu::destroy_3d(m_Ez_plus);
    
    for(auto it = m_charges.begin(); it != m_charges.end(); it++) {
	memory_gpu::wfree(it->second);
    }

    for(auto it = m_forces.begin(); it != m_forces.end(); it++) {
	memory_gpu::wfree(it->second);
    }

    if(m_rho_haloex != NULL) {
	delete m_rho_haloex;
    }

    if(m_Ex_haloex != NULL) {
	delete m_Ex_haloex;
    }

    if(m_Ey_haloex != NULL) {
	delete m_Ey_haloex;
    }

    if(m_Ez_haloex != NULL) {
	delete m_Ez_haloex;
    }

    if(m_phi_haloex != NULL) {
	delete m_phi_haloex;
    }
}

void mesh_gpu::set_size(int nx, int ny, int nz)
{
    if(nx < 2 || ny < 2 || nz < 2) {
	throw std::invalid_argument("Invalid mesh_gpu size!");
    }

    m_size[0] = nx;
    m_size[1] = ny;
    m_size[2] = nz;

    m_initialized = true;
    m_dirty = true;
    m_logger->trace("Discretization mesh_gpu is %d x %d x %d",
		    m_size[0], m_size[1], m_size[2]);
	m_logger->info("Number of mesh_gpu nodes: %d",
		m_size[0] * m_size[1] * m_size[2]);
}

void mesh_gpu::handle_box_resize()
{
    m_h[0] = m_domain->m_global_box.xsize / m_size[0];
    m_h[1] = m_domain->m_global_box.ysize / m_size[1];
    m_h[2] = m_domain->m_global_box.zsize / m_size[2];
    
    m_h3 = m_h[0] * m_h[1] * m_h[2];
    
    m_hinv[0] = m_size[0] / m_domain->m_global_box.xsize;
    m_hinv[1] = m_size[1] / m_domain->m_global_box.ysize;
    m_hinv[2] = m_size[2] / m_domain->m_global_box.zsize;
    
    m_h3inv = m_hinv[0] * m_hinv[1] * m_hinv[2];
}

void mesh_gpu::commit()
{
    if(!m_domain->m_initialized) {
	throw std::logic_error("mesh_gpu commit called, but domain_gpu is not initialized!");
    }

    if(!m_initialized) {
	throw std::logic_error("mesh_gpu commit called without size being initialized!");
    }

    for(auto it = m_charges.begin(); it != m_charges.end(); it++) {
	memory::wfree(it->second);
    }

    for(auto it = m_forces.begin(); it != m_forces.end(); it++) {
	memory::wfree(it->second);
    }

    m_ncharges.clear();
    m_charges.clear();
    m_forces.clear();	


    if(m_dirty) {
	m_h[0] = m_domain->m_global_box.xsize / m_size[0];
	m_h[1] = m_domain->m_global_box.ysize / m_size[1];
	m_h[2] = m_domain->m_global_box.zsize / m_size[2];

	m_h3 = m_h[0] * m_h[1] * m_h[2];

	m_hinv[0] = m_size[0] / m_domain->m_global_box.xsize;
	m_hinv[1] = m_size[1] / m_domain->m_global_box.ysize;
	m_hinv[2] = m_size[2] / m_domain->m_global_box.zsize;

	m_logger->trace("hinv[0] %f hinv[1] %f hinv[2] %f",m_hinv[0],m_hinv[1],m_hinv[2]);

	m_h3inv = m_hinv[0] * m_hinv[1] * m_hinv[2];

	m_own_size[0] = m_size[0] / m_proc_grid->m_size[0];
	m_own_size[1] = m_size[1] / m_proc_grid->m_size[1];
	m_own_size[2] = m_size[2] / m_proc_grid->m_size[2];
	
	int *c = m_proc_grid->m_coords;
	m_own_offset[0] = c[0] * m_own_size[0];
	m_own_offset[1] = c[1] * m_own_size[1];
	m_own_offset[2] = c[2] * m_own_size[2];
	
	memory_gpu::destroy_3d(m_rho);
	memory_gpu::create_3d(m_rho, m_own_size[0], m_own_size[1], m_own_size[2],
			  true);  // make sure ρ is cleared -- it's accumulating

	memory_gpu::destroy_3d(m_phi);
	memory_gpu::create_3d(m_phi, m_own_size[0], m_own_size[1], m_own_size[2]);

	memory_gpu::destroy_3d(m_Ex);
	memory_gpu::create_3d(m_Ex, m_own_size[0], m_own_size[1], m_own_size[2]);

	memory_gpu::destroy_3d(m_Ey);
	memory_gpu::create_3d(m_Ey, m_own_size[0], m_own_size[1], m_own_size[2]);

	memory_gpu::destroy_3d(m_Ez);
	memory_gpu::create_3d(m_Ez, m_own_size[0], m_own_size[1], m_own_size[2]);
	
	// ρ halo setup
	// extra cells in each direction: see what charge assigner requires
	// from below and from above
	int extra = m_chass->m_ics_to - m_chass->m_ics_from;

	// for odd orders, we have an additional layer at the right, since
	// charges closer to the right border should go for example
	// at (0, 1, 2) instead of (-1, 0, 1)
	if(m_chass->m_order % 2) {
	    extra++;
	}

	m_ext_size[0] = m_own_size[0] + extra;
	m_ext_size[1] = m_own_size[1] + extra;
	m_ext_size[2] = m_own_size[2] + extra;
	memory_gpu::destroy_3d(m_rho_plus);
	memory_gpu::create_3d(m_rho_plus, m_ext_size[0], m_ext_size[1],
			  m_ext_size[2],
			  true);  // make sure ρ is cleared -- it's accumulating

	memory_gpu::destroy_3d(m_phi_plus);
	memory_gpu::create_3d(m_phi_plus, m_ext_size[0], m_ext_size[1], m_ext_size[2], true);
	
	memory_gpu::destroy_3d(m_Ex_plus);
	memory_gpu::create_3d(m_Ex_plus, m_ext_size[0], m_ext_size[1],
			  m_ext_size[2],
			  true);

	memory_gpu::destroy_3d(m_Ey_plus);
	memory_gpu::create_3d(m_Ey_plus, m_ext_size[0], m_ext_size[1],
			  m_ext_size[2],
			  true);

	memory_gpu::destroy_3d(m_Ez_plus);
	memory_gpu::create_3d(m_Ez_plus, m_ext_size[0], m_ext_size[1],
			  m_ext_size[2],
			  true);


	if(m_rho_haloex != NULL) {
	    delete m_rho_haloex;
	}

	if(m_Ex_haloex != NULL) {
	    delete m_Ex_haloex;
	}
	
	if(m_Ey_haloex != NULL) {
	    delete m_Ey_haloex;
	}
	
	if(m_Ez_haloex != NULL) {
	    delete m_Ez_haloex;
	}

	if(m_phi_haloex != NULL) {
	    delete m_phi_haloex;
	}
	
	int left = -m_chass->m_ics_from;
	int right = m_chass->m_ics_to;
	if(m_chass->m_order % 2) {
	    right++;
	}

	m_rho_haloex = new haloex_gpu(m_local_comm->m_comm,
				  &(m_proc_grid->m_hood[0][0]),
				  0,
				  m_rho_plus,
				  m_ext_size,
				  left,
				  right,
				  IRIS_TAG_RHO_HALO);

	m_Ex_haloex = new haloex_gpu(m_local_comm->m_comm,
				 &(m_proc_grid->m_hood[0][0]),
				 1,
				 m_Ex_plus,
				 m_ext_size,
				 left,
				 right,
				 IRIS_TAG_EX_HALO);

	m_Ey_haloex = new haloex_gpu(m_local_comm->m_comm,
				 &(m_proc_grid->m_hood[0][0]),
				 1,
				 m_Ey_plus,
				 m_ext_size,
				 left,
				 right,
				 IRIS_TAG_EY_HALO);

	m_Ez_haloex = new haloex_gpu(m_local_comm->m_comm,
				 &(m_proc_grid->m_hood[0][0]),
				 1,
				 m_Ez_plus,
				 m_ext_size,
				 left,
				 right,
				 IRIS_TAG_EZ_HALO);

	m_phi_haloex = new haloex_gpu(m_local_comm->m_comm,
				 &(m_proc_grid->m_hood[0][0]),
				 1,
				 m_phi_plus,
				 m_ext_size,
				 left,
				 right,
				 IRIS_TAG_PHI_HALO);
	
	// other configuration that depends on ours must be reset
	if(m_solver != NULL) {
	    m_solver->set_dirty(true);
	}

	m_dirty = false;

	m_logger->trace("Local mesh_gpu is %d x %d x %d starting at [%d, %d, %d]",
			m_own_size[0], m_own_size[1], m_own_size[2],
			m_own_offset[0], m_own_offset[1], m_own_offset[2]);
	m_logger->trace("Hx = %g, Hy = %g, Hz = %g", m_h[0], m_h[1], m_h[2]);
    }
}


void mesh_gpu::assign_charges()
{
    iris_real *sendbuf_gpu;
    iris_real recvbuf[2];
	#warning "use gpu buffer manager here..."
	memory_gpu::create_1d(sendbuf_gpu,2,true);

    for(auto it = m_ncharges.begin(); it != m_ncharges.end(); it++) {
		int ncharges = it->second;
		iris_real *charges = m_charges[it->first];

		m_logger->trace("assign_charge called assign_charges_gpu");
		assign_charges1(ncharges, charges,sendbuf_gpu);
    }

    m_logger->trace("assign_charge calling MPI_Allreduce");
   m_logger->trace("%s %d",__FUNCTION__,__LINE__); MPI_Allreduce(sendbuf_gpu, recvbuf, 2, IRIS_REAL, MPI_SUM, m_iris->server_comm());
    m_logger->trace("assign_charge called MPI_Allreduce");
    m_qtot = recvbuf[0];
    m_q2tot = recvbuf[1];
	memory_gpu::wfree(sendbuf_gpu);
}

void mesh_gpu::exchange_rho_halo()
{
    //    MPI_Barrier(m_local_comm->m_comm); // do we need this ???
    m_rho_haloex->exch_full();
    extract_rho();
}


void mesh_gpu::ijk_to_xyz(int i, int j, int k,
		      iris_real &x, iris_real &y, iris_real &z)
{
    x = m_domain->m_local_box.xlo + i * m_h[0];
    y = m_domain->m_local_box.ylo + j * m_h[1];
    z = m_domain->m_local_box.zlo + k * m_h[2];
}


// Ex, Ey and Ez are calculated. Now, in order to interpolate them back
// to the original atoms, each proc will need controbutions from its
// neighbours.
void mesh_gpu::exchange_field_halo()
{
    imtract_field();
    m_Ex_haloex->exch_full();
    m_Ey_haloex->exch_full();
    m_Ez_haloex->exch_full();
}

void mesh_gpu::exchange_phi_halo()
{
    imtract_phi();
    m_phi_haloex->exch_full();
}

void mesh_gpu::assign_forces(bool ad)
{
    bool include_energy_virial = true;  // send the energy and virial to only one of the clients; to the others send 0
    MPI_Comm comm = m_iris->client_comm();

    for(auto it = m_ncharges.begin(); it != m_ncharges.end(); it++) {
	int peer = it->first;
	int ncharges = it->second;
	int size = 7*sizeof(iris_real) +             // 1 real for the E(k) energy + 6 reals for the virial
	    ncharges * 4 * sizeof(iris_real);        // 4 reals for each charge: id, Fx, Fy, Fz
	#warning "buffer manager???"
	iris_real *forces = (iris_real *)memory_gpu::wmalloc(size);

	assign_energy_virial_data(forces,include_energy_virial);
	
	m_forces[peer] = forces;
	if(ad) {
	    assign_forces1_ad(ncharges, m_charges[it->first], forces);
	}else {
	    assign_forces1(ncharges, m_charges[it->first], forces);
	}

m_logger->trace("%s %d",__FUNCTION__,__LINE__); MPI_Request req;
	m_iris->send_event(comm, peer, IRIS_TAG_FORCES, size, forces, &req, NULL);
m_logger->trace("%s %d",__FUNCTION__,__LINE__); MPI_Request_free(&req);
	include_energy_virial = false;
    }
}
