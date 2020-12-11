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
#include <cmath>
#include "poisson_solver_p3m_gpu.h"
#include "domain_gpu.h"
#include "mesh_gpu.h"
#include "charge_assigner_gpu.h"
#include "memory.h"
#include "math_util.h"
#include "logger_gpu.h"
#include "fft3D_gpu.h"
#include "openmp.h"
#include "timer.h"
#include "remap_gpu.h"
#include "grid_gpu.h"
#include "comm_rec_gpu.h"
#include "utils.h"

#define  _PI   3.141592653589793238462643383279
#define _2PI   6.283185307179586476925286766559
#define _4PI  12.56637061435917295385057353311
#define  EPS   1.0e-7

using namespace ORG_NCSA_IRIS;

poisson_solver_p3m_gpu::poisson_solver_p3m_gpu(class iris_gpu *obj)
    : poisson_solver_gpu(obj), m_greenfn(NULL),
      m_denominator_x(NULL), m_denominator_y(NULL), m_denominator_z(NULL), 
      m_kx(NULL), m_ky(NULL), m_kz(NULL), m_vc(NULL),
      m_fft1(NULL), m_fft2(NULL),
      m_work1(NULL), m_work2(NULL), m_work3(NULL),m_work2y(NULL),m_work2z(NULL),
      m_remap(NULL), m_fft_grid(NULL), m_fft_size { 0, 0, 0 }, m_fft_offset { 0, 0, 0 }
{
};

poisson_solver_p3m_gpu::~poisson_solver_p3m_gpu()
{
    memory_gpu::destroy_1d(m_greenfn);
    memory_gpu::destroy_1d(m_denominator_x);    
    memory_gpu::destroy_1d(m_denominator_y);    
    memory_gpu::destroy_1d(m_denominator_z);
    memory_gpu::destroy_1d(m_kx);
    memory_gpu::destroy_1d(m_ky);
    memory_gpu::destroy_1d(m_kz);
    memory_gpu::destroy_2d(m_vc);
    memory_gpu::destroy_1d(m_work1);
    memory_gpu::destroy_1d(m_work2);
    memory_gpu::destroy_1d(m_work3);

    memory_gpu::destroy_1d(m_work2y);
    memory_gpu::destroy_1d(m_work2z);
    
    if(m_fft1) { delete m_fft1; }
    if(m_fft2) { delete m_fft2; }
    if(m_fft_grid) { delete m_fft_grid; }
    if(m_remap) { delete m_remap; }
}

void poisson_solver_p3m_gpu::handle_box_resize()
{
    calculate_green_function();
    calculate_k();
    calculate_virial_coeff();
}

void poisson_solver_p3m_gpu::commit()
{
    if(!m_dirty) {
	return;
    }

    solver_param_t p = m_iris->get_solver_param(IRIS_SOLVER_P3M_USE_COLLECTIVE);
    bool use_collective = (p.i == 1)?true:false;
    
    if(m_fft_grid) { delete m_fft_grid; }

    m_fft_grid = new grid_gpu(m_iris, "P3M FFT GRID");
    if (m_mesh->m_size[0] > m_local_comm->m_size) {
	m_fft_grid->set_pref(0, 1, 1);  // e.g. grid will be 64x1x1, mesh will be 2x128x128
    }else if (m_mesh->m_size[0]*m_mesh->m_size[1] > m_local_comm->m_size) {
	m_fft_grid->set_pref(0, 0, 1);  // e.g. grid will be 64x2x1, mesh will be 2x64x128
    }

    m_fft_grid->commit();

    m_fft_size[0] = m_mesh->m_size[0] / m_fft_grid->m_size[0];
    m_fft_size[1] = m_mesh->m_size[1] / m_fft_grid->m_size[1];
    m_fft_size[2] = m_mesh->m_size[2] / m_fft_grid->m_size[2];

    int *c = m_fft_grid->m_coords;
    m_fft_offset[0] = c[0] * m_fft_size[0];
    m_fft_offset[1] = c[1] * m_fft_size[1];
    m_fft_offset[2] = c[2] * m_fft_size[2];

    if(m_remap) { delete m_remap; }

    m_remap = new remap_gpu(m_iris,
			m_mesh->m_own_offset,
			m_mesh->m_own_size,
			m_fft_offset,
			m_fft_size,
			1,
			0, "initial_remap",
			use_collective);

    memory_gpu::destroy_1d(m_greenfn);
    memory_gpu::create_1d(m_greenfn, m_fft_size[0] * m_fft_size[1] * m_fft_size[2]);

    memory_gpu::destroy_1d(m_kx);
    memory_gpu::create_1d(m_kx, m_fft_size[0]);

    memory_gpu::destroy_1d(m_ky);
    memory_gpu::create_1d(m_ky, m_fft_size[1]);

    memory_gpu::destroy_1d(m_kz);
    memory_gpu::create_1d(m_kz, m_fft_size[2]);

    memory_gpu::destroy_2d(m_vc);
    memory_gpu::create_2d(m_vc, m_fft_size[0]*m_fft_size[1]*m_fft_size[2], 6);

    if (m_denominator_x==NULL) {
	memory_gpu::create_1d(m_denominator_x, m_fft_size[0]);
	memory_gpu::create_1d(m_denominator_y, m_fft_size[1]);
	memory_gpu::create_1d(m_denominator_z, m_fft_size[2]);
	calculate_denominator();
    }
    calculate_green_function();
    calculate_k();
    calculate_virial_coeff();

    if(m_fft1 != NULL) { delete m_fft1; }
    m_fft1 = new fft3d_gpu(m_iris,
		       m_fft_offset, m_fft_size,
		       m_fft_offset, m_fft_size, "fft1", use_collective);
		       

    if(m_fft2 != NULL) { delete m_fft2; }
    m_fft2 = new fft3d_gpu(m_iris,
		       m_fft_offset, m_fft_size,
		       m_mesh->m_own_offset, m_mesh->m_own_size, "fft2", use_collective);
    
    int n = 2 * m_fft1->m_count;
    
    memory_gpu::destroy_1d(m_work1);
    memory_gpu::create_1d(m_work1, n);
    
    memory_gpu::destroy_1d(m_work2);
    memory_gpu::create_1d(m_work2, n);

    memory_gpu::destroy_1d(m_work3);
    memory_gpu::create_1d(m_work3, n);

    memory_gpu::destroy_1d(m_work2y);
    memory_gpu::create_1d(m_work2y, n);

    memory_gpu::destroy_1d(m_work2z);
    memory_gpu::create_1d(m_work2z, n);
    
    m_dirty = false;
}

// Hockney/Eastwood modified Green function corresponding to the
// charge assignment functions
//
// G(k) = _4PI/k^2 * [ sum_b(k.(k+b)/(k+b).(k+b)  *  Wn^2(k+b) * rho^2(k+b) ] / sum_b(Wn^2(k+b))^2
//
void poisson_solver_p3m_gpu::calculate_green_function()
{
    const iris_real alpha = m_iris->m_alpha;

    const iris_real xL = m_domain->m_global_box.xsize;
    const iris_real yL = m_domain->m_global_box.ysize;
    const iris_real zL = m_domain->m_global_box.zsize;

    const int xM = m_mesh->m_size[0];
    const int yM = m_mesh->m_size[1];
    const int zM = m_mesh->m_size[2];
    
    const int nbx = static_cast<int> ((alpha*xL/(_PI*xM)) * pow(-log(EPS),0.25));
    const int nby = static_cast<int> ((alpha*yL/(_PI*yM)) * pow(-log(EPS),0.25));
    const int nbz = static_cast<int> ((alpha*zL/(_PI*zM)) * pow(-log(EPS),0.25));

    if(nbx == 0 && nby == 0 && nbz == 0) {
	calculate_gf_fact();
    }else {
	calculate_gf_full();
    }
}

void poisson_solver_p3m_gpu::solve()
{
  solve_async();
  //solve_sync();
}

void poisson_solver_p3m_gpu::solve_sync()
{
    m_logger->info("Solving Poisson's Equation now");

    m_remap->perform(m_mesh->m_rho, m_work2, m_work1);
    m_fft1->compute_fw(m_work2, m_work1);

    if(m_iris->m_compute_global_energy || m_iris->m_compute_global_virial) {
	kspace_eng(m_work1);
    }

    kspace_phi(m_work1);
    // do tuk work1 e tochno
    kspace_Ex(m_work1, m_work2);
    // m_mesh->dump_ascii_from_gpu("work2",m_work2,1,1,2*m_fft1->m_count);
    // exit(777);
    m_fft2->compute_bk(m_work2, m_mesh->m_Ex);
    // m_mesh->dump_ascii_from_gpu("Ex",m_mesh->m_Ex,m_mesh->m_own_size[0],m_mesh->m_own_size[1],m_mesh->m_own_size[2]);
    // exit(777);
    kspace_Ey(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, m_mesh->m_Ey);
    // m_mesh->dump_ascii_from_gpu("Ey",m_mesh->m_Ey,m_mesh->m_own_size[0],m_mesh->m_own_size[1],m_mesh->m_own_size[2]);
    //exit(777);
    kspace_Ez(m_work1, m_work2);
    m_fft2->compute_bk(m_work2, m_mesh->m_Ez);
    // m_mesh->dump_ascii_from_gpu("Ez",m_mesh->m_Ez,m_mesh->m_own_size[0],m_mesh->m_own_size[1],m_mesh->m_own_size[2]);
    // exit(777);

    //////////////// we do not need this ////////////////////////
    // m_fft2->compute_bk(m_work1, &(m_mesh->m_phi[0][0][0])); //
    /////////////////////////////////////////////////////////////
}

void poisson_solver_p3m_gpu::solve_async()
{

    m_logger->trace("Solving Poisson's Equation now");
    m_logger->info("Solving Poisson's Equation now remap async");
    m_remap->perform(m_mesh->m_rho, m_work2, m_work1);
    m_fft1->compute_fw(m_work2, m_work1);

    if(m_iris->m_compute_global_energy || m_iris->m_compute_global_virial) {
	kspace_eng(m_work1);
    }
    printf("back fft\n");
    kspace_phi(m_work1);
    // do tuk work1 e tochno
    kspace_Ex(m_work1, m_work2);
    kspace_Ey(m_work1, m_work2y);
    kspace_Ez(m_work1, m_work2z);
    // m_mesh->dump_ascii_from_gpu("work2",m_work2,1,1,2*m_fft1->m_count);
    // exit(777);
    //m_fft2->compute_bk(m_work2, m_mesh->m_Ex);
    
    collective_fft3D_state statex,statey,statez;
    cudaError_t res = cudaStreamCreateWithFlags(&(statex.gpu_stream),cudaStreamNonBlocking);
	if (res != cudaSuccess )
	{
		printf("res %d \n",res);
		exit(555);
	}
    res = cudaStreamCreateWithFlags(&(statey.gpu_stream),cudaStreamNonBlocking);
	if (res != cudaSuccess )
	{
		printf("res %d \n",res);
		exit(555);
	}
    res = cudaStreamCreateWithFlags(&(statez.gpu_stream),cudaStreamNonBlocking);
	if (res != cudaSuccess )
	{
		printf("res %d \n",res);
		exit(555);
	}
    cudaEventCreate(&statex.fft_ready);
    cudaEventCreate(&statey.fft_ready);
    cudaEventCreate(&statez.fft_ready);

    // cudaEventCreate(&statex.init_remap_ready);
    // cudaEventCreate(&statey.init_remap_ready);
    // cudaEventCreate(&statez.init_remap_ready);
    
    m_fft2->compute_bk_remap_dir_init(0,m_work2,statex);
    m_fft2->compute_bk_remap_dir_init(0,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_init(0,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_pack(0,m_work2,statex);
    m_fft2->compute_bk_remap_dir_pack(0,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_pack(0,m_work2z,statez);
    
    m_fft2->compute_bk_remap_dir_communicate1(0,m_work2,statex);
    m_fft2->compute_bk_remap_dir_communicate1(0,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_communicate1(0,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_communicate(0,m_work2,statex);
    m_fft2->compute_bk_remap_dir_communicate(0,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_communicate(0,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_finalize1(0,m_work2,statex);
    m_fft2->compute_bk_remap_dir_finalize1(0,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_finalize1(0,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_finalize(0,m_work2,statex);
    m_fft2->compute_bk_remap_dir_finalize(0,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_finalize(0,m_work2z,statez);

    m_fft2->compute_bk_fft_dir(0,m_work2,statex);
    m_fft2->compute_bk_fft_dir(0,m_work2y,statey);
    m_fft2->compute_bk_fft_dir(0,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_init(1,m_work2,statex);
    m_fft2->compute_bk_remap_dir_init(1,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_init(1,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_pack(1,m_work2,statex);
    m_fft2->compute_bk_remap_dir_pack(1,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_pack(1,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_communicate1(1,m_work2,statex);
    m_fft2->compute_bk_remap_dir_communicate1(1,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_communicate1(1,m_work2z,statez);
    
    m_fft2->compute_bk_remap_dir_communicate(1,m_work2,statex);
    m_fft2->compute_bk_remap_dir_communicate(1,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_communicate(1,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_finalize1(1,m_work2,statex);
    m_fft2->compute_bk_remap_dir_finalize1(1,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_finalize1(1,m_work2z,statez);
    
    m_fft2->compute_bk_remap_dir_finalize(1,m_work2,statex);
    m_fft2->compute_bk_remap_dir_finalize(1,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_finalize(1,m_work2z,statez);

    m_fft2->compute_bk_fft_dir(1,m_work2,statex);
    m_fft2->compute_bk_fft_dir(1,m_work2y,statey);
    m_fft2->compute_bk_fft_dir(1,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_init(2,m_work2,statex);
    m_fft2->compute_bk_remap_dir_init(2,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_init(2,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_pack(2,m_work2,statex);
    m_fft2->compute_bk_remap_dir_pack(2,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_pack(2,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_communicate1(2,m_work2,statex);
    m_fft2->compute_bk_remap_dir_communicate1(2,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_communicate1(2,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_communicate(2,m_work2,statex);
    m_fft2->compute_bk_remap_dir_communicate(2,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_communicate(2,m_work2z,statez);
    
    m_fft2->compute_bk_remap_dir_finalize1(2,m_work2,statex);
    m_fft2->compute_bk_remap_dir_finalize1(2,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_finalize1(2,m_work2z,statez);

    m_fft2->compute_bk_remap_dir_finalize(2,m_work2,statex);
    m_fft2->compute_bk_remap_dir_finalize(2,m_work2y,statey);
    m_fft2->compute_bk_remap_dir_finalize(2,m_work2z,statez);

    m_fft2->compute_bk_fft_dir(2,m_work2,statex);
    m_fft2->compute_bk_fft_dir(2,m_work2y,statey);
    m_fft2->compute_bk_fft_dir(2,m_work2z,statez);

    m_fft2->compute_bk_finalize_init(m_work2,m_mesh->m_Ex,statex);
    m_fft2->compute_bk_finalize_init(m_work2y,m_mesh->m_Ey,statey);
    m_fft2->compute_bk_finalize_init(m_work2z,m_mesh->m_Ez,statez);

    m_fft2->compute_bk_finalize_pack(m_work2,m_mesh->m_Ex,statex);
    m_fft2->compute_bk_finalize_pack(m_work2y,m_mesh->m_Ey,statey);
    m_fft2->compute_bk_finalize_pack(m_work2z,m_mesh->m_Ez,statez);

    m_fft2->compute_bk_finalize_communicate1(m_work2,m_mesh->m_Ex,statex);
    m_fft2->compute_bk_finalize_communicate1(m_work2y,m_mesh->m_Ey,statey);
    m_fft2->compute_bk_finalize_communicate1(m_work2z,m_mesh->m_Ez,statez);

    m_fft2->compute_bk_finalize_communicate(m_work2,m_mesh->m_Ex,statex);
    m_fft2->compute_bk_finalize_communicate(m_work2y,m_mesh->m_Ey,statey);
    m_fft2->compute_bk_finalize_communicate(m_work2z,m_mesh->m_Ez,statez);

    m_fft2->compute_bk_finalize_finalize1(m_work2,m_mesh->m_Ex,statex);
    m_fft2->compute_bk_finalize_finalize1(m_work2y,m_mesh->m_Ey,statey);
    m_fft2->compute_bk_finalize_finalize1(m_work2z,m_mesh->m_Ez,statez);

    m_fft2->compute_bk_finalize_finalize(m_work2,m_mesh->m_Ex,statex);
    m_fft2->compute_bk_finalize_finalize(m_work2y,m_mesh->m_Ey,statey);
    m_fft2->compute_bk_finalize_finalize(m_work2z,m_mesh->m_Ez,statez);


    free_collective_fft3D_memory(statex);
    free_collective_fft3D_memory(statey);
    free_collective_fft3D_memory(statez);
    // m_mesh->dump_ascii_from_gpu("Ex",m_mesh->m_Ex,m_mesh->m_own_size[0],m_mesh->m_own_size[1],m_mesh->m_own_size[2]);
    // exit(777);
    
    //m_fft2->compute_bk(m_work2y, m_mesh->m_Ey);
    // m_mesh->dump_ascii_from_gpu("Ey",m_mesh->m_Ey,m_mesh->m_own_size[0],m_mesh->m_own_size[1],m_mesh->m_own_size[2]);
    //exit(777);

    //m_fft2->compute_bk(m_work2z, m_mesh->m_Ez);
    // m_mesh->dump_ascii_from_gpu("Ez",m_mesh->m_Ez,m_mesh->m_own_size[0],m_mesh->m_own_size[1],m_mesh->m_own_size[2]);
    // exit(777);

    //////////////// we do not need this ////////////////////////
    // m_fft2->compute_bk(m_work1, &(m_mesh->m_phi[0][0][0])); //
    /////////////////////////////////////////////////////////////
    printf("end solve async\n");
}
