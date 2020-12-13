// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2020, the National Center for Supercomputing Applications
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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include "iris_gpu.h"
#include "fft_pencil_gpu.h"
#include "logger_gpu.h"
#include "grid_gpu.h"
#include "remap_gpu.h"
#include "mesh_gpu.h"
#include "memory.h"
#include "cuda_parameters.h"
#include "fft.cuh"

using namespace ORG_NCSA_IRIS;

fft_pencil_gpu::fft_pencil_gpu(iris_gpu *obj, const char *in_name, bool in_use_collective)
    : fft_base_gpu(obj, in_name, in_use_collective)
{
    // After forward FFT, the layout is YZX
    m_out_slow = 1;
    m_out_mid = 2;
    m_out_fast = 0;
    
    setup_remaps();
    setup_plans();
}

fft_pencil_gpu::~fft_pencil_gpu()
{
    delete m_fw_remap1;
    delete m_fw_remap2;
    
    delete m_bk_remap1;
    delete m_bk_remap2;
    
#ifdef IRIS_CUDA
    FFTW_(destroy_plan)(m_fw_plan1);
    FFTW_(destroy_plan)(m_fw_plan2);
    FFTW_(destroy_plan)(m_fw_plan3);
    FFTW_(destroy_plan)(m_bk_plan1);
    FFTW_(destroy_plan)(m_bk_plan2);
    FFTW_(destroy_plan)(m_bk_plan3);
#endif
}

void fft_pencil_gpu::setup_remaps()
{
    
    /////////////////////////////////////////
    // First forward remap: XY[Z] -> ZX[Y] //
    /////////////////////////////////////////

    
    int size1[3];
    int offset1[3];
    grid_gpu *tmp1 = new grid_gpu(m_iris, "FFT-010");
    tmp1->set_pref(0, 1, 0);
    tmp1->commit();
    size1[0] = m_mesh->m_size[0] / tmp1->m_size[0];
    size1[1] = m_mesh->m_size[1] / tmp1->m_size[1];
    size1[2] = m_mesh->m_size[2] / tmp1->m_size[2];
    int *c = tmp1->m_coords;
    offset1[0] = c[0] * size1[0];
    offset1[1] = c[1] * size1[1];
    offset1[2] = c[2] * size1[2];
    delete tmp1;

    m_fw_remap1 = new remap_gpu(m_iris,
				m_mesh->m_own_offset,
				m_mesh->m_own_size,
				offset1,
				size1,
				2,
				1, "remap-fw1",
				m_use_collective);


    //////////////////////////////////////////
    // Second forward remap: ZX[Y] -> YZ[X] //
    //////////////////////////////////////////

    
    grid_gpu *tmp2 = new grid_gpu(m_iris, "FFT-100");
    tmp2->set_pref(1, 0, 0);
    tmp2->commit();
    m_out_size[0] = m_mesh->m_size[0] / tmp2->m_size[0];
    m_out_size[1] = m_mesh->m_size[1] / tmp2->m_size[1];
    m_out_size[2] = m_mesh->m_size[2] / tmp2->m_size[2];
    c = tmp2->m_coords;
    m_out_offset[0] = c[0] * m_out_size[0];
    m_out_offset[1] = c[1] * m_out_size[1];
    m_out_offset[2] = c[2] * m_out_size[2];
    delete tmp2;

    int tmp_offset1[3];
    int tmp_size1[3];
    int tmp_offset2[3];
    int tmp_size2[3];
    
    tmp_offset1[0] = offset1[2];
    tmp_offset1[1] = offset1[0];
    tmp_offset1[2] = offset1[1];
    tmp_size1[0] = size1[2];
    tmp_size1[1] = size1[0];
    tmp_size1[2] = size1[1];
    
    tmp_offset2[0] = m_out_offset[2];
    tmp_offset2[1] = m_out_offset[0];
    tmp_offset2[2] = m_out_offset[1];
    tmp_size2[0] = m_out_size[2];
    tmp_size2[1] = m_out_size[0];
    tmp_size2[2] = m_out_size[1];
    
    m_fw_remap2 = new remap_gpu(m_iris,
				tmp_offset1,
				tmp_size1,
				tmp_offset2,
				tmp_size2,
				2,
				1, "remap-fw2",
				m_use_collective);

    
    //////////////////////////////////////////
    // First backward remap: YZ[X] -> ZX[Y] //
    //////////////////////////////////////////

    
    tmp_offset1[0] = m_out_offset[1];
    tmp_offset1[1] = m_out_offset[2];
    tmp_offset1[2] = m_out_offset[0];
    tmp_size1[0] = m_out_size[1];
    tmp_size1[1] = m_out_size[2];
    tmp_size1[2] = m_out_size[0];
    
    tmp_offset2[0] = offset1[1];
    tmp_offset2[1] = offset1[2];
    tmp_offset2[2] = offset1[0];
    tmp_size2[0] = size1[1];
    tmp_size2[1] = size1[2];
    tmp_size2[2] = size1[0];
    
    m_bk_remap1 = new remap_gpu(m_iris,
				tmp_offset1,
				tmp_size1,
				tmp_offset2,
				tmp_size2,
				2,
				2, "remap-bk1",
				m_use_collective);

    
    ///////////////////////////////////////////
    // Second backward remap: ZX[Y] -> XY[Z] //
    ///////////////////////////////////////////

    
    tmp_offset1[0] = offset1[2];
    tmp_offset1[1] = offset1[0];
    tmp_offset1[2] = offset1[1];
    tmp_size1[0] = size1[2];
    tmp_size1[1] = size1[0];
    tmp_size1[2] = size1[1];
    
    tmp_offset2[0] = m_mesh->m_own_offset[2];
    tmp_offset2[1] = m_mesh->m_own_offset[0];
    tmp_offset2[2] = m_mesh->m_own_offset[1];
    tmp_size2[0] = m_mesh->m_own_size[2];
    tmp_size2[1] = m_mesh->m_own_size[0];
    tmp_size2[2] = m_mesh->m_own_size[1];
    
    m_bk_remap2 = new remap_gpu(m_iris,
				tmp_offset1,
				tmp_size1,
				tmp_offset2,
				tmp_size2,
				2,
				2, "remap-bk2",
				m_use_collective);
}

void fft_pencil_gpu::setup_plans()
{
#ifdef IRIS_CUDA
    m_fw_plan1 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_mesh->m_own_size[2]),          // size = Z
			     m_mesh->m_own_size[0]*m_mesh->m_own_size[1], 
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_mesh->m_own_size[2],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_mesh->m_own_size[2],  // distance between arrays
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);

    m_bk_plan3 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_mesh->m_own_size[2]),          // size = Z
			     m_mesh->m_own_size[0]*m_mesh->m_own_size[1], 
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_mesh->m_own_size[2],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_mesh->m_own_size[2],  // distance between arrays
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);

    m_fw_plan2 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_mesh->m_own_size[1]),          // size = Y
			     m_mesh->m_own_size[0]*m_mesh->m_own_size[2], 
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_mesh->m_own_size[1],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_mesh->m_own_size[1],  // distance between arrays
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);

    m_bk_plan2 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_mesh->m_own_size[1]),          // size = Z
			     m_mesh->m_own_size[0]*m_mesh->m_own_size[2], 
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_mesh->m_own_size[1],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_mesh->m_own_size[1],  // distance between arrays
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);


    m_fw_plan3 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_mesh->m_own_size[0]),          // size = Y
			     m_mesh->m_own_size[1]*m_mesh->m_own_size[2], 
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_mesh->m_own_size[0],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_mesh->m_own_size[0],  // distance between arrays
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);

    m_bk_plan1 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_mesh->m_own_size[0]),          // size = Z
			     m_mesh->m_own_size[1]*m_mesh->m_own_size[2], 
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_mesh->m_own_size[0],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_mesh->m_own_size[0],  // distance between arrays
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);
#endif
}

iris_real *fft_pencil_gpu::compute_fw(iris_real ***src, iris_real *dest)
{
    int nthreads = IRIS_CUDA_SHARED_BLOCK_SIZE;
    int nblocks = get_NBlocks_X(m_count,nthreads);
    get_data_from_mesh_kernel<<<nblocks, nthreads>>>(src, m_count, dest);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    
#ifdef IRIS_CUDA
    FFTW_(execute_dft)(m_fw_plan1, (complex_t *)dest, (complex_t *)dest);
    m_fw_remap1->perform(dest, dest, m_scratch);
    FFTW_(execute_dft)(m_fw_plan2, (complex_t *)dest, (complex_t *)dest);
    m_fw_remap2->perform(dest, dest, m_scratch);
    FFTW_(execute_dft)(m_fw_plan3, (complex_t *)dest, (complex_t *)dest);
#endif
    
    return dest;
    
}

void fft_pencil_gpu::compute_bk(iris_real *src, iris_real ***dest)
{

#ifdef IRIS_CUDA
    FFTW_(execute_dft)(m_bk_plan1, (complex_t *)src, (complex_t *)src);
    m_bk_remap1->perform(src, src, m_scratch);
    FFTW_(execute_dft)(m_bk_plan2, (complex_t *)src, (complex_t *)src);
    m_bk_remap2->perform(src, src, m_scratch);
    FFTW_(execute_dft)(m_bk_plan3, (complex_t *)src, (complex_t *)src);
#endif

	
    int nthreads = IRIS_CUDA_SHARED_BLOCK_SIZE;
    int nblocks = get_NBlocks_X(m_count,nthreads);
    send_data_to_mesh_kernel<<<nblocks,nthreads>>>(src, m_count, dest);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}
