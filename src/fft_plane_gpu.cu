// -*- c++ -*-
//==============================================================================
// IRIS - Long-range Interaction Solver Library
//
// Copyright (c) 2017-2021, the National Center for Supercomputing Applications
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
#include "fft_plane_gpu.h"
#include "logger_gpu.h"
#include "grid_gpu.h"
#include "remap_gpu.h"
#include "mesh_gpu.h"
#include "memory.h"
#include "cuda_parameters.h"
#include "fft.cuh"

using namespace ORG_NCSA_IRIS;

fft_plane_gpu::fft_plane_gpu(iris_gpu *obj, const char *in_name, bool in_use_collective)
    : fft_base_gpu(obj, in_name, in_use_collective)
{
    // after forward FFT, layout is YZX
    m_out_slow = 1;
    m_out_mid = 2;
    m_out_fast = 0;
    
    setup_remaps();
    setup_plans();
}

fft_plane_gpu::~fft_plane_gpu()
{
    delete m_fw_remap;
    delete m_bk_remap;
#ifdef IRIS_CUDA
    FFTW_(destroy_plan)(m_fw_plan1);
    FFTW_(destroy_plan)(m_fw_plan2);
    FFTW_(destroy_plan)(m_bk_plan1);
    FFTW_(destroy_plan)(m_bk_plan2);
#endif
}

void fft_plane_gpu::setup_remaps()
{
    grid_gpu *tmp = new grid_gpu(m_iris, "FFT-101");
    tmp->set_pref(1, 0, 1);
    tmp->commit();

    m_out_size[0] = m_mesh->m_size[0] / tmp->m_size[0];
    m_out_size[1] = m_mesh->m_size[1] / tmp->m_size[1];
    m_out_size[2] = m_mesh->m_size[2] / tmp->m_size[2];
    int *c = tmp->m_coords;
    m_out_offset[0] = c[0] * m_out_size[0];
    m_out_offset[1] = c[1] * m_out_size[1];
    m_out_offset[2] = c[2] * m_out_size[2];

    delete tmp;
    
    m_fw_remap = new remap_gpu(m_iris,
			m_mesh->m_own_offset,
			m_mesh->m_own_size,
			m_out_offset,
			m_out_size,
			2,
			2, "remap-fw",
			m_use_collective);


    int tmp_offset1[3];
    int tmp_size1[3];
    int tmp_offset2[3];
    int tmp_size2[3];

    tmp_offset1[0] = m_out_offset[1];
    tmp_offset1[1] = m_out_offset[2];
    tmp_offset1[2] = m_out_offset[0];
    tmp_size1[0] = m_out_size[1];
    tmp_size1[1] = m_out_size[2];
    tmp_size1[2] = m_out_size[0];
    
    tmp_offset2[0] = m_mesh->m_own_offset[1];
    tmp_offset2[1] = m_mesh->m_own_offset[2];
    tmp_offset2[2] = m_mesh->m_own_offset[0];
    tmp_size2[0] = m_mesh->m_own_size[1];
    tmp_size2[1] = m_mesh->m_own_size[2];
    tmp_size2[2] = m_mesh->m_own_size[0];
    
    m_bk_remap = new remap_gpu(m_iris,
			       tmp_offset1,
			       tmp_size1,
			       tmp_offset2,
			       tmp_size2,
			       2,
			       1, "remap-bk",
			       m_use_collective);
}

void fft_plane_gpu::setup_plans()
{
    int n[] = { m_mesh->m_own_size[1], m_mesh->m_own_size[2] };

#ifdef IRIS_CUDA
    m_fw_plan1 =
	FFTW_(plan_many_dft)(2,          // 2D FFT
			     n,          // NxP arrays
			     m_mesh->m_own_size[0],    // M arrays
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     n[0]*n[1],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     n[0]*n[1],  // distance between arrays
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);

    m_bk_plan2 =
	FFTW_(plan_many_dft)(2,          // 2D FFT
			     n,          // NxP size
			     m_mesh->m_own_size[0],  // M arrays
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     n[0]*n[1],  // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     n[0]*n[1],  // distance between arrays
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);

    m_fw_plan2 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_out_size[0]),        // array of M elements
			     m_out_size[1] * m_out_size[2],   // NxP arrays
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,           // contiguous input
			     m_out_size[0],         // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_out_size[0],         // distance between arrays
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);

    m_bk_plan1 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &(m_out_size[0]),        // array of M elements
			     m_out_size[1] * m_out_size[2],   // NxP arrays
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     m_out_size[0],         // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     m_out_size[0],         // distance between arrays
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);
#endif    
}

__global__ void get_data_from_mesh_kernel(iris_real*** src3D, int count, iris_real* dest)
{
    iris_real *src = &(src3D[0][0][0]);
    
    int rndx = IRIS_CUDA_INDEX(x);
    int rchunk_size = IRIS_CUDA_CHUNK(x,count);
    
    int i_from = rndx*rchunk_size, i_to = MIN((rndx+1)*rchunk_size,count);
    
    for(int i=i_from;i<i_to;i++) {
	int j = 2*i;
	dest[j] = src[i];
	dest[j+1] = 0.0;
    }
}

__global__
void send_data_to_mesh_kernel(iris_real* src, int count, iris_real ***dest3D)
{
    iris_real *dest = &(dest3D[0][0][0]);
    
    int rndx = IRIS_CUDA_INDEX(x);
    int rchunk_size = IRIS_CUDA_CHUNK(x,count);
    
    int i_from = rndx*rchunk_size, i_to = MIN((rndx+1)*rchunk_size,count);
    
    for(int i=i_from;i<i_to;i++) {
	int j = 2*i;
	dest[i] = src[j];
    }
}

iris_real *fft_plane_gpu::compute_fw(iris_real ***src, iris_real *dest)
{
    int nthreads = IRIS_CUDA_SHARED_BLOCK_SIZE;
    int nblocks = get_NBlocks_X(m_count,nthreads);
    get_data_from_mesh_kernel<<<nblocks, nthreads>>>(src, m_count, dest);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
    
#ifdef IRIS_CUDA
    FFTW_(execute_dft)(m_fw_plan1, (complex_t *)dest, (complex_t *)dest);
    m_fw_remap->perform(dest, dest, m_scratch);
    FFTW_(execute_dft)(m_fw_plan2, (complex_t *)dest, (complex_t *)dest);
#endif
    
    return dest;
}

void fft_plane_gpu::compute_bk(iris_real *src, iris_real ***dest)
{
    
#ifdef IRIS_CUDA
    FFTW_(execute_dft)(m_bk_plan1, (complex_t *)src, (complex_t *)src);
    m_bk_remap->perform(src, src, m_scratch);
    FFTW_(execute_dft)(m_bk_plan2, (complex_t *)src, (complex_t *)src);
#endif

	
    int nthreads = IRIS_CUDA_SHARED_BLOCK_SIZE;
    int nblocks = get_NBlocks_X(m_count,nthreads);
    
    send_data_to_mesh_kernel<<<nblocks,nthreads>>>(src, m_count, dest);
    cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}
