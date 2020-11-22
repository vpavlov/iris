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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#include "remap_item_complex_permute_gpu.h"
#include "cuda_parameters.h"

using namespace ORG_NCSA_IRIS;

remap_item_complex_permute_gpu::remap_item_complex_permute_gpu()
{
}

remap_item_complex_permute_gpu::~remap_item_complex_permute_gpu()
{
}

__global__
void unpack_kernel1(iris_real *src, iris_real *dest,
				int m_nx, int m_ny, int m_nz,
				int m_stride_plane, int m_stride_line)
{
    int xndx = IRIS_CUDA_INDEX(x);
    int xchunk_size = IRIS_CUDA_CHUNK(x,m_nx);
    int yndx = IRIS_CUDA_INDEX(y);
    int ychunk_size = IRIS_CUDA_CHUNK(y,m_ny);
    int zndx = IRIS_CUDA_INDEX(z);
    int zchunk_size = IRIS_CUDA_CHUNK(z,m_nz);

	int i_from = xndx*xchunk_size, i_to = MIN((xndx+1)*xchunk_size,m_nx);
	int j_from = yndx*ychunk_size, j_to = MIN((yndx+1)*ychunk_size,m_ny);
	int k_from = zndx*zchunk_size, k_to = MIN((zndx+1)*zchunk_size,m_nz);

    for(int i = i_from; i < i_to; i++) {
	int plane = i * m_stride_line;
	int si_i = 2*i*m_ny*m_nz;
	for(int j = j_from; j < j_to; j++) {
		int di_j = plane + 2*j;
		int si_j = si_i + 2*j*m_nz;
	    for(int k = k_from; k < k_to; k++) {
			int si_k = si_j + 2*k;
			int di_k = di_j + k*m_stride_plane;
			dest[di_k] = src[si_k];
			si_k = si_k + 1;
			di_k = di_k + 1;
			dest[di_k] = src[si_k];
		}
	}
	}
}

void remap_item_complex_permute_gpu::unpack(iris_real *src, iris_real *dest)
{
	int nblocks1 = get_NBlocks(m_nx,IRIS_CUDA_NTHREADS_3D);
	int nblocks2 = get_NBlocks(m_ny,IRIS_CUDA_NTHREADS_3D);
	int nblocks3 = get_NBlocks(m_nz,IRIS_CUDA_NTHREADS_3D);
    int nthreads1 = MIN((m_nx+nblocks1+1)/nblocks1,IRIS_CUDA_NTHREADS_3D);
    int nthreads2 = MIN((m_ny+nblocks2+1)/nblocks2,IRIS_CUDA_NTHREADS_3D);
    int nthreads3 = MIN((m_nz+nblocks3+1)/nblocks3,IRIS_CUDA_NTHREADS_3D);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);
	
	unpack_kernel1<<<blocks,threads>>>
	(src, dest, m_nx, m_ny, m_nz, m_stride_plane, m_stride_line);
	cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}
