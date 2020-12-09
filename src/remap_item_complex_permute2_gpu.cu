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
#include "remap_item_complex_permute2_gpu.h"
#include "cuda_parameters.h"

using namespace ORG_NCSA_IRIS;

remap_item_complex_permute2_gpu::remap_item_complex_permute2_gpu()
{
}

remap_item_complex_permute2_gpu::~remap_item_complex_permute2_gpu()
{
}

__global__
void unpack_kernel2(iris_real *src, int src_offset, iris_real *dest, int dest_offset,
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

	src = &(src[src_offset]);
	dest = &(dest[dest_offset]);

    for(int i = i_from; i < i_to; i++) {
    int si_i = 2*i*m_ny*m_nz;
	for(int j = j_from; j < j_to; j++) {
	    int di_j = 2*i + j*m_stride_plane;
	    int si_j = si_i + 2*j*m_nz;
	    for(int k = k_from; k < k_to; k++) {
            int si_k = si_j + 2*k;
            int di_k = di_j + k*m_stride_line;
			dest[di_k] = src[si_k];
            si_k = si_k + 1;
            di_k = di_k + 1;
			dest[di_k] = src[si_k];
		}
	}
	}
}

void remap_item_complex_permute2_gpu::unpack(iris_real *src, int src_offset, iris_real *dest, int dest_offset)
{
    int nthreads1 = get_NThreads_X(m_nx);
	int nthreads2 = get_NThreads_Y(m_ny);
	int nthreads3 = get_NThreads_Z(m_nz);
    int nblocks1 = get_NBlocks_X(m_nx,nthreads1);
	int nblocks2 = get_NBlocks_Y(m_ny,nthreads2);
	int nblocks3 = get_NBlocks_Z(m_nz,nthreads3);

	auto blocks = dim3(nblocks1,nblocks2,nblocks3);
    auto threads = dim3(nthreads1,nthreads2,nthreads3);
	
	unpack_kernel2<<<blocks,threads>>>
	(src, src_offset, dest, dest_offset, m_nx, m_ny, m_nz, m_stride_plane, m_stride_line);
	cudaDeviceSynchronize();
    HANDLE_LAST_CUDA_ERROR;
}
