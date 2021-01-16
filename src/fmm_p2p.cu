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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifdef IRIS_CUDA
#include <assert.h>
#include "cuda.h"
#include "fmm.h"
#include "real.h"
#include "ses.h"
#include "fmm_pair.h"

using namespace ORG_NCSA_IRIS;


///////////////////////
// P2P inside a cell //
///////////////////////


//
// NOTE: Think twice before changing this. Have in mind that a single division here slows down the whole kernel twice!
//
__device__ __forceinline__ void d_coulomb(iris_real3 tp, iris_real3 sp, iris_real sq, iris_real &sum_phi, iris_real3 &e)
{
    iris_real dx = tp.x - sp.x;
    iris_real dy = tp.y - sp.y;
    iris_real dz = tp.z - sp.z;
    iris_real rlen = __rsqrt(__fma(dx, dx, __fma(dy, dy, __fma(dz, dz, 0.0))));
    dx *= rlen * rlen;
    dy *= rlen * rlen;
    dz *= rlen * rlen;
    rlen *= sq;
    sum_phi += rlen;
    e.x = __fma(dx, rlen, e.x);
    e.y = __fma(dy, rlen, e.y);
    e.z = __fma(dz, rlen, e.z);
}

__global__ void k_p2p_self(cell_t *m_cells, particle_t *m_particles, int offset)
{
    int leaf_idx = blockIdx.y * gridDim.z + blockIdx.z;   // Which interaction pair we're processing
    int cellID = leaf_idx + offset;                       // This is C -> C, so cellID = sourceID = destID
    int i = IRIS_CUDA_TID;                                // Target particle inside cellID
    int npart = m_cells[cellID].num_children;             // Number of particles in the cell
    
    if(i >= npart) {                                      // Make sure this is a valid particle
	return;
    }

    float phi = 0.0;                                      // Computed potential
    iris_real3 e = make_iris_real3(0.0, 0.0, 0.0);        // Computed field
    
    int start = m_cells[cellID].first_child;              // Where do the particles in cellID start
    int end = start + npart;                              // Where do the particles in cellID end
    
    int ii = start + i;                                   // Index of the target particle in m_particles
    iris_real4 tmp = *reinterpret_cast<iris_real4 *>(m_particles[ii].xyzq);
    iris_real3 tpp = make_iris_real3(tmp.x, tmp.y, tmp.z);// target particle position
    float tq = tmp.w;
    
    // Process all particles except i-th one to avoid self-interaction and infinities
    // Use two identical loops (before i; after i) to avoid an 'if'
    for(int j=start;j<ii;j++) {
	iris_real4 tmp = *reinterpret_cast<iris_real4 *>(m_particles[j].xyzq);
	iris_real3 spp = make_iris_real3(tmp.x, tmp.y, tmp.z);    // source particle position
	iris_real sq = tmp.w;                             // source charge
	d_coulomb(tpp, spp, sq, phi, e);                  // compute Coulomb interaction
    }
    for(int j=ii+1;j<end;j++) {
	iris_real4 tmp = *reinterpret_cast<iris_real4 *>(m_particles[j].xyzq);
	iris_real3 spp = make_iris_real3(tmp.x, tmp.y, tmp.z);    // source particle position
	iris_real sq = tmp.w;                             // source charge
	d_coulomb(tpp, spp, sq, phi, e);                  // compute Coulomb interaction
    }

    // atomically reduce the computed potential and field to the target particle's result
    atomicAdd(m_particles[ii].tgt + 0, phi);
    atomicAdd(m_particles[ii].tgt + 1, tq*e.x);
    atomicAdd(m_particles[ii].tgt + 2, tq*e.y);
    atomicAdd(m_particles[ii].tgt + 3, tq*e.z);
}

void fmm::eval_p2p_self_gpu()
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;
    
    dim3 nthreads(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks((m_max_particles-1)/IRIS_CUDA_NTHREADS + 1, nleafs, 1);
    k_p2p_self<<<nblocks, nthreads, 0, m_streams[1]>>>(m_cells, m_particles, offset);
}


/////////////////////////
// P2P with neighbours //
/////////////////////////


__device__ __forceinline__ void __reduce_warpx(iris_real4 &phie, int tid)
{
    phie.x += __shfl_down_sync(0xFFFFFFFF, phie.x, 1);
    phie.y += __shfl_up_sync  (0xFFFFFFFF, phie.y, 1);
    phie.z += __shfl_down_sync(0xFFFFFFFF, phie.z, 1);
    phie.w += __shfl_up_sync  (0xFFFFFFFF, phie.w, 1);
    
    if (tid & 1) {
	phie.x = phie.y;
	phie.z = phie.w;
    }
    
    phie.x += __shfl_down_sync(0xFFFFFFFF, phie.x, 2);
    phie.z += __shfl_up_sync  (0xFFFFFFFF, phie.z, 2);
    
    if (tid & 2) {
        phie.x = phie.z;
    }
    
    phie.x += __shfl_down_sync(0xFFFFFFFF, phie.x, 4);
}

__device__ __forceinline__ void __reduce_warpy(iris_real4 &phie, int tid)
{
    phie.x += __shfl_down_sync(0xFFFFFFFF, phie.x, 8);
    phie.y += __shfl_up_sync  (0xFFFFFFFF, phie.y, 8);
    phie.z += __shfl_down_sync(0xFFFFFFFF, phie.z, 8);
    phie.w += __shfl_up_sync  (0xFFFFFFFF, phie.w, 8);
    
    if (tid & 1) {
	phie.x = phie.y;
	phie.z = phie.w;
    }
    
    phie.x += __shfl_down_sync(0xFFFFFFFF, phie.x, 16);
    phie.z += __shfl_up_sync  (0xFFFFFFFF, phie.z, 16);
    
    if (tid & 2) {
        phie.x = phie.z;
    }
}

__global__ void k_p2p_neigh(interact_item_t *list, cell_t *m_cells, cell_t *m_xcells, particle_t *m_particles, xparticle_t **m_xparticles,
			    iris_real gxsize, iris_real gysize, iris_real gzsize)
{
    int pair_idx = blockIdx.y * gridDim.z + blockIdx.z;                  // Which interaction pair we're processing
    int srcID = list[pair_idx].sourceID;                                 // Source cell
    int destID = list[pair_idx].targetID;                                // Destination cell

    int block_tid = threadIdx.y * 8 + threadIdx.x;                       // Unique thread index inside the block
    int si = blockIdx.x*64 + block_tid;                                  // Index of the source particle to start from

    particle_t *sparticles;                                             // Source particles
    particle_t *dparticles = m_particles + m_cells[destID].first_child;  // Destination particles

    
    // When the source cell is an alien cell, look at xparticles instead of particles.
    // Moreover, don't bother setting its calculated potential and field -- we won't use it...
    bool do_other_side;
    if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_LEAF) {
	do_other_side = false;
	if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_L1) {
	    sparticles = (particle_t *)(m_xparticles[0] + m_xcells[srcID].first_child + si);
	}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_L2) {
	    sparticles = (particle_t *)(m_xparticles[1] + m_xcells[srcID].first_child + si);
	}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_L3) {
	    sparticles = (particle_t *)(m_xparticles[2] + m_xcells[srcID].first_child + si);
	}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_L4) {
	    sparticles = (particle_t *)(m_xparticles[3] + m_xcells[srcID].first_child + si);
	}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_L5) {
	    sparticles = (particle_t *)(m_xparticles[4] + m_xcells[srcID].first_child + si);
	}else if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_L6) {
	    sparticles = (particle_t *)(m_xparticles[5] + m_xcells[srcID].first_child + si);
	}
    }else {
	do_other_side = true;
	sparticles = m_particles + m_xcells[srcID].first_child + si;
    }

    
    // Store 64 source particles (one per thread) in the src shared buffer.
    // In case there are not enough particles, store a dummy particle
    // that is sufficiently far from all the others. This is crucially needed for the
    // warp reduction -- all threads must participate, otherwise we have pigs in space...
    //
    // Shared buffer is used to avoid fetches from RAM and cache misses in the inner loop
    __shared__ iris_real4 src[64];
    if(si < m_xcells[srcID].num_children) {
	src[block_tid].x = __fma(list[pair_idx].ix, gxsize, sparticles->xyzq[0]);
	src[block_tid].y = __fma(list[pair_idx].iy, gysize, sparticles->xyzq[1]);
	src[block_tid].z = __fma(list[pair_idx].iz, gzsize, sparticles->xyzq[2]);
	src[block_tid].w = sparticles->xyzq[3];
    }else {
	src[block_tid] = make_iris_real4(1.0e+32, 1.0e+32, 1.0e+32, 0.0);
    }
    __syncthreads();

    
    // Initialize the resulting potential/field for the source particles
    iris_real4 s_phie[8];
    for(int k=0;k<8;k++) {
	s_phie[k] = make_iris_real4(0.0, 0.0, 0.0, 0.0);
    }

    // Process destination particles in batches of 8
    int nc = m_cells[destID].num_children;                               // original # of destination particles
    int batches = nc/8;                                                  // # of batches
    int di = threadIdx.y;                                                // index of destination particle
    for(int k=0;k<=batches;k++) {
	// Move particle data in dest. The last batch may not be whole, in which
	// case set a dummy dest of a faraway particle (but not same as the
	// source far away particles). This is crucially needed for the warp
	// reduction -- all threads must participate...
	iris_real4 dest;
	if(di >= nc) {
	    dest = make_iris_real4(1.0e+16, 1.0e+16, 1.0e+16, 0.0);
	}else {
	    dest = *reinterpret_cast<iris_real4 *>(dparticles[di].xyzq);
	}

	// Initialize the resulting potential/field for the dest particle
	iris_real4 d_phie = make_iris_real4(0.0, 0.0, 0.0, 0.0);
	
	int si = threadIdx.x;                                            // index of the source particle
	for(int m=0;m<8;m++) {
	    // This is just coulomb, but simultaneous for the src and dest potential/field
	    iris_real dx = src[si].x - dest.x;
	    iris_real dy = src[si].y - dest.y;
	    iris_real dz = src[si].z - dest.z;
	    iris_real rlen = __rsqrt(__fma(dx, dx, __fma(dy, dy, __fma(dz, dz, 0.0))));

	    iris_real sphi = dest.w * rlen;                              // potential at source
	    iris_real dphi = src[si].w * rlen;                           // potential at dest

	    rlen *= dphi * sphi;

	    dx *= rlen;
	    dy *= rlen;
	    dz *= rlen;

	    s_phie[m].x += sphi;
	    s_phie[m].y += dx;
	    s_phie[m].z += dy;
	    s_phie[m].w += dz;
	    
	    d_phie.x += dphi;
	    d_phie.y -= dx;
	    d_phie.z -= dy;
	    d_phie.w -= dz;

	    si += 8;
	}

	// Warp reduction of the destination potential/field.
	// Threads (Y, 0)     d_phie.x contains the potential
	// Threads (Y, 1,2,3) d_phie.y,z,w contain the field
	// Threads (Y, 4..7) contain garbage
	__reduce_warpx(d_phie, threadIdx.x);
	if (threadIdx.x <  4) {
	    atomicAdd(dparticles[di].tgt + threadIdx.x, d_phie.x);
	}
	
	di += 8;
    }

    // This is only needed in case we want to assign the calculated potential/field
    // to the source particle. For particles coming from other ranks this is not needed.
    if(do_other_side) {
	sparticles -= si;
	for(int k=0;k<8;k++) {
	    int si = blockIdx.x*64 + k*8 + threadIdx.x;
	    __reduce_warpy(s_phie[k], threadIdx.y);  // Similar to the warpx above
	    if ((threadIdx.y & 3) < 4) {
		atomicAdd(sparticles[si].tgt + (threadIdx.y & 3), s_phie[k].x);
	    }
	}
    }
}

void fmm::eval_p2p_gpu()
{
    int n = m_p2p_list.size();
    if(n == 0) {
	return;
    }

    m_p2p_list_gpu = (interact_item_t *)memory::wmalloc_gpu_cap(m_p2p_list_gpu, n, sizeof(interact_item_t), &m_p2p_list_cap);
    cudaMemcpyAsync(m_p2p_list_gpu, m_p2p_list.data(), n * sizeof(interact_item_t), cudaMemcpyDefault, m_streams[1]);
    cudaEventRecord(m_p2p_memcpy_done, m_streams[1]);

    // NOTE: The whole thing only works for 8x8x1 blocks, so don't try to change it.
    dim3 nthreads(8, 8, 1);
    dim3 nblocks((m_max_particles-1)/64 + 1, n, 1);
    k_p2p_neigh<<<nblocks, nthreads, 0, m_streams[1]>>>(m_p2p_list_gpu, m_cells, m_xcells, m_particles, m_xparticles,
							m_domain->m_global_box.xsize, m_domain->m_global_box.ysize, m_domain->m_global_box.zsize);
    cudaEventSynchronize(m_p2p_memcpy_done);
    m_p2p_list.clear();
}

#endif
