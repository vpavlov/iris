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
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "comm_rec.h"
#include "fmm.h"
#include "real.h"
#include "cuda.h"

using namespace ORG_NCSA_IRIS;

__global__ void k_load_charges(iris_real *charges, int ncharges, int hwm,
			       iris_real xlo, iris_real ylo, iris_real zlo,
			       iris_real lsx, iris_real lsy, iris_real lsz,
			       int max_level, int offset, particle_t *m_particles, int rank,
			       int *atom_types)
{
    IRIS_CUDA_SETUP_WS(ncharges);
    for(int i=from;i<to;i++) {
	iris_real tx = (charges[i * 5 + 0] - xlo) / lsx;
	iris_real ty = (charges[i * 5 + 1] - ylo) / lsy;
	iris_real tz = (charges[i * 5 + 2] - zlo) / lsz;

	int lc[3];
	lc[0] = (int) tx;
	lc[1] = (int) ty;
	lc[2] = (int) tz;
	
	int id = 0;
	for(int l=0;l<max_level; l++) {
	    for(int d=0;d<3;d++) {
		id += (lc[d] & 1) << (3*l + d);
		lc[d] >>= 1;
	    }
	}
	
	int cellID = offset + id;
	int chargeID = (int)charges[i*5 + 4];
	
	m_particles[i+hwm].rank = rank;
	m_particles[i+hwm].index = chargeID; // (chargeID > 0)?chargeID:-chargeID;
	m_particles[i+hwm].cellID = cellID;
	m_particles[i+hwm].xyzq[0] = charges[i*5+0];
	m_particles[i+hwm].xyzq[1] = charges[i*5+1];
	m_particles[i+hwm].xyzq[2] = charges[i*5+2];
	m_particles[i+hwm].xyzq[3] = charges[i*5+3];
	m_particles[i+hwm].tgt[0] = 0.0;
	m_particles[i+hwm].tgt[1] = 0.0;
	m_particles[i+hwm].tgt[2] = 0.0;
	m_particles[i+hwm].tgt[3] = 0.0;
	atom_types[i+hwm] = (chargeID > 0)?0:1;
    }
}

__global__ void k_extract_cellID(particle_t *m_particles, int n, int *cellID_keys)
{
    IRIS_CUDA_SETUP_WS(n);
    for(int i=from;i<to;i++) {
	cellID_keys[i] = m_particles[i].cellID;
    }
}

void fmm::load_particles_gpu()
{
    // Find the total amount of local charges, including halo atoms
    // This is just a simple sum of ncharges from all incoming rank
    int total_local_charges = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	total_local_charges += m_iris->m_ncharges[rank];
    }

    // Allocate GPU memory for m_particles (and m_xparticles, it will be at the end of m_particles)
    // Also, allocate same sized array for the atom types (=1 if halo atom, =0 if own atom)
    m_particles = (particle_t *)memory::wmalloc_gpu_cap((void *)m_particles, total_local_charges, sizeof(particle_t), &m_npart_cap);
    m_atom_types = (int *)memory::wmalloc_gpu_cap((void *)m_atom_types, total_local_charges, sizeof(int), &m_at_cap);
    m_cellID_keys = (int *)memory::wmalloc_gpu_cap((void *)m_cellID_keys, total_local_charges, sizeof(int), &m_cellID_keys_cap);

    // Allocate GPU memory for the charges coming from all client ranks
    // This is done all at once so as no to interfere with mem transfer/kernel overlapping in the next loop
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	int ncharges = m_iris->m_ncharges[rank];
	iris_real *charges = m_iris->m_charges[rank];
	m_charges_gpu[rank] = (iris_real *)memory::wmalloc_gpu_cap((void *)m_charges_gpu[rank], ncharges, 5*sizeof(iris_real), &m_charges_gpu_cap[rank]);
    }

    // Start the particle loading itself. The <H2D, kernel> pairs runs on separate streams and thus overlaps memory transfer for more than 1 source rank
    int offset = cell_meta_t::offset_for_level(max_level());
    int nd = 1 << max_level();
    int hwm = 0;    
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	int ncharges = m_iris->m_ncharges[rank];
	iris_real *charges = m_iris->m_charges[rank];
	cudaMemcpyAsync(m_charges_gpu[rank], charges, ncharges * 5 * sizeof(iris_real), cudaMemcpyDefault, m_streams[rank % IRIS_CUDA_FMM_NUM_STREAMS]);
	
	int nthreads = IRIS_CUDA_NTHREADS;
	int nblocks IRIS_CUDA_NBLOCKS(ncharges, nthreads);
	k_load_charges<<<nblocks, nthreads, 0, m_streams[rank % IRIS_CUDA_FMM_NUM_STREAMS]>>>(m_charges_gpu[rank], ncharges, hwm,
											      m_domain->m_global_box.xlo, m_domain->m_global_box.ylo, m_domain->m_global_box.zlo,
											      m_leaf_size[0], m_leaf_size[1], m_leaf_size[2],
											      max_level(), offset, m_particles, rank,
											      m_atom_types);
	cudaError_t err = cudaGetLastError();
	IRIS_CUDA_HANDLE_ERROR(err);
	hwm += ncharges;
    }
    cudaDeviceSynchronize();  // all k_load_charges kernels must have finished to have valid m_particles
        
    // At this point we have the m_particles filled up, with mixed halo/own atoms, etc.
    // We need to sort them according to the m_atom_type keys array and split into m_particles and m_xparticles
    thrust::device_ptr<int>         keys(m_atom_types);
    thrust::device_ptr<particle_t>  values(m_particles);
    thrust::sort_by_key(keys, keys+total_local_charges, values);

    // Now the first part of m_particles contains local atoms; second contains halo atoms
    // Number of local particles can be taken from iris: num_local_atoms
    // Number of halo particles is total_local_charges - num_local_atoms
    m_nparticles = m_iris->num_local_atoms();
    m_nxparticles = total_local_charges - m_nparticles;
    m_xparticles = m_particles + m_nparticles;

    // Get the cellIDs in the reordered array
    int nthreads = IRIS_CUDA_NTHREADS;
    int nblocks IRIS_CUDA_NBLOCKS(total_local_charges, nthreads);
    k_extract_cellID<<<nblocks, nthreads>>>(m_particles, total_local_charges, m_cellID_keys);

    // We now need to sort the m_particles and m_xparticles arrays by cellID
    thrust::device_ptr<int>         keys2(m_cellID_keys);
    thrust::device_ptr<particle_t>  part(m_particles);
    thrust::device_ptr<particle_t>  xpart(m_xparticles);
    thrust::sort_by_key(keys2, keys2+m_nparticles, part);
    thrust::sort_by_key(keys2+m_nparticles, keys2+total_local_charges, xpart);
    
    cudaDeviceSynchronize();  // all k_load_charges kernels must have finished to have valid m_particles
    m_logger->info("FMM/GPU: This rank owns %d + %d halo particles", m_nparticles, m_nxparticles);
}

#endif
