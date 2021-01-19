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
#include <assert.h>
#include <thrust/device_vector.h>
#include "cuda.h"
#include "fmm.h"
#include "tags.h"
#include "comm_rec.h"

using namespace ORG_NCSA_IRIS;


/////////////////////////
// Exchange halo (GPU) //
/////////////////////////


__device__ iris_real d_distance_to(box_t<iris_real> &box, iris_real x, iris_real y, iris_real z)
{
    iris_real dx = (x > box.xhi) * (x - box.xhi) + (x < box.xlo) * (box.xlo - x);
    iris_real dy = (y > box.yhi) * (y - box.yhi) + (y < box.ylo) * (box.ylo - y);
    iris_real dz = (z > box.zhi) * (z - box.zhi) + (z < box.zlo) * (box.zlo - z);
    return sqrt(dx*dx + dy*dy + dz*dz);
}


__global__ void k_border_leafs(box_t<iris_real> rank_box, int *m_a2a_cell_cnt, int start, int end, cell_t *m_xcells, iris_real m_let_corr,
			       iris_real gxsize, iris_real gysize, iris_real gzsize, iris_real m_mac)
{
    int n = IRIS_CUDA_TID + start;
    if(n >= end) {
	return;
    }

    if(m_xcells[n].num_children == 0) {  // only full cells
    	return;
    }

    int per_idx = blockIdx.y;
    int ix = per_idx % 3 - 1;
    per_idx /= 3;
    int iy = per_idx % 3 - 1;
    per_idx /= 3;
    int iz = per_idx % 3 - 1;

    iris_real dn = m_xcells[n].ses.r + m_let_corr;
    iris_real cx = m_xcells[n].ses.c.r[0];
    iris_real cy = m_xcells[n].ses.c.r[1];
    iris_real cz = m_xcells[n].ses.c.r[2];

    iris_real x = __fma(ix, gxsize, cx);
    iris_real y = __fma(iy, gysize, cy);
    iris_real z = __fma(iz, gzsize, cz);
    iris_real rn = d_distance_to(rank_box, x, y, z);
    
    if (dn/rn < m_mac) {
    	return;
    }

    
    m_a2a_cell_cnt[n-start] = m_xcells[n].num_children;
    // WTF: If you remove this nonsese, the kernel always returns 0 in m_a2a_cell_cnt?!?!? Bug in CUDA compiler optimization????
    if(m_a2a_cell_cnt[n-start] == 0) {
    	printf("WTF CUDA???\n");
    }
}


__global__ void k_fill_sendbuf(xparticle_t *out_sendbuf, particle_t *m_particles, cell_t *m_xcells, int *m_a2a_cell_cnt, int *m_a2a_cell_disp, int offset)
{
    int leaf_idx = blockIdx.y * gridDim.z + blockIdx.z;   // Which leaf ?
    int cellID = leaf_idx + offset;
    int j = IRIS_CUDA_TID;                                // Target particle inside cellID
    int npart = m_xcells[cellID].num_children;            // Number of particles in the cell

    if(j >= npart) {                                      // Make sure this is a valid particle
    	return;
    }

    if(m_a2a_cell_cnt[leaf_idx] == 0) {
    	return;
    }
    
    cell_t *leaf = &m_xcells[cellID];
    int disp = m_a2a_cell_disp[leaf_idx];
    
    out_sendbuf[disp+j].xyzq[0] = m_particles[leaf->first_child+j].xyzq[0];
    out_sendbuf[disp+j].xyzq[1] = m_particles[leaf->first_child+j].xyzq[1];
    out_sendbuf[disp+j].xyzq[2] = m_particles[leaf->first_child+j].xyzq[2];
    out_sendbuf[disp+j].xyzq[3] = m_particles[leaf->first_child+j].xyzq[3];
    out_sendbuf[disp+j].cellID = m_particles[leaf->first_child+j].cellID;
}

int fmm::collect_halo_for_gpu(int rank, int hwm)
{
    int start = cell_meta_t::offset_for_level(max_level());
    int end = m_tree_size;
    int nleafs = end - start;

    thrust::device_vector<int> *cell_cnt = (thrust::device_vector<int> *)m_a2a_cell_cnt_gpu;
    thrust::device_vector<int> *cell_disp = (thrust::device_vector<int> *)m_a2a_cell_disp_gpu;
    
    cell_cnt->resize(nleafs);
    cell_disp->resize(nleafs);
    cudaMemsetAsync(cell_cnt->data().get(), 0, nleafs * sizeof(int), m_streams[0]);

    int *cnt_ptr = cell_cnt->data().get();
    int *disp_ptr = cell_disp->data().get();
    
    dim3 nthreads(MIN(IRIS_CUDA_NTHREADS, nleafs), 1, 1);
    dim3 nblocks((nleafs - 1)/IRIS_CUDA_NTHREADS + 1, 27, 1);
    k_border_leafs<<<nblocks, nthreads, 0, m_streams[0]>>>(m_domain->m_local_boxes[rank], cnt_ptr, start, end, m_xcells, m_let_corr,
							   m_domain->m_global_box.xsize, m_domain->m_global_box.ysize, m_domain->m_global_box.zsize, m_mac);
    thrust::exclusive_scan(thrust::cuda::par.on(m_streams[0]), cnt_ptr, cnt_ptr + nleafs, disp_ptr, 0);
    
    int last_count, last_disp;
    cudaMemcpyAsync(&last_count, cnt_ptr + nleafs - 1, sizeof(int), cudaMemcpyDefault, m_streams[0]);
    cudaMemcpyAsync(&last_disp, disp_ptr + nleafs - 1, sizeof(int), cudaMemcpyDefault, m_streams[0]);
    cudaStreamSynchronize(m_streams[0]);
    
    int halo_size = last_count + last_disp;

    thrust::device_vector<xparticle_t> *sendbuf = (thrust::device_vector<xparticle_t> *)m_a2a_sendbuf_gpu;
    sendbuf->resize(sendbuf->size() + halo_size);

    dim3 nthreads2(IRIS_CUDA_NTHREADS, 1, 1);
    dim3 nblocks2((m_max_particles-1)/IRIS_CUDA_NTHREADS + 1, nleafs, 1);
    k_fill_sendbuf<<<nblocks2, nthreads2, 0, m_streams[0]>>>(sendbuf->data().get() + hwm, m_particles, m_xcells, cnt_ptr, disp_ptr, start);
    
    return halo_size;
}

void fmm::exchange_p2p_halo_gpu()
{
    timer tm;
    tm.start();
    
    m_a2a_send_cnt.assign(m_local_comm->m_size, 0);
    m_a2a_send_disp.assign(m_local_comm->m_size, 0);
    m_a2a_recv_cnt.resize(m_local_comm->m_size);
    m_a2a_recv_disp.resize(m_local_comm->m_size);
    thrust::device_vector<xparticle_t> *sendbuf = (thrust::device_vector<xparticle_t> *)m_a2a_sendbuf_gpu;
    sendbuf->clear();

    int hwm = 0;
    for(int rank=0;rank<m_local_comm->m_size;rank++) {
    	if(rank == m_local_comm->m_rank) {
    	    continue;
    	}
    	int cnt = collect_halo_for_gpu(rank, hwm);
    	m_a2a_send_cnt[rank] = cnt;
    	m_a2a_send_disp[rank] = hwm;
    	hwm += cnt;
    }

    for(int i=0;i<m_local_comm->m_size;i++) {
    	m_logger->tace("Will be sending %d particles to %d, starting from %d", m_a2a_send_cnt[i], i, m_a2a_send_disp[i]);
    }
    
    MPI_Alltoall(m_a2a_send_cnt.data(), 1, MPI_INT, m_a2a_recv_cnt.data(), 1, MPI_INT, m_local_comm->m_comm);

    int rsize = 0;
    for(int i=0;i<m_local_comm->m_size;i++) {
    	m_a2a_recv_disp[i] = rsize;
    	rsize += m_a2a_recv_cnt[i];
    }
    
    for(int i=0;i<m_local_comm->m_size;i++) {
    	m_logger->trace("Will be receiving %d particles from %d, starting from %d", m_a2a_recv_cnt[i], i, m_a2a_recv_disp[i]);
    }

    for(int i=0;i<m_local_comm->m_size;i++) {
    	m_a2a_send_cnt[i] *= sizeof(xparticle_t);
    	m_a2a_send_disp[i] *= sizeof(xparticle_t);
    	m_a2a_recv_cnt[i] *= sizeof(xparticle_t);
    	m_a2a_recv_disp[i] *= sizeof(xparticle_t);
    }

    m_a2a_recvbuf.resize(rsize);
    m_a2a_sendbuf_cpu.resize(hwm);
    m_xparticles[0] = (xparticle_t *)memory::wmalloc_gpu_cap(m_xparticles[0], rsize, sizeof(xparticle_t), m_xparticles_cap);
    
    cudaMemcpyAsync(m_a2a_sendbuf_cpu.data(), sendbuf->data().get(), hwm*sizeof(xparticle_t), cudaMemcpyDefault, m_streams[0]);
    cudaStreamSynchronize(m_streams[0]);
    
    MPI_Alltoallv(m_a2a_sendbuf_cpu.data(), m_a2a_send_cnt.data(), m_a2a_send_disp.data(), MPI_BYTE,
    		  m_a2a_recvbuf.data(), m_a2a_recv_cnt.data(), m_a2a_recv_disp.data(), MPI_BYTE,
    		  MPI_COMM_WORLD);

    cudaMemcpyAsync(m_xparticles[0], m_a2a_recvbuf.data(), rsize*sizeof(xparticle_t), cudaMemcpyDefault, m_streams[0]);
    distribute_particles(m_xparticles[0], rsize, IRIS_FMM_CELL_ALIEN_L1, m_xcells);
    
    tm.stop();
    m_logger->time("Halo exchange wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

int fmm::comm_LET_gpu()
{
    cudaMemcpyAsync(m_cells_cpu, m_cells, m_tree_size*sizeof(cell_t), cudaMemcpyDefault, m_streams[1]);
    cudaMemcpyAsync(m_M_cpu, m_M, m_tree_size*2*m_nterms*sizeof(iris_real), cudaMemcpyDefault, m_streams[1]);
    cudaStreamSynchronize(m_streams[1]);
    m_has_cells_cpu = true;
    return comm_LET_cpu(m_cells_cpu, m_M_cpu);
}

__global__ void k_inhale_cells(unsigned char *m_recvbuf, int in_count, cell_t *m_xcells, iris_real *m_M, int unit_size, int m_nterms)
{
    int i = IRIS_CUDA_TID;
    if(i < in_count) {
	int cellID = *(int *)(m_recvbuf + unit_size * i);
	memcpy(&(m_xcells[cellID].ses), m_recvbuf + unit_size * i + sizeof(int), sizeof(sphere_t));
	memcpy(m_M + cellID*2*m_nterms, m_recvbuf + unit_size * i + sizeof(int) + sizeof(sphere_t), 2*m_nterms*sizeof(iris_real));
	m_xcells[cellID].flags |= (IRIS_FMM_CELL_ALIEN_NL | IRIS_FMM_CELL_VALID_M);
    }
}

void fmm::inhale_xcells_gpu(int in_count)
{
    int unit_size = sizeof(int) + sizeof(sphere_t) + 2*m_nterms*sizeof(iris_real);
    int rsize = in_count * unit_size;
    m_recvbuf_gpu = (unsigned char *)memory::wmalloc_gpu_cap(m_recvbuf_gpu, rsize, 1, &m_recvbuf_gpu_cap);
    cudaMemcpy(m_recvbuf_gpu, m_recvbuf, rsize, cudaMemcpyDefault);
    int nthreads = IRIS_CUDA_NTHREADS;
    int nblocks = IRIS_CUDA_NBLOCKS(in_count, nthreads);
    k_inhale_cells<<<nblocks, nthreads>>>(m_recvbuf_gpu, in_count, m_xcells, m_M, unit_size, m_nterms);
}


// void fmm::send_particles_to_neighbour_gpu(int rank, void *out_sendbuf_void, std::vector<xparticle_t> *out_sendbuf_cpu,
// 					  MPI_Request *out_cnt_req, MPI_Request *out_data_req, cudaStream_t &stream,
// 					  int *in_halo_cell_cnt, int *in_halo_cell_disp)
// {
//     thrust::device_vector<xparticle_t> *out_sendbuf = (thrust::device_vector<xparticle_t> *)out_sendbuf_void;

//     int start = cell_meta_t::offset_for_level(max_level());
//     int end = m_tree_size;
//     int nleafs = end - start;

//     // TODO: figure out partial periodic and non-periodic PBC
//     cudaMemsetAsync(in_halo_cell_cnt, 0, nleafs * sizeof(int), stream);
//     dim3 nthreads(MIN(IRIS_CUDA_NTHREADS, nleafs), 1, 1);
//     dim3 nblocks((nleafs - 1)/IRIS_CUDA_NTHREADS + 1, 27, 1);
//     k_border_leafs<<<nblocks, nthreads, 0, stream>>>(m_domain->m_local_boxes[rank], in_halo_cell_cnt, start, end, m_xcells, m_let_corr,
// 						     m_domain->m_global_box.xsize, m_domain->m_global_box.ysize, m_domain->m_global_box.zsize, m_mac);

//     thrust::exclusive_scan(thrust::cuda::par.on(stream), in_halo_cell_cnt, in_halo_cell_cnt + nleafs, in_halo_cell_disp, 0);
//     int last_count, last_disp;
//     cudaMemcpyAsync(&last_count, in_halo_cell_cnt + nleafs-1, sizeof(int), cudaMemcpyDefault, stream);
//     cudaMemcpyAsync(&last_disp, in_halo_cell_disp + nleafs-1, sizeof(int), cudaMemcpyDefault, stream);
//     cudaStreamSynchronize(stream);
//     int part_count = last_count + last_disp;
    
//     m_logger->info("Will be sending %d particles to neighbour %d", part_count, rank);
//     MPI_Isend(&part_count, 1, MPI_INT, rank, IRIS_TAG_FMM_P2P_HALO_CNT, m_local_comm->m_comm, out_cnt_req);

//     out_sendbuf_cpu->resize(part_count);
//     out_sendbuf->resize(part_count);
//     dim3 nthreads2(IRIS_CUDA_NTHREADS, 1, 1);
//     dim3 nblocks2((m_max_particles-1)/IRIS_CUDA_NTHREADS + 1, nleafs, 1);
//     k_prepare_sendbuf<<<nblocks2, nthreads2, 0, stream>>>(out_sendbuf->data().get(), m_particles,
// 							  m_xparticles[0], m_xparticles[1], m_xparticles[2],
// 							  m_xparticles[3], m_xparticles[4], m_xparticles[5],
// 							  m_xcells, in_halo_cell_cnt, in_halo_cell_disp, start, end);
//     cudaMemcpyAsync(out_sendbuf_cpu->data(), out_sendbuf->data().get(), part_count * sizeof(xparticle_t), cudaMemcpyDefault, stream);
//     cudaStreamSynchronize(stream);
    
//     MPI_Isend(out_sendbuf_cpu->data(), part_count*sizeof(xparticle_t), MPI_BYTE, rank, IRIS_TAG_FMM_P2P_HALO, m_local_comm->m_comm, out_data_req);
// }

// void fmm::recv_particles_from_neighbour_gpu(int rank, int alien_index, int alien_flag)
// {
//     int part_count;
//     MPI_Recv(&part_count, 1, MPI_INT, rank, IRIS_TAG_FMM_P2P_HALO_CNT, m_local_comm->m_comm, MPI_STATUS_IGNORE);

//     m_logger->info("Will be receiving %d paricles from neighbour %d", part_count, rank);

//     xparticle_t *tmp;
//     cudaMallocHost((void **)&tmp, part_count * sizeof(xparticle_t));
//     m_xparticles[alien_index] = (xparticle_t *)memory::wmalloc_gpu_cap(m_xparticles[alien_index], part_count, sizeof(xparticle_t), &m_xparticles_cap[alien_index]);

//     MPI_Recv(tmp, part_count*sizeof(xparticle_t), MPI_BYTE, rank, IRIS_TAG_FMM_P2P_HALO, m_local_comm->m_comm, MPI_STATUS_IGNORE);
    
//     cudaMemcpyAsync(m_xparticles[alien_index], tmp, part_count * sizeof(xparticle_t), cudaMemcpyDefault, m_streams[0]);
//     distribute_particles(m_xparticles[alien_index], part_count, alien_flag, m_xcells);
//     cudaFreeHost(tmp);
// }
