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
#include <algorithm>
#include <execution>
#include "fmm.h"
#include "fmm_cell.h"
#include "fmm_particle.h"
#include "fmm_kernels.h"
#include "fmm_pair.h"
#include "logger.h"
#include "domain.h"
#include "comm_rec.h"
#include "timer.h"
#include "openmp.h"
#include "proc_grid.h"
#include "tags.h"
#include "ses.h"
#include "utils.h"

#ifdef IRIS_CUDA
#include "cuda_runtime_api.h"
#include "cuda.h"
#endif

using namespace ORG_NCSA_IRIS;

#define _LN8 2.0794415416798357  // natural logarithm of 8

#define MIN_DEPTH 2
#define MAX_DEPTH 12

fmm::fmm(iris *obj):
    solver(obj), m_order(0), m_depth(0), m_mac(0.0), m_mac_let_corr(0.0), m_nterms(0),
    m_leaf_size{0.0, 0.0, 0.0}, m_local_root_level(0), 
    m_tree_size(0), m_cell_meta(NULL), m_M(NULL), m_L(NULL),
    m_cells(NULL), m_xcells(NULL), m_nparticles(0), m_npart_cap(0), m_particles(NULL),
    m_nxparticles(0), m_nxpart_cap(0), m_xparticles(NULL),
    m_dirty(true), m_sendcnt(NULL), m_senddisp(NULL), m_recvcnt(NULL), m_recvdisp(NULL),
    m_p2m_count(0), m_m2m_count(0), m_m2l_count(0), m_p2p_count(0),
    m_l2l_count(0), m_l2p_count(0), m_p2m_alien_count(0), m_m2m_alien_count(0),
    m_sendbuf(NULL), m_recvbuf(NULL), m_sendbuf_cap(0), m_recvbuf_cap(0)
#ifdef IRIS_CUDA
    ,m_atom_types(NULL), m_at_cap(0), m_keys(NULL), m_keys_cap(0),
    m_cells_cpu(NULL), m_xcells_cpu(NULL), m_M_cpu(NULL), m_recvbuf_gpu(NULL), m_recvbuf_gpu_cap(0),
    m_p2p_list_gpu(NULL), m_p2p_list_cap(0), m_m2l_list_gpu(NULL), m_m2l_list_cap(0),
    m_particles_cpu(NULL), m_particles_cpu_cap(0)
#endif
{
    int size = sizeof(box_t<iris_real>) * m_iris->m_server_size;
    m_ext_boxes = (box_t<iris_real> *)memory::wmalloc(size);

#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	for(int i=0;i<IRIS_CUDA_FMM_NUM_STREAMS;i++) {
	    cudaStreamCreate(&m_streams[i]);
	}
	cudaEventCreate(&m_m2l_memcpy_done);
	cudaEventCreate(&m_p2p_memcpy_done);
    }
    cudaDeviceSetLimit(cudaLimitStackSize, 32768);  // otherwise distribute_particles won't work because of the welzl recursion
    cudaMalloc((void **)&m_evir_gpu, 7*sizeof(iris_real));
    cudaMalloc((void **)&m_max_particles_gpu, sizeof(int));
    IRIS_CUDA_CHECK_ERROR;
#endif
    
}

fmm::~fmm()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	cudaFree(m_max_particles_gpu);
	cudaFree(m_evir_gpu);
	for(int i=0;i<IRIS_CUDA_FMM_NUM_STREAMS;i++) {
	    cudaStreamDestroy(m_streams[i]);
	}
	cudaEventDestroy(m_m2l_memcpy_done);
	cudaEventDestroy(m_p2p_memcpy_done);
	
	memory::wfree(m_ext_boxes);
	memory::destroy_1d_gpu(m_M);
	if(m_M_cpu != NULL) { memory::wfree_gpu(m_M_cpu); };
	memory::destroy_1d_gpu(m_L);
	if(m_cells != NULL) { memory::wfree_gpu(m_cells); }
	if(m_cells_cpu != NULL) { memory::wfree_gpu(m_cells_cpu); }
	if(m_xcells_cpu != NULL) { memory::wfree_gpu(m_xcells_cpu); }
	if(m_xcells != NULL) { memory::wfree_gpu(m_xcells); }
	if(m_particles != NULL) { memory::wfree_gpu(m_particles); }
	if(m_particles_cpu != NULL) { memory::wfree_gpu(m_particles_cpu, true); }
	// if(m_xparticles != NULL)  { memory::wfree_gpu(m_xparticles); }  // xparticles is at the end of particles
	for(auto it = m_charges_gpu.begin(); it != m_charges_gpu.end(); it++) {
	    memory::wfree_gpu(it->second);
	}
	if(m_atom_types != NULL) { memory::wfree_gpu(m_atom_types); }
	if(m_keys != NULL) { memory::wfree_gpu(m_keys); }
	if(m_recvbuf_gpu != NULL) { memory::wfree_gpu(m_recvbuf_gpu); }
	if(m_p2p_list_gpu != NULL) { memory::wfree_gpu(m_p2p_list_gpu); }
	if(m_m2l_list_gpu != NULL) { memory::wfree_gpu(m_m2l_list_gpu); }
    }else 
#endif
    {
	memory::wfree(m_ext_boxes);
	memory::destroy_1d(m_cell_meta);
	memory::destroy_1d(m_M);
	memory::destroy_1d(m_L);
	if(m_cells != NULL) { memory::wfree(m_cells); }
	if(m_xcells != NULL) { memory::wfree(m_xcells); }
	if(m_particles != NULL) { memory::wfree(m_particles); }
	if(m_xparticles != NULL)  { memory::wfree(m_xparticles); }
    }
    if(m_sendcnt != NULL) { memory::wfree(m_sendcnt); }
    if(m_senddisp != NULL) { memory::wfree(m_senddisp); }
    if(m_recvcnt != NULL) { memory::wfree(m_recvcnt); }
    if(m_recvdisp != NULL) { memory::wfree(m_recvdisp); }
    if(m_sendbuf != NULL) { memory::wfree(m_sendbuf); }
    if(m_recvbuf != NULL) { memory::wfree(m_recvbuf); }
}


void fmm::commit()
{
    if(m_dirty) {
	
	m_order = m_iris->m_order;  // if p = 2, we expand multipoles until Y_2^2
	if(m_order > IRIS_FMM_MAX_ORDER) {
	    throw std::logic_error("Max FMM order exceeded. Change IRIS_FMM_MAX_ORDER if you need to!");
	}
	int natoms = m_iris->m_natoms;  // atoms won't change during simulation (hopefully)
	solver_param_t t = m_iris->get_solver_param(IRIS_SOLVER_FMM_NCRIT);
	int ncrit = t.i;

	ncrit = MIN(ncrit, IRIS_MAX_NCRIT);

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_DEPTH);
	int user_depth = t.i;
	if(user_depth == -1) {
	    m_depth = (natoms > ncrit) ? int(log(natoms / ncrit)/_LN8) + 1 : 0;
	}else {
	    m_depth = user_depth;
	}
	m_depth = MAX(m_depth, MIN_DEPTH);
	m_depth = MIN(m_depth, MAX_DEPTH);

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC);
	m_mac = t.r;

	t = m_iris->get_solver_param(IRIS_SOLVER_FMM_MAC_CORR);
	m_mac_let_corr = t.r;
	
	m_nterms = (m_order + 1) * (m_order + 2) / 2;

	m_local_root_level = int(log(m_local_comm->m_size-1) / _LN8) + 1;
	if(m_local_comm->m_size == 1) {
	    m_local_root_level = 0;
	}

	m_tree_size = ((1 << 3 * m_depth) - 1) / 7;

	handle_box_resize();

#ifdef IRIS_CUDA
	if(m_iris->m_cuda) {
	    memory::destroy_1d_gpu(m_M);
	    memory::create_1d_gpu(m_M, m_tree_size*2*m_nterms);
	    if(m_M_cpu != NULL) { memory::wfree_gpu(m_M_cpu); };
	    m_M_cpu = (iris_real *)memory::wmalloc_gpu(m_tree_size*2*m_nterms*sizeof(iris_real), false, true);
	    memory::destroy_1d_gpu(m_L);
	    memory::create_1d_gpu(m_L, m_tree_size*2*m_nterms);

	    if(m_cells != NULL) { memory::wfree_gpu(m_cells); }
	    m_cells = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t));

	    if(m_cells_cpu != NULL) { memory::wfree_gpu(m_cells_cpu); }
	    m_cells_cpu = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t), false, true);

	    if(m_xcells_cpu != NULL) { memory::wfree_gpu(m_xcells_cpu); }
	    m_xcells_cpu = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t), false, true);
	    
	    if(m_xcells != NULL) { memory::wfree_gpu(m_xcells); }
	    m_xcells = (cell_t *)memory::wmalloc_gpu(m_tree_size * sizeof(cell_t));
	}else
#endif
	{
	    memory::destroy_1d(m_M);
	    memory::create_1d(m_M, m_tree_size*2*m_nterms);
	    memory::destroy_1d(m_L);
	    memory::create_1d(m_L, m_tree_size*2*m_nterms);
	    
	    memory::destroy_1d(m_cells);
	    memory::create_1d(m_cells, m_tree_size);
	    
	    memory::destroy_1d(m_xcells);
	    memory::create_1d(m_xcells, m_tree_size);
	}

	memory::destroy_1d(m_sendcnt);
	memory::create_1d(m_sendcnt, m_local_comm->m_size);
	
	memory::destroy_1d(m_senddisp);
	memory::create_1d(m_senddisp, m_local_comm->m_size);
	
	memory::destroy_1d(m_recvcnt);
	memory::create_1d(m_recvcnt, m_local_comm->m_size);
	
	memory::destroy_1d(m_recvdisp);
	memory::create_1d(m_recvdisp, m_local_comm->m_size);
	
	m_dirty = false;
	if(m_iris->m_cuda) {
	    m_logger->info("FMM/GPU: order = %d; depth = %d; tree size = %d; local root level = %d", m_order, m_depth, m_tree_size, m_local_root_level);
	}else {
	    m_logger->info("FMM: order = %d; depth = %d; tree size = %d; local root level = %d", m_order, m_depth, m_tree_size, m_local_root_level);
	}
    }
    calc_ext_boxes();  // this is needed every time
}

void fmm::generate_cell_meta()
{
    memory::destroy_1d(m_cell_meta);
    memory::create_1d(m_cell_meta, m_tree_size);
    for(int i=0;i<m_tree_size;i++) {
	m_cell_meta[i].set(m_cell_meta, i, &m_domain->m_global_box, m_leaf_size, max_level(), m_local_comm->m_size, m_local_root_level);
    }
}

void fmm::set_leaf_size()
{
    int nd = 1 << max_level();

    m_leaf_size[0] = m_domain->m_global_box.xsize / nd;
    m_leaf_size[1] = m_domain->m_global_box.ysize / nd;
    m_leaf_size[2] = m_domain->m_global_box.zsize / nd;
}

void fmm::handle_box_resize()
{
    set_leaf_size();
    // Cell meta is no longer needed, except for one-sided comm, which is not implemented yet (maybe never)
    generate_cell_meta();  
}

void fmm::compute_energy_and_virial()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	compute_energy_and_virial_gpu();
    }else
#endif
    {
	compute_energy_and_virial_cpu();
    }
}

void fmm::compute_energy_and_virial_cpu()
{
    iris_real ener = 0.0;
    for(int i=0;i<m_nparticles;i++) {
    	ener += m_particles[i].tgt[0] * m_particles[i].xyzq[3];
    }
    m_iris->m_Ek = ener * 0.5 * m_units->ecf;
    // TODO: calculate virial in m_iris->m_virial[0..5]
}

void fmm::send_forces_to(particle_t *in_particles, int peer, int start, int end, bool include_energy_virial)
{
    MPI_Comm comm = m_iris->client_comm();
    int ncharges = end - start;
    int size = 7*sizeof(iris_real) + ncharges*4*sizeof(iris_real);
    iris_real *forces = (iris_real *)memory::wmalloc(size);
    
    if(include_energy_virial) {
	forces[0] = m_iris->m_Ek;
	forces[1] = m_iris->m_virial[0];
	forces[2] = m_iris->m_virial[1];
	forces[3] = m_iris->m_virial[2];
	forces[4] = m_iris->m_virial[3];
	forces[5] = m_iris->m_virial[4];
	forces[6] = m_iris->m_virial[5];
    }else {
	forces[0] = 0.0;
	forces[1] = 0.0;
	forces[2] = 0.0;
	forces[3] = 0.0;
	forces[4] = 0.0;
	forces[5] = 0.0;
	forces[6] = 0.0;
    }
    
    m_iris->m_forces[peer] = forces;

    for(int i=start;i<end;i++) {
	iris_real factor = m_units->ecf;  // 1/4pieps

	forces[7 + (i-start)*4 + 0] = in_particles[i].index + 0.33333333;
	forces[7 + (i-start)*4 + 1] = factor * in_particles[i].tgt[1];
	forces[7 + (i-start)*4 + 2] = factor * in_particles[i].tgt[2];
	forces[7 + (i-start)*4 + 3] = factor * in_particles[i].tgt[3];
    }

    MPI_Request req;
    m_iris->send_event(comm, peer, IRIS_TAG_FORCES, size, forces, &req, NULL);
    MPI_Request_free(&req);
}

void fmm::send_back_forces()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	send_back_forces_gpu();
    }else
#endif
    {
	send_back_forces_cpu(m_particles, true);
    }
}

void fmm::send_back_forces_cpu(particle_t *in_particles, bool sort)
{
    bool include_energy_virial = true;  // send the energy and virial to only one of the clients; to the others send 0
    if(sort) {
	sort_back_particles(in_particles, m_nparticles);
    }
    int start = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++) {
	int end;
	for(end = start ; end<m_nparticles ; end++) {
	    if(in_particles[end].rank != rank) {
		break;
	    }
	}
	send_forces_to(in_particles, rank, start, end, include_energy_virial);
	include_energy_virial = false;
	start = end;
    }
}

void fmm::solve()
{    
    timer tm;
    m_logger->trace("FMM solve() start");

    tm.start();
    
    if(m_iris->m_compute_global_energy) {
	m_iris->m_Ek = 0.0;
    }

    if(m_iris->m_compute_global_virial) {
	m_iris->m_virial[0] =
	    m_iris->m_virial[1] =
	    m_iris->m_virial[2] =
	    m_iris->m_virial[3] =
	    m_iris->m_virial[4] =
	    m_iris->m_virial[5] = 0.0;
    }

    m_p2p_list.clear();
    m_m2l_list.clear();
    m_p2p_skip.clear();
    m_m2l_skip.clear();
    m_has_cells_cpu = false;
    
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	cudaMemsetAsync(m_max_particles_gpu, 0, sizeof(int), m_streams[0]);
       	cudaMemsetAsync(m_M, 0, m_tree_size*2*m_nterms*sizeof(iris_real), m_streams[1]);
	cudaMemsetAsync(m_L, 0, m_tree_size*2*m_nterms*sizeof(iris_real), m_streams[2]);
	cudaMemsetAsync(m_cells, 0, m_tree_size*sizeof(cell_t), m_streams[3]);
    }else
#endif
    {
	m_max_particles = 0;
	memset(m_M, 0, m_tree_size*2*m_nterms*sizeof(iris_real));
	memset(m_L, 0, m_tree_size*2*m_nterms*sizeof(iris_real));
	memset(m_cells, 0, m_tree_size*sizeof(cell_t));
    }
    
    m_p2m_count = m_m2m_count = m_m2l_count = m_p2p_count = m_l2l_count = m_l2p_count = m_p2m_alien_count = m_m2m_alien_count = 0;

    local_tree_construction();
    exchange_LET();    
    dual_tree_traversal();
    compute_energy_and_virial();
    send_back_forces();
    
    tm.stop();
    m_logger->info("FMM: Total step wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
    m_logger->info("FMM: TTS: %f ns/day (2 fs step)", 24*60*60/(tm.read_wall()*500000));
}

void fmm::local_tree_construction()
{
    load_particles();                                          // creates and sorts the m_particles array
    distribute_particles(m_particles, m_nparticles, IRIS_FMM_CELL_LOCAL, m_cells);  // distribute particles into leaf cells
    link_parents(m_cells);                                     // link parents and calculate parent's SES
    eval_p2m(m_cells, false);                                  // eval P2M for leaf nodes
    eval_m2m(m_cells, false);                                  // eval M2M for non-leaf nodes
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_p2p_self_gpu();
    }else
#endif
    {
	eval_p2p_self_cpu();
    }
    
// #ifdef IRIS_CUDA
//     if(m_iris->m_cuda) {
// 	print_tree_gpu("Cell", m_cells);
//     }else
// #endif
//     {
// 	print_tree("Cell", m_cells, 0, m_M);
//     }
}


////////////////////
// Load particles //
////////////////////


void fmm::load_particles()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	load_particles_gpu();
    }else 
#endif
    {
	load_particles_cpu();
    }
}

void h_load_charges(iris_real *charges, int ncharges, int hwm,
		    iris_real xlo, iris_real ylo, iris_real zlo,
		    iris_real lsx, iris_real lsy, iris_real lsz,
		    int max_level, int offset, particle_t *m_particles, int rank,
		    int nthreads)
{
    // timer tm;
    // tm.start();
    
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(ncharges, nthreads, &from, &to);
	for(int i=from;i<to;i++) {
	    iris_real tx = (charges[i * 5 + 0] - xlo) / lsx;
	    iris_real ty = (charges[i * 5 + 1] - ylo) / lsy;
	    iris_real tz = (charges[i * 5 + 2] - zlo) / lsz;
	    int chargeID = (int)charges[i*5 + 4];
	    
	    int cellID = cell_meta_t::leaf_coords_to_ID(tx, ty, tz, max_level);
	    
	    m_particles[i+hwm].xyzq[0] = charges[i*5+0];
	    m_particles[i+hwm].xyzq[1] = charges[i*5+1];
	    m_particles[i+hwm].xyzq[2] = charges[i*5+2];
	    m_particles[i+hwm].xyzq[3] = charges[i*5+3];
	    m_particles[i+hwm].cellID = cellID;
	    m_particles[i+hwm].rank = rank;
	    m_particles[i+hwm].index = chargeID;
	    m_particles[i+hwm].tgt[0] = 0.0;
	    m_particles[i+hwm].tgt[1] = 0.0;
	    m_particles[i+hwm].tgt[2] = 0.0;
	    m_particles[i+hwm].tgt[3] = 0.0;
	    m_particles[i+hwm].dummy[0] = (chargeID > 0)?0:1;
	}
    }

    // tm.stop();
    // printf("h_load_charges %g\n", tm.read_wall());
}

void fmm::load_particles_cpu()
{
    timer tm;
    tm.start();
    
    // Find the total amount of local charges, including halo atoms
    // This is just a simple sum of ncharges from all incoming rank
    int total_local_charges = 0;
    for(int rank = 0; rank < m_iris->m_client_size; rank++ ) {
	total_local_charges += m_iris->m_ncharges[rank];
    }

    m_particles = (particle_t *)memory::wmalloc_cap((void *)m_particles, total_local_charges, sizeof(particle_t), &m_npart_cap);

    int offset = cell_meta_t::offset_for_level(max_level());
    int nd = 1 << max_level();
    int hwm = 0;    
    for(int rank = 0; rank < m_iris->m_client_size; rank++) {
	int ncharges = m_iris->m_ncharges[rank];
	iris_real *charges = m_iris->m_charges[rank];
	h_load_charges(charges, ncharges, hwm,
		       m_domain->m_global_box.xlo, m_domain->m_global_box.ylo, m_domain->m_global_box.zlo,
		       m_leaf_size[0], m_leaf_size[1], m_leaf_size[2],
		       max_level(), offset, m_particles, rank,
		       m_iris->m_nthreads);
	hwm += ncharges;
    }
    
    // At this point we have the m_particles filled up, with mixed halo/own atoms, etc.
    // We need to sort them according to the atom type and split into m_particles and m_xparticles
    std::sort(m_particles, m_particles+total_local_charges, [](const particle_t &a, const particle_t &b) { return a.dummy[0] < b.dummy[0]; });

    // Now the first part of m_particles contains local atoms; second contains halo atoms
    // Number of local particles are those with key = 0
    // Number of halo particles is total_local_charges - num_local_atoms
    m_nparticles = std::count_if(m_particles, m_particles+total_local_charges, [](const particle_t &a) { return a.dummy[0] == 0; });
    m_nxparticles = total_local_charges - m_nparticles;
    m_xparticles = m_particles + m_nparticles;

    std::sort(m_particles, m_particles+m_nparticles, [](const particle_t &a, const particle_t &b) { return a.cellID < b.cellID; });
    std::sort(m_xparticles, m_xparticles+m_nxparticles, [](const particle_t &a, const particle_t &b) { return a.cellID < b.cellID; });

    tm.stop();
    m_logger->time("Load particles wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
    
    m_logger->info("FMM/CPU: This rank owns %d + %d halo particles", m_nparticles, m_nxparticles);
}


//////////////////////////
// Distribute particles //
//////////////////////////


void fmm::distribute_particles(particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	distribute_particles_gpu(in_particles, in_count, in_flags, out_target);
    }else 
#endif
    {
	distribute_particles_cpu(in_particles, in_count, in_flags, out_target);
    }
}

int __compare_cellID(const void *a, const void *b)
{
    particle_t *aptr = (particle_t *)a;
    particle_t *bptr = (particle_t *)b;
    if(aptr->cellID < bptr->cellID) {
	return -1;
    }else if(aptr->cellID > bptr->cellID) {
	return 1;
    }else {
	return 0;
    }
}

void fmm::distribute_particles_cpu(particle_t *in_particles, int in_count, int in_flags, struct cell_t *out_target)
{
    timer tm;
    tm.start();

    if(in_count == 0) {
	return;
    }
    
    box_t<iris_real> *gbox = &m_domain->m_global_box;
    int nleafs = (1 << 3 * max_level());
    int offset = cell_meta_t::offset_for_level(max_level());
    
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(nleafs, m_iris->m_nthreads, &from, &to);
	for(int i=from;i<to;i++) {
	    int cellID = offset + i;
	    particle_t key;
	    key.cellID = cellID;
	    particle_t *tmp = (particle_t *)bsearch(&key, in_particles, in_count, sizeof(particle_t), __compare_cellID);
	    if(tmp == NULL) {
		continue;
	    }

	    int left = tmp - in_particles;
	    int right = left;

	    while(left > 0 && in_particles[left].cellID >= cellID)             { left--; }
	    while(left < in_count && in_particles[left].cellID < cellID)       { left++; }
	    while(right < in_count-1 && in_particles[right].cellID <= cellID)  { right++; }
	    while(right >= 0 && in_particles[right].cellID > cellID)           { right--; }

	    int num_children = (right - left + 1);
	    if(num_children <= 0) {
		continue;
	    }
	    
	    out_target[cellID].first_child = left;
	    out_target[cellID].num_children = num_children;
	    out_target[cellID].flags = in_flags;
	    out_target[cellID].compute_com(in_particles);

#if defined _OPENMP
#pragma omp critical
	    {
#endif
		m_max_particles = MAX(m_max_particles, num_children);
#if defined _OPENMP
	    }
#endif
	}
    }
    
    tm.stop();
    m_logger->time("Distribute particles wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
};

void fmm::relink_parents(cell_t *io_cells)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	relink_parents_gpu(io_cells);
    }else
#endif
    {
	relink_parents_cpu(io_cells);
    }
}

void fmm::relink_parents_cpu(cell_t *io_cells)
{
    // first, clear the num_children and ses of all non-leaf cells
    int end = cell_meta_t::offset_for_level(max_level());
    for(int i=0;i<end;i++) {
	io_cells[i].flags &= ~IRIS_FMM_CELL_HAS_CHILDREN;
    }

    // second, for all shared cells (above local root level) recalculate ses
    end = cell_meta_t::offset_for_level(m_local_root_level);
    for(int i=0;i<end;i++) {
	io_cells[i].ses.r = 0.0;
    }

    // now, link all leafs upwards
    link_parents(io_cells);
}

void fmm::link_parents(cell_t *io_cells)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	link_parents_gpu(io_cells);
    }else
#endif
    {
	link_parents_cpu(io_cells);
    }
}

void fmm::link_parents_cpu(cell_t *io_cells)
{
    timer tm;
    tm.start();
    
    for(int i=max_level();i>0;i--) {
	int start = cell_meta_t::offset_for_level(i);
	int end = cell_meta_t::offset_for_level(i+1);
	int n = end - start;

#if defined _OPENMP
#pragma omp parallel
#endif
	{
	    int tid = THREAD_ID;
	    int from, to;
	    setup_work_sharing(n, m_iris->m_nthreads, &from, &to);
	    for(int j=start+from;j<start+to;j++) {
		if((io_cells[j].num_children != 0) ||                   // cell is a non-empty leaf
		   (io_cells[j].flags & IRIS_FMM_CELL_HAS_CHILDREN) ||  // or cell is a non-leaf and has some children
		   (io_cells[j].flags & IRIS_FMM_CELL_ALIEN_NL)) {        // or is an alien cell
		    int parent = cell_meta_t::parent_of(j);
#if defined _OPENMP
#pragma omp atomic update
#endif
		    io_cells[parent].flags += (IRIS_FMM_CELL_HAS_CHILD1 << ((j - start) % 8));
		}
	    }
	}
    }

    for(int i=max_level()-1;i>=0;i--) {
	int start = cell_meta_t::offset_for_level(i);
	int end = cell_meta_t::offset_for_level(i+1);
	int n = end - start;

#if defined _OPENMP
#pragma omp parallel
#endif
	{
	    int tid = THREAD_ID;
	    int from, to;
	    setup_work_sharing(n, m_iris->m_nthreads, &from, &to);
	
	    for(int j=start+from;j<start+to;j++) {
		if(io_cells[j].ses.r != 0.0) {
		    continue;
		}
		sphere_t S[8];
		int ns = 0;
		for(int k = 0;k<8;k++) {
		    int mask = IRIS_FMM_CELL_HAS_CHILD1 << k;
		    if(io_cells[j].flags & mask) {
			int childID = end + 8*(j-start) + k;
			S[ns].c.r[0] = io_cells[childID].ses.c.r[0];
			S[ns].c.r[1] = io_cells[childID].ses.c.r[1];
			S[ns].c.r[2] = io_cells[childID].ses.c.r[2];
			S[ns].r = io_cells[childID].ses.r;
			ns++;
		    }
		}
		ses_of_spheres(S, ns, &(io_cells[j].ses));
	    }
	}
    }

    tm.stop();
    m_logger->time("Link parents wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::eval_p2m(cell_t *in_cells, bool alien_only)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_p2m_gpu(in_cells, alien_only);
    }else
#endif
    {
	eval_p2m_cpu(in_cells, alien_only);
    }
}

void fmm::eval_p2m_cpu(cell_t *in_cells, bool alien_only)
{
    timer tm;
    tm.start();
    
    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(nleafs, m_iris->m_nthreads, &from, &to);
	for(int i=offset+from;i<offset+to;i++) {
	    cell_t *leaf = &in_cells[i];
	    
	    // no particles here -- continue
	    if(leaf->num_children == 0) {
		continue;
	    }
	    
	    // we only want alien cells, but this one is local -- continue
	    if(alien_only && !(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF)) {
		continue;
	    }
	    
	    // it has been send from exchange_LET AND from halo exchange -- continue
	    if(alien_only && (leaf->flags & IRIS_FMM_CELL_ALIEN_NL)) {
		continue;
	    }
	    for(int j=0;j<leaf->num_children;j++) {
		iris_real x, y, z, q;
		if(leaf->flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		    x = m_xparticles[leaf->first_child+j].xyzq[0] - in_cells[i].ses.c.r[0];
		    y = m_xparticles[leaf->first_child+j].xyzq[1] - in_cells[i].ses.c.r[1];
		    z = m_xparticles[leaf->first_child+j].xyzq[2] - in_cells[i].ses.c.r[2];
		    q = m_xparticles[leaf->first_child+j].xyzq[3];
		}else {
		    x = m_particles[leaf->first_child+j].xyzq[0] - in_cells[i].ses.c.r[0];
		    y = m_particles[leaf->first_child+j].xyzq[1] - in_cells[i].ses.c.r[1];
		    z = m_particles[leaf->first_child+j].xyzq[2] - in_cells[i].ses.c.r[2];
		    q = m_particles[leaf->first_child+j].xyzq[3];
		}
		p2m(m_order, x, y, z, q, m_M + i * 2 * m_nterms);
		in_cells[i].flags |= IRIS_FMM_CELL_VALID_M;
	    }
	}
    }
    
    tm.stop();
    m_logger->time("P2M wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::eval_m2m(cell_t *in_cells, bool invalid_only)
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_m2m_gpu(in_cells, invalid_only);
    }else
#endif
    {
	eval_m2m_cpu(in_cells, invalid_only);
    }
}

void h_eval_m2m(cell_t *in_cells, bool invalid_only, int offset, int children_offset, iris_real *m_M, int m_nterms, int m_order, int nthreads)
{
    int ntcells = children_offset - offset;
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+2)];
	int scratch_size = 2*m_nterms*sizeof(iris_real);
	
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(ntcells, nthreads, &from, &to);
	
	for(int i = from;i<to;i++) {
	    int tcellID = offset + i;
	    if(invalid_only && (in_cells[tcellID].flags & IRIS_FMM_CELL_VALID_M)) {
		continue;
	    }
	    
	    iris_real cx = in_cells[tcellID].ses.c.r[0];
	    iris_real cy = in_cells[tcellID].ses.c.r[1];
	    iris_real cz = in_cells[tcellID].ses.c.r[2];
	    
	    iris_real *M = m_M + tcellID * 2 * m_nterms;
	    
	    bool valid_m = false;
	    for(int j=0;j<8;j++) {
		int scellID = children_offset + 8*(tcellID-offset) + j;
		int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
		if(!(in_cells[tcellID].flags & mask)) {
		    continue;
		}
		iris_real x = in_cells[scellID].ses.c.r[0] - cx;
		iris_real y = in_cells[scellID].ses.c.r[1] - cy;
		iris_real z = in_cells[scellID].ses.c.r[2] - cz;
		memset(scratch, 0, scratch_size);
		m2m(m_order, x, y, z, m_M + scellID * 2 * m_nterms, M, scratch);
		valid_m = true;
	    }
	    if(valid_m) {
		in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_M;
	    }
	}
    }
}

void fmm::eval_m2m_cpu(cell_t *in_cells, bool invalid_only)
{
    timer tm;
    tm.start();
    
    int last_level = invalid_only ? 0 : m_local_root_level;
    for(int level = max_level()-1;level>=last_level;level--) {
	int start = cell_meta_t::offset_for_level(level);
	int end = cell_meta_t::offset_for_level(level+1);
	h_eval_m2m(in_cells, invalid_only, start, end, m_M, m_nterms, m_order, m_iris->m_nthreads);	
    }

    tm.stop();
    m_logger->time("M2M wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::exchange_LET()
{
    // timer tm, tm3;
    // tm.start();

#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	cudaMemcpyAsync(m_xcells, m_cells, m_tree_size * sizeof(cell_t), cudaMemcpyDefault, m_streams[0]);  // copy local tree to LET
    }else
#endif
    {
	memcpy(m_xcells, m_cells, m_tree_size * sizeof(cell_t));  // copy local tree to LET
    }
    
    if(m_local_comm->m_size > 1) {
	distribute_particles(m_xparticles, m_nxparticles, IRIS_FMM_CELL_ALIEN_LEAF, m_xcells);  // distribute particles into leaf cells
	comm_LET();
	recalculate_LET();
    }
    
// #ifdef IRIS_CUDA
//     if(m_iris->m_cuda) {
// 	print_tree_gpu("Xcell", m_xcells);
//     }else
// #endif
//     {
// 	print_tree("Xcell", m_xcells, 0, m_M);
//     }
    
   // tm.stop();
   // m_logger->info("FMM: Exchange LET Total wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());    
}

void fmm::comm_LET()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	int count = comm_LET_gpu();
	inhale_xcells_gpu(count);
    }else
#endif
    {
	int count = comm_LET_cpu(m_cells, m_M);
	inhale_xcells(count);
    }
}

void fmm::recalculate_LET()
{
    relink_parents(m_xcells);
    eval_p2m(m_xcells, true);
    eval_m2m(m_xcells, true);
}

#ifdef IRIS_CUDA
void fmm::print_tree_gpu(const char *label, cell_t *in_cells)
{
    cell_t *tmp = (cell_t *)memory::wmalloc(m_tree_size * sizeof(cell_t));
    iris_real *tmpM = (iris_real *)memory::wmalloc(m_tree_size * 2 * m_nterms * sizeof(iris_real));
    cudaMemcpy(tmp, in_cells, m_tree_size * sizeof(cell_t), cudaMemcpyDefault);
    cudaMemcpy(tmpM, m_M, m_tree_size * 2 * m_nterms * sizeof(iris_real), cudaMemcpyDefault);
    print_tree(label, tmp, 0, tmpM);
    memory::wfree(tmp);
    memory::wfree(tmpM);
}
#endif

void fmm::print_tree(const char *label, cell_t *in_cells, int cellID, iris_real *in_M)
{
    int level = cell_meta_t::level_of(cellID);
    if(level == max_level()) {
	m_logger->info("%*s%s %d (L%d) N=%d F=0x%x C=(%f,%f,%f) R=%f M[0] = %f", level+1, " ", label, cellID, level, in_cells[cellID].num_children, in_cells[cellID].flags,
		       in_cells[cellID].ses.c.r[0],
		       in_cells[cellID].ses.c.r[1],
		       in_cells[cellID].ses.c.r[2],
		       in_cells[cellID].ses.r,
		       in_M[cellID*2*m_nterms]);
    }else {
	int num_children = 0;
	int mask = IRIS_FMM_CELL_HAS_CHILD1;
	for(int i=0;i<8;i++) {
	    if(in_cells[cellID].flags & mask) {
		num_children++;
	    }
	    mask <<= 1;
	}
	
	m_logger->info("%*s%s %d (L%d) N=%d F=0x%x C=(%f,%f,%f) R=%f M[0] = %f", level+1, " ", label, cellID, level, num_children, in_cells[cellID].flags,
		       in_cells[cellID].ses.c.r[0],
		       in_cells[cellID].ses.c.r[1],
		       in_cells[cellID].ses.c.r[2],
		       in_cells[cellID].ses.r,
		       in_M[cellID*2*m_nterms]);
    }
    if(level < max_level()) {
	int this_offset = cell_meta_t::offset_for_level(level);
	int children_offset = cell_meta_t::offset_for_level(level+1);
	for(int j=0;j<8;j++) {
	    int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
	    if(in_cells[cellID].flags & mask) {
		int childID = children_offset + 8*(cellID-this_offset) + j;
		print_tree(label, in_cells, childID, in_M);
	    }
	}
    }
}

void fmm::dual_tree_traversal()
{
    // timer tm;
    // tm.start();
            
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	dual_tree_traversal_gpu();
	eval_p2p_gpu();
	eval_m2l_gpu();
	eval_l2l();
	eval_l2p();
    }else
#endif
    {
	dual_tree_traversal_cpu(m_xcells, m_cells);
	eval_p2p_cpu();
	eval_m2l_cpu();
	eval_l2l();
	eval_l2p();
    }

    // tm.stop();
    // m_logger->info("FMM: Dual Tree Traversal wall/cpu time %lf/%lf (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::dual_tree_traversal_gpu()
{
    if(!m_has_cells_cpu) {
	cudaMemcpyAsync(m_cells_cpu, m_cells, m_tree_size*sizeof(cell_t), cudaMemcpyDefault, m_streams[0]);
    }
    cudaMemcpyAsync(m_xcells_cpu, m_xcells, m_tree_size*sizeof(cell_t), cudaMemcpyDefault, m_streams[0]);
    cudaStreamSynchronize(m_streams[0]);
    dual_tree_traversal_cpu(m_xcells_cpu, m_cells_cpu);
}

void fmm::dual_tree_traversal_cpu(cell_t *src_cells, cell_t *dest_cells)
{
    assert(m_queue.empty());
    
    for(int ix = -m_proc_grid->m_pbc[0]; ix <= m_proc_grid->m_pbc[0]; ix++) {
	for(int iy = -m_proc_grid->m_pbc[1]; iy <= m_proc_grid->m_pbc[1]; iy++) {
	    for(int iz = -m_proc_grid->m_pbc[2]; iz <= m_proc_grid->m_pbc[2]; iz++) {
		pair_t root(0, 0);
		m_queue.push_back(root);
		traverse_queue(src_cells, dest_cells, ix, iy, iz);
	    }
	}
    }
}

void fmm::traverse_queue(cell_t *src_cells, cell_t *dest_cells, int ix, int iy, int iz)
{
    while(!m_queue.empty()) {
	pair_t pair = m_queue.front();
	m_queue.pop_front();

	int src_level = cell_meta_t::level_of(pair.sourceID);
	int tgt_level = cell_meta_t::level_of(pair.targetID);
	
	if((tgt_level == max_level()) || (src_level != max_level() && src_cells[pair.sourceID].ses.r > dest_cells[pair.targetID].ses.r)) {
	    cell_t *src = src_cells + pair.sourceID;
	    int level = cell_meta_t::level_of(pair.sourceID);
	    int this_offset = cell_meta_t::offset_for_level(level);
	    int children_offset = cell_meta_t::offset_for_level(level+1);
	    int mask = IRIS_FMM_CELL_HAS_CHILD1;
	    for(int i=0;i<8;i++) {
		if(src_cells[pair.sourceID].flags & mask) {
		    int childID = children_offset + (pair.sourceID - this_offset)*8 + i;
		    interact(src_cells, dest_cells, childID, pair.targetID, ix, iy, iz);
		}
		mask <<= 1;
	    }
	}else {
	    cell_t *target = dest_cells + pair.targetID;
	    int level = cell_meta_t::level_of(pair.targetID);
	    int this_offset = cell_meta_t::offset_for_level(level);
	    int children_offset = cell_meta_t::offset_for_level(level+1);
	    int mask = IRIS_FMM_CELL_HAS_CHILD1;
	    for(int i=0;i<8;i++) {
		if(dest_cells[pair.targetID].flags & mask) {
		    int childID = children_offset + (pair.targetID - this_offset)*8 + i;
		    interact(src_cells, dest_cells, pair.sourceID, childID, ix, iy, iz);
		}
		mask <<= 1;
	    }
	}
    }
}

void fmm::interact(cell_t *src_cells, cell_t *dest_cells, int srcID, int destID, int ix, int iy, int iz)
{
    iris_real src_cx = src_cells[srcID].ses.c.r[0] + ix * m_domain->m_global_box.xsize;
    iris_real src_cy = src_cells[srcID].ses.c.r[1] + iy * m_domain->m_global_box.ysize;
    iris_real src_cz = src_cells[srcID].ses.c.r[2] + iz * m_domain->m_global_box.zsize;

    iris_real dest_cx = dest_cells[destID].ses.c.r[0];
    iris_real dest_cy = dest_cells[destID].ses.c.r[1];
    iris_real dest_cz = dest_cells[destID].ses.c.r[2];

    iris_real dx = dest_cx - src_cx;
    iris_real dy = dest_cy - src_cy;
    iris_real dz = dest_cz - src_cz;
    
    iris_real rn = sqrt(dx*dx + dy*dy + dz*dz);
    iris_real dn = src_cells[srcID].ses.r + dest_cells[destID].ses.r;
    if(dn/rn < m_mac) {
	do_m2l_interact(srcID, destID, ix, iy, iz);
    }else if(cell_meta_t::level_of(srcID) == max_level() && cell_meta_t::level_of(destID) == max_level()) {
	do_p2p_interact(srcID, destID, ix, iy, iz);
    }else {
	pair_t pair(srcID, destID);
	m_queue.push_back(pair);
    }
}

#define M2L_CHUNK_SIZE 8192

void fmm::do_m2l_interact(int srcID, int destID, int ix, int iy, int iz)
{
    if(ix == 0 && iy == 0 && iz == 0) {
    	pair_t p(destID, srcID);
    	auto skip = m_m2l_skip.find(p);
    	if(skip != m_m2l_skip.end()) {
    	    return;
    	}
    	pair_t pp(srcID, destID);
    	m_m2l_skip[pp] = true;
    }
    
    interact_item_t t(srcID, destID, ix, iy, iz);
    m_m2l_list.push_back(t);

#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	if(m_m2l_list.size() >= M2L_CHUNK_SIZE) {
	    eval_m2l_gpu();
	}
    }
#endif
}

void fmm::do_p2p_interact(int srcID, int destID, int ix, int iy, int iz)
{
    // P2P self is handled separately, just ignore it here
    if(srcID == destID && ix == 0 && iy == 0 && iz == 0) {  
    	return;
    }

    if(m_proc_grid->m_pbc[0] != 0 && m_proc_grid->m_pbc[0] != 0 && m_proc_grid->m_pbc[0] != 0) {
	do_p2p_interact_pbc(srcID, destID, ix, iy, iz);
    }else if(m_proc_grid->m_pbc[0] == 0 && m_proc_grid->m_pbc[0] == 0 && m_proc_grid->m_pbc[0] == 0) {
	throw std::logic_error("Open boundary P2P not implemented!");
    }else {
	throw std::logic_error("Partial PBC P2P not implemented!");
    }
}

void fmm::do_p2p_interact_pbc(int srcID, int destID, int ix, int iy, int iz)
{
    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;

#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	pair_t p(destID, srcID);
	auto skip = m_p2p_skip.find(p);
	if(skip != m_p2p_skip.end()) {
	    return;
	}
	pair_t pp(srcID, destID);
	m_p2p_skip[pp] = true;
    }
#endif
    
    interact_item_t t(srcID, destID, ix, iy, iz);
    m_p2p_list.push_back(t);
    
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	if(m_p2p_list.size() >= nleafs) {
	    eval_p2p_gpu();
	}
    }
#endif
}

void fmm::eval_p2p_cpu()
{
    timer tm;
    tm.start();
    
    int n = m_p2p_list.size();
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(n, m_iris->m_nthreads, &from, &to);
	for(int i=from;i<to;i++) {
	    interact_item_t *item = &(m_p2p_list[i]);
	    eval_p2p(item->sourceID, item->targetID, item->ix, item->iy, item->iz);
	}
    }

    m_p2p_list.clear();
    
    tm.stop();
    m_logger->time("P2P Rest wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}


void fmm::eval_m2l_cpu()
{
    timer tm;
    tm.start();
    
    int n = m_m2l_list.size();
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(n, m_iris->m_nthreads, &from, &to);
	for(int i=from;i<to;i++) {
	    interact_item_t *item = &(m_m2l_list[i]);
	    eval_m2l(item->sourceID, item->targetID, item->ix, item->iy, item->iz);
	}
    }

    m_m2l_list.clear();
    
    tm.stop();
    m_logger->time("M2L wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::eval_m2l(int srcID, int destID, int ix, int iy, int iz)
{
    iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+2)];
    
    assert((m_xcells[srcID].flags & IRIS_FMM_CELL_VALID_M));

    iris_real sx = m_xcells[srcID].ses.c.r[0] + ix * m_domain->m_global_box.xsize;
    iris_real sy = m_xcells[srcID].ses.c.r[1] + iy * m_domain->m_global_box.ysize;
    iris_real sz = m_xcells[srcID].ses.c.r[2] + iz * m_domain->m_global_box.zsize;

    iris_real tx = m_cells[destID].ses.c.r[0];
    iris_real ty = m_cells[destID].ses.c.r[1];
    iris_real tz = m_cells[destID].ses.c.r[2];

    iris_real x = tx - sx;
    iris_real y = ty - sy;
    iris_real z = tz - sz;

    bool do_other_side = false;
    if(ix == 0 && iy == 0 && iz == 0) {
	do_other_side = true;
    }
    memset(scratch, 0, 2*m_nterms*sizeof(iris_real));
    h_m2l(m_order, x, y, z, m_M + srcID * 2 * m_nterms, m_L + destID * 2 * m_nterms, scratch,
	  m_M + destID * 2 * m_nterms, m_L + srcID * 2 * m_nterms, do_other_side);

#if defined _OPENMP
#pragma omp atomic
#endif
    m_cells[destID].flags |= IRIS_FMM_CELL_VALID_L;

    if(do_other_side) {
#if defined _OPENMP
#pragma omp atomic
#endif
	m_cells[srcID].flags |= IRIS_FMM_CELL_VALID_L;
    }
}

void fmm::eval_l2l()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_l2l_gpu();
    }else
#endif
    {
	eval_l2l_cpu();
    }
    
}

inline void __coulomb(iris_real tx, iris_real ty, iris_real tz, iris_real sx, iris_real sy, iris_real sz, iris_real sq,
		      iris_real &sum_phi, iris_real &sum_ex, iris_real &sum_ey, iris_real &sum_ez)
{
    iris_real dx = tx - sx;
    iris_real dy = ty - sy;
    iris_real dz = tz - sz;
    iris_real rlen = 1/sqrt(dx*dx + dy*dy + dz*dz);
    dx *= rlen * rlen;
    dy *= rlen * rlen;
    dz *= rlen * rlen;
    rlen *= sq;
    sum_phi += rlen;
    sum_ex += dx * rlen;
    sum_ey += dy * rlen;
    sum_ez += dz * rlen;
}

void fmm::eval_p2p_self_cpu()
{
    timer tm;
    tm.start();
    
    int offset = cell_meta_t::offset_for_level(max_level());

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(m_tree_size-offset, m_iris->m_nthreads, &from, &to);
	for(int cellID=offset+from;cellID<offset+to;cellID++) {
	    for(int i=0;i<m_cells[cellID].num_children;i++) {
		iris_real tx = m_particles[m_cells[cellID].first_child + i].xyzq[0];
		iris_real ty = m_particles[m_cells[cellID].first_child + i].xyzq[1];
		iris_real tz = m_particles[m_cells[cellID].first_child + i].xyzq[2];
		iris_real tq = m_particles[m_cells[cellID].first_child + i].xyzq[3];
		
		iris_real sum_phi = 0.0;
		iris_real sum_ex = 0.0;
		iris_real sum_ey = 0.0;
		iris_real sum_ez = 0.0;
		for(int j=0;j<m_cells[cellID].num_children;j++) {
		    if(i==j) {
			continue;
		    }
		    iris_real sx, sy, sz, sq;
		    sx = m_particles[m_cells[cellID].first_child + j].xyzq[0];
		    sy = m_particles[m_cells[cellID].first_child + j].xyzq[1];
		    sz = m_particles[m_cells[cellID].first_child + j].xyzq[2];
		    sq = m_particles[m_cells[cellID].first_child + j].xyzq[3];
		    __coulomb(tx, ty, tz, sx, sy, sz, sq, sum_phi, sum_ex, sum_ey, sum_ez);
		}
		
		m_particles[m_cells[cellID].first_child + i].tgt[0] += sum_phi;
		m_particles[m_cells[cellID].first_child + i].tgt[1] += tq*sum_ex;
		m_particles[m_cells[cellID].first_child + i].tgt[2] += tq*sum_ey;
		m_particles[m_cells[cellID].first_child + i].tgt[3] += tq*sum_ez;
	    }
	}
    }

    tm.stop();
    m_logger->time("P2P Self wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::eval_p2p(int srcID, int destID, int ix, int iy, int iz)
{
    for(int i=0;i<m_cells[destID].num_children;i++) {
	iris_real tx = m_particles[m_cells[destID].first_child + i].xyzq[0];
	iris_real ty = m_particles[m_cells[destID].first_child + i].xyzq[1];
	iris_real tz = m_particles[m_cells[destID].first_child + i].xyzq[2];
	iris_real tq = m_particles[m_cells[destID].first_child + i].xyzq[3];

	iris_real sum_phi = 0.0;
	iris_real sum_ex = 0.0;
	iris_real sum_ey = 0.0;
	iris_real sum_ez = 0.0;
	for(int j=0;j<m_xcells[srcID].num_children;j++) {
	    iris_real sx, sy, sz, sq;
	    if(m_xcells[srcID].flags & IRIS_FMM_CELL_ALIEN_LEAF) {
		sx = m_xparticles[m_xcells[srcID].first_child + j].xyzq[0] + ix * m_domain->m_global_box.xsize;
		sy = m_xparticles[m_xcells[srcID].first_child + j].xyzq[1] + iy * m_domain->m_global_box.ysize;
		sz = m_xparticles[m_xcells[srcID].first_child + j].xyzq[2] + iz * m_domain->m_global_box.zsize;
		sq = m_xparticles[m_xcells[srcID].first_child + j].xyzq[3];
	    }else {
		sx = m_particles[m_xcells[srcID].first_child + j].xyzq[0] + ix * m_domain->m_global_box.xsize;
		sy = m_particles[m_xcells[srcID].first_child + j].xyzq[1] + iy * m_domain->m_global_box.ysize;
		sz = m_particles[m_xcells[srcID].first_child + j].xyzq[2] + iz * m_domain->m_global_box.zsize;
		sq = m_particles[m_xcells[srcID].first_child + j].xyzq[3];
	    }
	    __coulomb(tx, ty, tz, sx, sy, sz, sq, sum_phi, sum_ex, sum_ey, sum_ez);
	}

#pragma omp atomic
	m_particles[m_cells[destID].first_child + i].tgt[0] += sum_phi;
#pragma omp atomic
	m_particles[m_cells[destID].first_child + i].tgt[1] += tq*sum_ex;
#pragma omp atomic
	m_particles[m_cells[destID].first_child + i].tgt[2] += tq*sum_ey;
#pragma omp atomic
	m_particles[m_cells[destID].first_child + i].tgt[3] += tq*sum_ez;
    }
}

void h_eval_l2l(cell_t *in_cells, int offset, int children_offset, iris_real *m_L, int m_nterms, int m_order, int nthreads)
{
    int nscells = children_offset - offset;
#if defined _OPENMP
#pragma omp parallel
#endif
    {
	iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+2)];
	int scratch_size = 2*m_nterms*sizeof(iris_real);
	
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(nscells, nthreads, &from, &to);
	for(int i = from;i<to;i++) {
	    int scellID = offset + i;
	    if(!(in_cells[scellID].flags & IRIS_FMM_CELL_VALID_L)) {
		continue;
	    }
	    
	    iris_real cx = in_cells[scellID].ses.c.r[0];
	    iris_real cy = in_cells[scellID].ses.c.r[1];
	    iris_real cz = in_cells[scellID].ses.c.r[2];

	    iris_real *L = m_L + scellID * 2 * m_nterms;
	    
	    for(int j=0;j<8;j++) {
		int tcellID = children_offset + 8*i + j;
		int mask = IRIS_FMM_CELL_HAS_CHILD1 << j;
		if(!(in_cells[scellID].flags & mask)) {
		    continue;
		}
		iris_real x = cx - in_cells[tcellID].ses.c.r[0];
		iris_real y = cy - in_cells[tcellID].ses.c.r[1];
		iris_real z = cz - in_cells[tcellID].ses.c.r[2];

		memset(scratch, 0, scratch_size);
		l2l(m_order, x, y, z, L, m_L + tcellID * 2 * m_nterms, scratch);
		in_cells[tcellID].flags |= IRIS_FMM_CELL_VALID_L;
	    }
	}
    }
}

void fmm::eval_l2l_cpu()
{
    timer tm;
    tm.start();
    
    for(int level = 0; level < m_depth-1; level++) {
	int start = cell_meta_t::offset_for_level(level);
	int end = cell_meta_t::offset_for_level(level+1);
	h_eval_l2l(m_cells, start, end, m_L, m_nterms, m_order, m_iris->m_nthreads);
    }

    tm.stop();
    m_logger->time("L2L wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::eval_l2p()
{
#ifdef IRIS_CUDA
    if(m_iris->m_cuda) {
	eval_l2p_gpu();
    }else
#endif
    {
	eval_l2p_cpu();
    }
}

void fmm::eval_l2p_cpu()
{
    timer tm;
    tm.start();

    int offset = cell_meta_t::offset_for_level(max_level());
    int nleafs = m_tree_size - offset;

#if defined _OPENMP
#pragma omp parallel
#endif
    {
	int tid = THREAD_ID;
	int from, to;
	setup_work_sharing(nleafs, m_iris->m_nthreads, &from, &to);
    
	iris_real scratch[(IRIS_FMM_MAX_ORDER+1) * (IRIS_FMM_MAX_ORDER+2)];

	for(int i=offset+from;i<offset+to;i++) {
	    cell_t *leaf = m_cells + i;
	    if(leaf->num_children == 0 || !(leaf->flags & IRIS_FMM_CELL_VALID_L)) {
		continue;
	    }
	    for(int j=0;j<leaf->num_children;j++) {
		iris_real x = leaf->ses.c.r[0] - m_particles[leaf->first_child+j].xyzq[0];
		iris_real y = leaf->ses.c.r[1] - m_particles[leaf->first_child+j].xyzq[1];
		iris_real z = leaf->ses.c.r[2] - m_particles[leaf->first_child+j].xyzq[2];
		iris_real q = m_particles[leaf->first_child+j].xyzq[3];
		iris_real phi, Ex, Ey, Ez;
		
		memset(scratch, 0, 2*m_nterms*sizeof(iris_real));
		l2p(m_order, x, y, z, q, m_L + i * 2 * m_nterms, scratch, &phi, &Ex, &Ey, &Ez);
		
		m_particles[leaf->first_child+j].tgt[0] += phi;
		m_particles[leaf->first_child+j].tgt[1] += Ex;
		m_particles[leaf->first_child+j].tgt[2] += Ey;
		m_particles[leaf->first_child+j].tgt[3] += Ez;
	    }
	}
    }

    tm.stop();
    m_logger->time("L2P wall/cpu time: %g/%g (%.2lf%% util)", tm.read_wall(), tm.read_cpu(), (tm.read_cpu() * 100.0) /tm.read_wall());
}

void fmm::calc_ext_boxes()
{
    iris_real R = sqrt(m_leaf_size[0]*m_leaf_size[0] + m_leaf_size[1]*m_leaf_size[1] + m_leaf_size[2]*m_leaf_size[2]);

    iris_real r_cut = R / m_mac;
    int nx = (int)(r_cut / m_leaf_size[0]);
    int ny = (int)(r_cut / m_leaf_size[1]);
    int nz = (int)(r_cut / m_leaf_size[2]);

    m_ext_box.xlo = m_domain->m_local_box.xlo - nx * m_leaf_size[0];
    m_ext_box.ylo = m_domain->m_local_box.ylo - ny * m_leaf_size[1];
    m_ext_box.zlo = m_domain->m_local_box.zlo - nz * m_leaf_size[2];
    
    m_ext_box.xhi = m_domain->m_local_box.xhi + nx * m_leaf_size[0];
    m_ext_box.yhi = m_domain->m_local_box.yhi + ny * m_leaf_size[1];
    m_ext_box.zhi = m_domain->m_local_box.zhi + nz * m_leaf_size[2];
    
    if(m_proc_grid->m_pbc[0] == 0) {
	m_ext_box.xlo = MAX(m_ext_box.xlo, 0.0); 
	m_ext_box.xhi = MIN(m_ext_box.xhi, m_domain->m_global_box.xhi);
    }
    
    if(m_proc_grid->m_pbc[1] == 0) {
	m_ext_box.ylo = MAX(m_ext_box.ylo, 0.0);
	m_ext_box.yhi = MIN(m_ext_box.yhi, m_domain->m_global_box.yhi);
    }

    if(m_proc_grid->m_pbc[2] == 0) {
	m_ext_box.zlo = MAX(m_ext_box.zlo, 0.0);
	m_ext_box.zhi = MIN(m_ext_box.zhi, m_domain->m_global_box.zhi);
    }

    
    m_ext_box.xsize = m_ext_box.xhi - m_ext_box.xlo;
    m_ext_box.ysize = m_ext_box.yhi - m_ext_box.ylo;
    m_ext_box.zsize = m_ext_box.zhi - m_ext_box.zlo;
    
    MPI_Allgather(&m_ext_box, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_ext_boxes, sizeof(box_t<iris_real>), MPI_BYTE,
		  m_local_comm->m_comm);

    m_logger->info("Extended box is %g x %g x %g: [%g:%g][%g:%g][%g:%g]",
		   m_ext_box.xsize, m_ext_box.ysize,
		   m_ext_box.zsize,
		   m_ext_box.xlo, m_ext_box.xhi,
		   m_ext_box.ylo, m_ext_box.yhi,
		   m_ext_box.zlo, m_ext_box.zhi);
}

box_t<iris_real> *fmm::get_ext_boxes()
{
    return m_ext_boxes;
}
