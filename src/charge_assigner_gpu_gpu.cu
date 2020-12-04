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
#include <stdexcept>
#include <vector>
#include <tuple>
#include "iris_gpu.h"
#include "charge_assigner_gpu.h"
#include "mesh_gpu.h"
#include "proc_grid_gpu.h"
#include "comm_rec_gpu.h"
#include "memory.h"
#include "domain_gpu.h"
#include "logger_gpu.h"
#include "event.h"
#include "openmp.h"
#include "cuda_parameters.h"

using namespace ORG_NCSA_IRIS;

static iris_real coeff2[2][2] =
    {{ 1/2.0,  1.0 },
     { 1/2.0, -1.0 }};

static iris_real coeff3[3][3] =
    {{ 1/8.0,  1/2.0,  1/2.0 },
     { 6/8.0,      0, -2/2.0 },
     { 1/8.0, -1/2.0,  1/2.0 }};

static iris_real coeff4[4][4] = 
    {{  1/48.0,  1/8.0,  1/4.0,  1/6.0 },
     { 23/48.0,  5/8.0, -1/4.0, -3/6.0 },
     { 23/48.0, -5/8.0, -1/4.0,  3/6.0 },
     {  1/48.0, -1/8.0,  1/4.0, -1/6.0 }};

static iris_real coeff5[5][5] =
    {{   1/384.0,   1/48.0,   1/16.0,  1/12.0,  1/24.0 },
     {  76/384.0,  22/48.0,   4/16.0, -2/12.0, -4/24.0 },
     { 230/384.0,        0, -10/16.0,       0,  6/24.0 },
     {  76/384.0, -22/48.0,   4/16.0,  2/12.0, -4/24.0 },
     {   1/384.0,  -1/48.0,   1/16.0, -1/12.0,  1/24.0 }};

static iris_real coeff6[6][6] =
    {{    1/3840.0,    1/384.0,   1/96.0,   1/48.0,  1/48.0,   1/120.0 },
     {  237/3840.0,   75/384.0,  21/96.0,   3/48.0, -3/48.0,  -5/120.0 },
     { 1682/3840.0,  154/384.0, -22/96.0, -14/48.0,  2/48.0,  10/120.0 },
     { 1682/3840.0, -154/384.0, -22/96.0,  14/48.0,  2/48.0, -10/120.0 },
     {  237/3840.0,  -75/384.0,  21/96.0,  -3/48.0, -3/48.0,   5/120.0 },
     {    1/3840.0,   -1/384.0,   1/96.0,  -1/48.0,  1/48.0,  -1/120.0 }};

static iris_real coeff7[7][7] =
    {{     1/46080.0,     1/3840.0,    1/768.0,   1/288.0,   1/192.0,  1/240.0,   1/720.0 },
     {   722/46080.0,   236/3840.0,   74/768.0,  20/288.0,   2/192.0, -4/240.0,  -6/720.0 },
     { 10543/46080.0,  1445/3840.0,   79/768.0, -43/288.0, -17/192.0,  5/240.0,  15/720.0 },
     { 23548/46080.0,            0, -308/768.0,         0,  28/192.0,        0, -20/720.0 },
     { 10543/46080.0, -1445/3840.0,   79/768.0,  43/288.0, -17/192.0, -5/240.0,  15/720.0 },
     {   722/46080.0,  -236/3840.0,   74/768.0, -20/288.0,   2/192.0,  4/240.0,  -6/720.0 },
     {     1/46080.0,    -1/3840.0,    1/768.0,  -1/288.0,   1/192.0, -1/240.0,   1/720.0 }};



static iris_real dcoeff2[2][1] =
    {{  1.0 },
     { -1.0 }};

static iris_real dcoeff3[3][2] =
    {{  1/2.0,  1.0 },
     {      0, -2.0 },
     { -1/2.0,  1.0 }};

static iris_real dcoeff4[4][3] = 
    {{  1/8.0,  1/2.0,  1/2.0 },
     {  5/8.0, -1/2.0, -3/2.0 },
     { -5/8.0, -1/2.0,  3/2.0 },
     { -1/8.0,  1/2.0, -1/2.0 }};

static iris_real dcoeff5[5][4] =
    {{   1/48.0,   1/8.0,  1/4.0,  1/6.0 },
     {  22/48.0,   4/8.0, -2/4.0, -4/6.0 },
     {        0, -10/8.0,       0,  6/6.0 },
     { -22/48.0,   4/8.0,  2/4.0, -4/6.0 },
     {  -1/48.0,   1/8.0, -1/4.0,  1/6.0 }};

static iris_real dcoeff6[6][5] =
    {{    1/384.0,   1/48.0,   1/16.0,  1/12.0,   1/24.0 },
     {   75/384.0,  21/48.0,   3/16.0, -3/12.0,  -5/24.0 },
     {  154/384.0, -22/48.0, -14/16.0,  2/12.0,  10/24.0 },
     { -154/384.0, -22/48.0,  14/16.0,  2/12.0, -10/24.0 },
     {  -75/384.0,  21/48.0,  -3/16.0, -3/12.0,   5/24.0 },
     {   -1/384.0,   1/48.0,  -1/16.0,  1/12.0,  -1/24.0 }};

static iris_real dcoeff7[7][6] =
    {{     1/3840.0,    1/384.0,   1/96.0,   1/48.0,  1/48.0,   1/120.0 },
     {   236/3840.0,   74/384.0,  20/96.0,   2/48.0, -4/48.0,  -6/120.0 },
     {  1445/3840.0,   79/384.0, -43/96.0, -17/48.0,  5/48.0,  15/120.0 },
     {            0, -308/384.0,        0,  28/48.0,       0, -20/120.0 },
     { -1445/3840.0,   79/384.0,  43/96.0, -17/48.0, -5/48.0,  15/120.0 },
     {  -236/3840.0,   74/384.0, -20/96.0,   2/48.0,  4/48.0,  -6/120.0 },
     {    -1/3840.0,    1/384.0,  -1/96.0,   1/48.0, -1/48.0,   1/120.0 }};

// Hockney and Eastwood modified Green function
//
// See for example Appendix A of
// "Comments on P3M, FMM, and the Ewald Method for Large Periodic Coulombic Systems"
// by E. L. Pollock and Jim Glosli

// Green's Function Denominator coefficients. They depend only on the order
// If you want higher order support, you can use lisp/gf_denom_coeff.lisp to
// calculate higher order coefficents and then ammend this code to include it.
static iris_real gfd_coeff1[] = { 1.0 };
static iris_real gfd_coeff2[] = { 1.0, -2.0/3 };
static iris_real gfd_coeff3[] = { 1.0, -1.0, 2.0/15 };
static iris_real gfd_coeff4[] = { 1.0, -4.0/3, 2.0/5, -4.0/315 };
static iris_real gfd_coeff5[] = { 1.0, -5.0/3, 7.0/9, -17.0/189, 2.0/2835 };
static iris_real gfd_coeff6[] = { 1.0, -2.0, 19.0/15, -256.0/945, 62.0/4725, -4.0/155925 };
static iris_real gfd_coeff7[] = { 1.0, -7.0/3, 28.0/15, -16.0/27, 26.0/405, -2.0/1485, 4.0/6081075 };

charge_assigner_gpu::charge_assigner_gpu(iris_gpu *obj)
    :state_accessor_gpu(obj), m_order(0), m_dirty(true), m_weights(NULL), m_dweights(NULL),
    m_coeff(NULL),m_dcoeff(NULL),m_gfd_coeff(NULL)
{
}

charge_assigner_gpu::~charge_assigner_gpu()
{
    memory_gpu::destroy_3d(m_weights);
    memory_gpu::destroy_3d(m_dweights);
}

void charge_assigner_gpu::set_order(int in_order)
{
    if(in_order < 2 || in_order > 7) {
	throw std::invalid_argument("Orders below 1 and above 7 are not supported!");
    }

    m_order = in_order;
    m_dirty = true;
    m_logger->trace("Charge assignment/interpolation order set to %d", m_order);
}

void charge_assigner_gpu::commit()
{
    // set default values
    if(m_order == 0) {
	set_order(1);
    }

    if(m_dirty) {
	m_ics_from = -(m_order-1)/2;
	m_ics_to = m_order/2;
	
	if(m_order % 2) {
	    m_ics_bump = 0.5;
	    m_ics_center = 0.0;
	}else {
	    m_ics_bump = 0.0;
	    m_ics_center = 0.5;
	}
	
    memory_gpu::create_1d(m_coeff,m_order*m_order);
    memory_gpu::create_1d(m_dcoeff,m_order*(m_order-1));
    memory_gpu::create_1d(m_gfd_coeff,m_order);

	if(m_order == 2) {
        cudaMemcpy(m_coeff,(iris_real *) coeff2,m_order*m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_dcoeff,(iris_real *) dcoeff2,m_order*(m_order-1)*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_gfd_coeff,(iris_real *) gfd_coeff2,m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
	}else if(m_order == 3) {
        cudaMemcpy(m_coeff,(iris_real *) coeff3,m_order*m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_dcoeff,(iris_real *) dcoeff3,m_order*(m_order-1)*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_gfd_coeff,(iris_real *) gfd_coeff3,m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        }else if(m_order == 4) {
        cudaMemcpy(m_coeff,(iris_real *) coeff4,m_order*m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_dcoeff,(iris_real *) dcoeff4,m_order*(m_order-1)*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_gfd_coeff,(iris_real *) gfd_coeff4,m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
	}else if(m_order == 5) {
	    cudaMemcpy(m_coeff,(iris_real *) coeff5,m_order*m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_dcoeff,(iris_real *) dcoeff5,m_order*(m_order-1)*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_gfd_coeff,(iris_real *) gfd_coeff5,m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
	}else if(m_order == 6) {
        cudaMemcpy(m_coeff,(iris_real *) coeff6,m_order*m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_dcoeff,(iris_real *) dcoeff6,m_order*(m_order-1)*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_gfd_coeff,(iris_real *) gfd_coeff6,m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
	}else if(m_order == 7) {
        cudaMemcpy(m_coeff,(iris_real *) coeff7,m_order*m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_dcoeff,(iris_real *) dcoeff7,m_order*(m_order-1)*sizeof(iris_real),
        cudaMemcpyHostToDevice);
        cudaMemcpy(m_gfd_coeff,(iris_real *) gfd_coeff7,m_order*sizeof(iris_real),
        cudaMemcpyHostToDevice);
	}

	memory_gpu::destroy_3d(m_weights);
	memory_gpu::create_3d(m_weights, m_iris->m_nthreads, 3, m_order);

	memory_gpu::destroy_3d(m_dweights);
	memory_gpu::create_3d(m_dweights, m_iris->m_nthreads, 3, m_order);
	
	if(m_mesh != NULL) {
	    m_mesh->m_dirty = true;
	}
	m_dirty = false;
    }
}

__device__
void compute_weights_dev(iris_real dx, iris_real dy, iris_real dz, 
                         iris_real* m_coeff, iris_real (&weights)[3][IRIS_MAX_ORDER], int order)
{
    iris_real r1, r2, r3;
    for(int i = 0; i < order; i++) {
        r1 = r2 = r3 = (iris_real)0.0;
        for(int j = order - 1; j >= 0; j--) {
            r1 = m_coeff[i*order + j] + r1 * dx;
            r2 = m_coeff[i*order + j] + r2 * dy;
            r3 = m_coeff[i*order + j] + r3 * dz;
        }
        weights[0][i] = r1;
        weights[1][i] = r2;
        weights[2][i] = r3;
    }
}

__global__
void compute_weights_kernel(iris_real dx, iris_real dy, iris_real dz,int tid, iris_real* m_coeff, iris_real*** m_weights, int m_order)
{
    auto i = IRIS_CUDA_INDEX(x);
    iris_real r1, r2, r3;

    r1 = r2 = r3 = (iris_real)0.0;

	
	for(int j = m_order - 1; j >= 0; j--) {
	    r1 = m_coeff[i*m_order + j] + r1 * dx;
	    r2 = m_coeff[i*m_order + j] + r2 * dy;
	    r3 = m_coeff[i*m_order + j] + r3 * dz;
	}
	
	m_weights[tid][0][i] = r1;
	m_weights[tid][1][i] = r2;
	m_weights[tid][2][i] = r3;
}

void charge_assigner_gpu::compute_weights(iris_real dx, iris_real dy, iris_real dz)
{
    int tid = THREAD_ID;
    compute_weights_kernel<<<1,m_order>>>(dx, dy, dz, tid, m_coeff, m_weights, m_order);
}

__global__
void compute_dweights_kernel(iris_real dx, iris_real dy, iris_real dz,int tid, iris_real* m_dcoeff, iris_real*** m_dweights, int m_order)
{
    auto i = IRIS_CUDA_INDEX(x);
    iris_real r1, r2, r3;

    r1 = r2 = r3 = (iris_real)0.0;
	
	for(int j = m_order - 2; j >= 0; j--) {
	    r1 = m_dcoeff[i*(m_order-1) + j] + r1 * dx;
	    r2 = m_dcoeff[i*(m_order-1) + j] + r2 * dy;
	    r3 = m_dcoeff[i*(m_order-1) + j] + r3 * dz;
	}
	
	m_dweights[tid][0][i] = r1;
	m_dweights[tid][1][i] = r2;
	m_dweights[tid][2][i] = r3;

}

void charge_assigner_gpu::compute_dweights(iris_real dx, iris_real dy, iris_real dz)
{
    int tid = THREAD_ID;
    compute_dweights_kernel<<<1,m_order>>>(dx, dy, dz, tid, m_dcoeff, m_dweights, m_order);
}
