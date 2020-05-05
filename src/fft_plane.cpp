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
#include "iris.h"
#include "fft_plane.h"
#include "logger.h"
#include "grid.h"
#include "remap.h"
#include "mesh.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

fft_plane::fft_plane(iris *obj, const char *in_name, bool in_use_collective)
    : fft_base(obj, in_name, in_use_collective)
{
    setup_remaps();
    setup_plans();
}

fft_plane::~fft_plane()
{
    delete m_remap1;
    delete m_remap2;
#ifdef FFT_FFTW
    FFTW_(destroy_plan)(m_forward_plan1);
    FFTW_(destroy_plan)(m_forward_plan2);
    FFTW_(destroy_plan)(m_backward_plan1);
    FFTW_(destroy_plan)(m_backward_plan2);
#endif
}

void fft_plane::setup_remaps()
{
    grid *tmp = new grid(m_iris, "FFT-101");
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
    
    m_remap1 = new remap(m_iris,
			m_mesh->m_own_offset,
			m_mesh->m_own_size,
			m_out_offset,
			m_out_size,
			2,
			2, "remap-fw",
			m_use_collective);


    // After forward FFT, the layout is YZX
    m_out_slow = 1;
    m_out_mid = 2;
    m_out_fast = 0;

    int tmp_offset1[3];
    int tmp_size1[3];
    int tmp_offset2[3];
    int tmp_size2[3];

    tmp_offset1[0] = m_out_offset[m_out_slow];
    tmp_offset1[1] = m_out_offset[m_out_mid];
    tmp_offset1[2] = m_out_offset[m_out_fast];
    tmp_size1[0] = m_out_size[m_out_slow];
    tmp_size1[1] = m_out_size[m_out_mid];
    tmp_size1[2] = m_out_size[m_out_fast];
    
    tmp_offset2[0] = m_mesh->m_own_offset[m_out_slow];
    tmp_offset2[1] = m_mesh->m_own_offset[m_out_mid];
    tmp_offset2[2] = m_mesh->m_own_offset[m_out_fast];
    tmp_size2[0] = m_mesh->m_own_size[m_out_slow];
    tmp_size2[1] = m_mesh->m_own_size[m_out_mid];
    tmp_size2[2] = m_mesh->m_own_size[m_out_fast];
    
    m_remap2 = new remap(m_iris,
			 tmp_offset1,
			 tmp_size1,
			 tmp_offset2,
			 tmp_size2,
			 2,
			 1, "remap-bk",
			 m_use_collective);

}

void fft_plane::setup_plans()
{
    int n[2];
    
    n[0] = m_mesh->m_own_size[1];  // YxZ arrays
    n[1] = m_mesh->m_own_size[2];

    int howmany = m_mesh->m_own_size[0];  // X times

    m_forward_plan1 =
	FFTW_(plan_many_dft)(2,          // 2D FFT
			     n,          // NxP arrays
			     howmany,    // M arrays
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

    m_backward_plan2 =
	FFTW_(plan_many_dft)(2,          // 2D FFT
			     n,          // NxP arrays
			     howmany,    // M arrays
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


    int nx = m_out_size[0];
    int howmany2 = m_out_size[1] * m_out_size[2];  // Y*Z times

    m_forward_plan2 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &nx,        // array of M elements
			     howmany2,   // NxP arrays
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,           // contiguous input
			     nx,         // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     nx,         // distance between arrays
			     FFTW_BACKWARD,
			     FFTW_ESTIMATE);

    

    m_backward_plan1 =
	FFTW_(plan_many_dft)(1,          // 1D FFT
			     &nx,        // array of M elements
			     howmany2,   // NxP arrays
			     NULL,       // input
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous input
			     nx,        // distance between arrays
			     NULL,       // output
			     NULL,       // same phisical as logical dimension
			     1,          // contiguous output
			     nx,         // distance between arrays
			     FFTW_FORWARD,
			     FFTW_ESTIMATE);
    
}

iris_real *fft_plane::compute_fw(iris_real *src, iris_real *dest)
{
    int j = 0;
    for(int i=0;i<m_count;i++) {
	dest[j++] = src[i];
	dest[j++] = 0.0;
    }

    FFTW_(execute_dft)(m_forward_plan1, (complex_t *)dest, (complex_t *)dest);
    m_remap1->perform(dest, dest, m_scratch);
    FFTW_(execute_dft)(m_forward_plan2, (complex_t *)dest, (complex_t *)dest);

    return dest;
    
}

void fft_plane::compute_bk(iris_real *src, iris_real *dest)
{
    FFTW_(execute_dft)(m_backward_plan1, (complex_t *)src, (complex_t *)src);
    m_remap2->perform(src, src, m_scratch);
    FFTW_(execute_dft)(m_backward_plan2, (complex_t *)src, (complex_t *)src);

    int j = 0;
    for(int i=0;i<m_count;i++) {
	dest[i] = src[j];
	j += 2;
    }
}
