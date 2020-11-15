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
#include "iris.h"
#include "fft3D_gpu.h"
#include "mesh.h"
#include "logger.h"
#include "grid.h"
#include "remap_gpu.h"
#include "comm_rec.h"
#include "memory.h"

using namespace ORG_NCSA_IRIS;

fft3d_gpu::fft3d_gpu(class iris *obj,
	     int *in_in_offset, int *in_in_size,
	     int *in_out_offset, int *in_out_size,
	     const char *in_name, bool in_use_collective)
    : state_accessor(obj), m_grids { NULL, NULL, NULL },
    m_own_size { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
    m_own_offset { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } },
    m_remaps { NULL, NULL, NULL, NULL },
    m_fw_plans { NULL, NULL, NULL },
    m_bk_plans { NULL, NULL, NULL },
    m_scratch(NULL),
    m_name(in_name)
{    
#if defined _OPENMP
#if defined FFT_FFTW
    FFTW_(init_threads);
    FFTW_(plan_with_nthreads(m_iris->m_nthreads));
#endif
#endif

    m_in_offset[0] = in_in_offset[0];
    m_in_offset[1] = in_in_offset[1];
    m_in_offset[2] = in_in_offset[2];

    m_in_size[0] = in_in_size[0];
    m_in_size[1] = in_in_size[1];
    m_in_size[2] = in_in_size[2];

    m_out_offset[0] = in_out_offset[0];
    m_out_offset[1] = in_out_offset[1];
    m_out_offset[2] = in_out_offset[2];

    m_out_size[0] = in_out_size[0];
    m_out_size[1] = in_out_size[1];
    m_out_size[2] = in_out_size[2];

    for(int i=0;i<3;i++) {
	setup_grid(i);
    }

    for(int i=0;i<4;i++) {
	setup_remap(i, in_use_collective);
    }

    for(int i=0;i<3;i++) {
	setup_plans(i);
    }

    // 2 *, because it contains complex numbers
    m_count = m_mesh->m_own_size[0] * m_mesh->m_own_size[1] *
	m_mesh->m_own_size[2];
    int n = 2 * m_count;

    memory_gpu::create_1d(m_scratch, n);
}

fft3d_gpu::~fft3d_gpu()
{
    // m_iris->m_logger->info("%s Remap0 wall/cpu time %lf/%lf (%.2lf%% util)", m_name, tm1[0].read_wall(), tm1[0].read_cpu(), (tm1[0].read_cpu() * 100.0) /tm1[0].read_wall());
    // m_iris->m_logger->info("%s Remap1 wall/cpu time %lf/%lf (%.2lf%% util)", m_name, tm1[1].read_wall(), tm1[1].read_cpu(), (tm1[1].read_cpu() * 100.0) /tm1[1].read_wall());
    // m_iris->m_logger->info("%s Remap2 wall/cpu time %lf/%lf (%.2lf%% util)", m_name, tm1[2].read_wall(), tm1[2].read_cpu(), (tm1[2].read_cpu() * 100.0) /tm1[2].read_wall());
    // m_iris->m_logger->info("%s Remap3 wall/cpu time %lf/%lf (%.2lf%% util)", m_name, tm1[3].read_wall(), tm1[3].read_cpu(), (tm1[3].read_cpu() * 100.0) /tm1[3].read_wall());
    // m_iris->m_logger->info("%s FFT wall/cpu time %lf/%lf (%.2lf%% util)", m_name, tm2.read_wall(), tm2.read_cpu(), (tm2.read_cpu() * 100.0) /tm2.read_wall());

    memory_gpu::destroy_1d(m_scratch);

    for(int i=0;i<3;i++) {
	if(m_grids[i] != NULL) { delete m_grids[i]; }
    }

    for(int i=0;i<4;i++) {
	if(m_remaps[i] != NULL) { delete m_remaps[i]; }
    }

    for(int i=0;i<3;i++) {
	if(m_fw_plans[i] != NULL) {
#ifdef FFT_FFTW
	    FFTW_(destroy_plan)(m_fw_plans[i]);
	    FFTW_(destroy_plan)(m_bk_plans[i]);
#endif
	}

    }

#ifdef FFT_FFTW
#ifdef _OPENMP
    FFTW_(cleanup_threads);
#else
    FFTW_(cleanup);
#endif
#endif

}

// To make all 1D FFTs local, we need to make sure that a proc owns
// an entire dimension (say X), hence the first 1 in the calls to
// set_user_pref below.
//
// If we can fit a whole XY (or XZ for that matter) plane, or several
// such planes, this is preferrable, because the FFTs in the other
// direction will also be local (there will be a local remap).
//
// Otherwise, just do a 2D decomposition, fixing X to 1. This leads to
// each proc owning a set of "pencils".
//
// We do this in turn for each dimension, so we have 3 different grids
void fft3d_gpu::setup_grid(int in_which)
{
    int last, xp1, yp1, zp1, xp2, yp2, zp2;

    const char *grid_name;
    switch(in_which) {
    case 0:
	grid_name = "FFT-Z";
	last = 1;
	zp1 = 1; yp1 = 1; xp1 = 0;
	zp2 = 1; yp2 = 0; xp2 = 0;
	break;
	
    case 1:
	grid_name = "FFT-Y";
	last = 0;
	yp1 = 1; zp1 = 0; xp1 = 0;
	yp2 = 1; zp2 = 0; xp2 = 0;
	break;

    case 2:
	grid_name = "FFT-X";
	last = 2;
	xp1 = 1; yp1 = 0; zp1 = 0;
	xp2 = 1; yp2 = 0; zp2 = 0;
	break;
    }

    m_grids[in_which] = new grid(m_iris, grid_name);

    if(m_mesh->m_size[last] >= m_local_comm->m_size) {
	m_grids[in_which]->set_pref(xp1, yp1, zp1);
    }else {
	// If there are more processors than mesh size in z, then make each
	// processor own a 2D sub-blocks of the YZ-plane
	m_grids[in_which]->set_pref(xp2, yp2, zp2);
    }
    
    m_grids[in_which]->commit();

    m_own_size[in_which][0] = m_mesh->m_size[0] / m_grids[in_which]->m_size[0];
    m_own_size[in_which][1] = m_mesh->m_size[1] / m_grids[in_which]->m_size[1];
    m_own_size[in_which][2] = m_mesh->m_size[2] / m_grids[in_which]->m_size[2];

    int *c = m_grids[in_which]->m_coords;
    m_own_offset[in_which][0] = c[0] * m_own_size[in_which][0];
    m_own_offset[in_which][1] = c[1] * m_own_size[in_which][1];
    m_own_offset[in_which][2] = c[2] * m_own_size[in_which][2];

    m_logger->trace("FFT brick %s is %d x %d x %d starting at [%d, %d, %d]",
		    grid_name,
		    m_own_size[in_which][0], m_own_size[in_which][1],
		    m_own_size[in_which][2],
		    m_own_offset[in_which][0], m_own_offset[in_which][1],
		    m_own_offset[in_which][2]);
}

// in_which:
//  = 0 for remap from m_mesh to m_grids[0]
//  = 1 for remap from m_grids[0] to m_grids[1]
//  = 2 for remap from m_grids[1] to m_grids[2]
//  = 3 for remap from m_grids[2] to m_grids[3]
//  = 4 for remap from m_grids[3] to m_mesh
void fft3d_gpu::setup_remap(int in_which, bool in_use_collective)
{
    char remap_name[256];
    switch(in_which) {
    case 0:  // XYZ -> XYZ
	sprintf(remap_name, "%s_remap0", m_name);
	m_remaps[in_which] = new remap_gpu(m_iris,
				       m_in_offset,  // from m_mesh
				       m_in_size,
				       m_own_offset[0],       // to m_grids[0]
				       m_own_size[0],
				       2,                     // complex
				       0, remap_name,         // no permutation
				       in_use_collective);
	break;

    case 1:  // XYZ -> ZXY
	sprintf(remap_name, "%s_remap1", m_name);
	m_remaps[in_which] = new remap_gpu(m_iris,
				       m_own_offset[0],       // from m_grids[0]
				       m_own_size[0],
				       m_own_offset[1],       // to m_grids[1]
				       m_own_size[1],
				       2,                     // complex
				       1, remap_name,         // x<-y<-z<-x
				       in_use_collective);
	break;

    case 2:  // ZXY -> YZX
	{
	    sprintf(remap_name, "%s_remap2", m_name);
	    int t_own_offset1[3];
	    int t_own_size1[3];
	    int t_own_offset2[3];
	    int t_own_size2[3];

	    t_own_offset1[0] = m_own_offset[1][2];
	    t_own_offset1[1] = m_own_offset[1][0];
	    t_own_offset1[2] = m_own_offset[1][1];
	    t_own_size1[0] = m_own_size[1][2];
	    t_own_size1[1] = m_own_size[1][0];
	    t_own_size1[2] = m_own_size[1][1];

	    t_own_offset2[0] = m_own_offset[2][2];
	    t_own_offset2[1] = m_own_offset[2][0];
	    t_own_offset2[2] = m_own_offset[2][1];
	    t_own_size2[0] = m_own_size[2][2];
	    t_own_size2[1] = m_own_size[2][0];
	    t_own_size2[2] = m_own_size[2][1];

	    m_remaps[in_which] = new remap_gpu(m_iris,
					   t_own_offset1,
					   t_own_size1,
					   t_own_offset2,
					   t_own_size2,
					   2,
					   1, remap_name,
					   in_use_collective);
	}
	break;

    case 3:  // ZXY -> XYZ
	{
	    sprintf(remap_name, "%s_remap3", m_name);
	    int t_own_offset1[3];
	    int t_own_size1[3];
	    int t_own_offset2[3];
	    int t_own_size2[3];

	    t_own_offset1[0] = m_own_offset[2][1];
	    t_own_offset1[1] = m_own_offset[2][2];
	    t_own_offset1[2] = m_own_offset[2][0];
	    t_own_size1[0] = m_own_size[2][1];
	    t_own_size1[1] = m_own_size[2][2];
	    t_own_size1[2] = m_own_size[2][0];

	    t_own_offset2[0] = m_out_offset[1];
	    t_own_offset2[1] = m_out_offset[2];
	    t_own_offset2[2] = m_out_offset[0];
	    t_own_size2[0] = m_out_size[1];
	    t_own_size2[1] = m_out_size[2];
	    t_own_size2[2] = m_out_size[0];

	    m_remaps[in_which] = new remap_gpu(m_iris,
					   t_own_offset1,
					   t_own_size1,
					   t_own_offset2,
					   t_own_size2,
					   2,
					   1, remap_name,
					   in_use_collective);
	}
	break;
    }

}

void fft3d_gpu::setup_plans(int in_which)
{
    int x, y, z;
    switch(in_which) {
    case 0:
	y = 0; z = 1; x = 2;
	break;

    case 1:
	z = 0; x = 1; y = 2;
	break;

    case 2:
	x = 0; y = 1; z = 2;
	break;

    }

    int n = m_mesh->m_size[x];
    int howmany = m_own_size[in_which][y] * m_own_size[in_which][z];
    
#ifdef FFT_FFTW
    m_fw_plans[in_which] = 
	FFTW_(plan_many_dft)(1,       // 1D
			     &n,      // M elements
			     howmany, // NxP times
			     NULL,    // input
			     NULL,    // same physical as logical dimension
			     1,       // contiguous input
			     n,       // distance to the next transform's data
			     NULL,    // output
			     NULL,    // same physical as logical dimension
			     1,       // contiguous output
			     n,       // distance to the next transform's data
			     FFTW_BACKWARD, // forward transformation; changed to backward to match lammps sign
			     FFTW_ESTIMATE);
    m_bk_plans[in_which] =
	FFTW_(plan_many_dft)(1,       // 1D
			     &n,      // M elements
			     howmany, // NxP times
			     NULL,    // input
			     NULL,    // same physical as logical dimension
			     1,       // contiguous input
			     n,       // distance to the next transform's data
			     NULL,    // output
			     NULL,    // same physical as logical dimension
			     1,       // contiguous output
			     n,       // distance to the next transform's data
			     FFTW_FORWARD, // backward transformation; changed to forward to match lammps sign
			     FFTW_ESTIMATE);
#endif

}

iris_real *fft3d_gpu::compute_fw(iris_real *src, iris_real *dest)
{
    // get data from the mesh
    int j = 0;
    for(int i=0;i<m_count;i++) {
	dest[j++] = src[i];
	dest[j++] = 0.0;
    }

    for(int i=0;i<3;i++) {
      tm1[i].start();
	m_remaps[i]->perform(dest, dest, m_scratch);
	tm1[i].stop();

	// m_logger->trace("AFTER REMAP %d", i);
	// for(int j=0;j<m_count;j++) {
	//     m_logger->trace("FFT[%d] = %.15g + j*%.15g",
	// 		    j, dest[j*2+0], dest[j*2+1]);
	// }

	tm2.start();
#ifdef FFT_FFTW
	FFTW_(execute_dft)(m_fw_plans[i],
			   (complex_t *)dest,
			   (complex_t *)dest);
#endif
	tm2.stop();
	// m_logger->trace("AFTER FFT %d", i);
	// for(int j=0;j<m_count;j++) {
	//     m_logger->trace("FFT[%d] = %.15g + j*%.15g",
	// 		    j, dest[j*2+0], dest[j*2+1]);
	// }

    }

    tm1[3].start();
    m_remaps[3]->perform(dest, dest, m_scratch);
    tm1[3].stop();

    // m_logger->trace("AFTER FINAL REMAP");
    // for(int j=0;j<m_count;j++) {
    // 	m_logger->trace("FFT[%d] = %.15g + j*%.15g",
    // 			j, dest[j*2+0], dest[j*2+1]);
    // }

    // now workspace contains 3D FFT of m_mesh->m_rho, in the original DD
    return dest;
}

void fft3d_gpu::compute_bk(iris_real *src, iris_real *dest)
{
    for(int i=0;i<3;i++) {
      tm1[i].start();
	m_remaps[i]->perform(src, src, m_scratch);
	tm1[i].stop();

	tm2.start();
#ifdef FFT_FFTW
	FFTW_(execute_dft)(m_bk_plans[i],
			   (complex_t *)src,
			   (complex_t *)src);
#endif
	tm2.stop();
    }

        tm1[3].start();
    m_remaps[3]->perform(src, src, m_scratch);
    	tm1[3].stop();

    // send data to the mesh
    int j = 0;
    for(int i=0;i<m_count;i++) {
	dest[i] = src[j];
	j += 2;
    }
}
