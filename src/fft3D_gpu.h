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
#ifndef __IRIS_FFT3D_H__
#define __IRIS_FFT3D_H__

#include "state_accessor.h"
#include "timer.h"

#ifdef FFT_FFTW

#include "cufftw.h"

#if IRIS_DOUBLE == 0

typedef fftwf_complex complex_t;
#define FFTW_(func)  fftwf_##func

#elif IRIS_DOUBLE == 1

typedef fftw_complex complex_t;
#define FFTW_(func)  fftw_##func

#else 

#error "Unknown IRIS_DOUBLE setting!"

#endif  // IRIS_DOUBLE

#endif  // FFT_FFTW


namespace ORG_NCSA_IRIS {

    class fft3d_gpu : protected state_accessor {

    public:
	fft3d_gpu(class iris *obj,
	      int *in_in_offset, int *in_in_size,
	      int *in_out_offset, int *in_out_size,
	      const char *in_name,
	      bool in_use_collective);

	~fft3d_gpu();

	iris_real *compute_fw(iris_real *src, iris_real *dest);
	void compute_bk(iris_real *src, iris_real *dest);

	void dump_workspace();

    private:
	void setup_grid(int in_which);
	void setup_remap(int in_which, bool in_use_collective);
	void setup_plans(int in_which);

    public:
	int m_count;               // number of items in own mesh

    private:
	const char *m_name;
	int m_in_size[3];
	int m_in_offset[3];
	int m_out_size[3];
	int m_out_offset[3];

	class grid *m_grids[3];    // proc grids in which 1 proc a whole dim
	int m_own_size[3][3];      // sizes for each of the grid
	int m_own_offset[3][3];    // offsets for each of the grid
	class remap_gpu *m_remaps[4];  // remaps between mesh->1d ffts->mesh
	iris_real *m_scratch;      // scratch space for remapping

	timer tm1[4], tm2;

#ifdef FFT_FFTW
	FFTW_(plan) m_fw_plans[3];
	FFTW_(plan) m_bk_plans[3];
#endif

    };

}

#endif
