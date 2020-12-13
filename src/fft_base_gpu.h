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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_GPU_FFT_BASE_H__
#define __IRIS_GPU_FFT_BASE_H__

#include "state_accessor_gpu.h"
#include "timer.h"

#ifdef IRIS_CUDA

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

#endif  // IRIS_CUDA

namespace ORG_NCSA_IRIS {

    class fft_base_gpu : protected state_accessor_gpu {

    public:
	fft_base_gpu(class iris_gpu *obj, const char *in_name, bool in_use_collective);
	~fft_base_gpu();

	virtual iris_real *compute_fw(iris_real ***src, iris_real *dest) = 0;
	virtual void compute_bk(iris_real *src, iris_real ***dest) = 0;

	// virtual void compute_bk_remap_init(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_remap_pack(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_remap_communicate(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_remap_communicate1(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_remap_finalize(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_remap_finalize1(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;
	
	// virtual void compute_bk_fft(int i, iris_real *src, collective_fft3D_state& fftstate) = 0;

	// virtual void compute_bk_final_init(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_final_pack(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_final_communicate(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_final_communicate1(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_final_finalize(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate) = 0;
	// virtual void compute_bk_final_finalize1(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate) = 0;
	

	int get_count() { return m_count; };
	int *get_out_size() { return m_out_size; };
	int *get_out_offset() { return m_out_offset; };
	int get_slow() { return m_out_slow; };
	int get_mid() { return m_out_mid; };
	int get_fast() { return m_out_fast; };
	
    protected:
	const char *m_name;
	bool        m_use_collective;
	long        m_count;          // total number of items to do FFT on (own mesh)
	int         m_out_size[3];    // grid state after forward FFT
	int         m_out_offset[3];  // grid state after forward FFT
	int         m_out_slow;       // which (0=X, 1=Y, 2=Z) is the slow moving index (after FW FFT)
	int         m_out_mid;        // mid moving index
	int         m_out_fast;       // fast moving index
	iris_real * m_scratch;        // scratch space for remapping
    };
    
}

#endif
