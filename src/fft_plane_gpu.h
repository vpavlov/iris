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
#ifndef __IRIS_GPU_FFT_PLANE_H__
#define __IRIS_GPU_FFT_PLANE_H__

#include "fft_base_gpu.h"

namespace ORG_NCSA_IRIS {

    class fft_plane_gpu : public fft_base_gpu {
	
    public:
	fft_plane_gpu(class iris_gpu *obj, const char *in_name, bool in_use_collective);
	virtual ~fft_plane_gpu();
	
	virtual iris_real *compute_fw(iris_real ***src, iris_real *dest);
	virtual void compute_bk(iris_real *src, iris_real ***dest);

	// virtual void compute_bk_remap_init(int i, iris_real *src, collective_fft3D_state& fftstate);
	// virtual void compute_bk_remap_pack(int i, iris_real *src, collective_fft3D_state& fftstate);
	// virtual void compute_bk_remap_communicate(int i, iris_real *src, collective_fft3D_state& fftstate);
	// virtual void compute_bk_remap_communicate1(int i, iris_real *src, collective_fft3D_state& fftstate);
	// virtual void compute_bk_remap_finalize(int i, iris_real *src, collective_fft3D_state& fftstate);
	// virtual void compute_bk_remap_finalize1(int i, iris_real *src, collective_fft3D_state& fftstate);
	
	// virtual void compute_bk_fft(int i, iris_real *src, collective_fft3D_state& fftstate);

	// virtual void compute_bk_final_init(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate);
	// virtual void compute_bk_final_pack(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate);
	// virtual void compute_bk_final_communicate(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate);
	// virtual void compute_bk_final_communicate1(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate);
	// virtual void compute_bk_final_finalize(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate);
	// virtual void compute_bk_final_finalize1(iris_real *src, iris_real ***dest, collective_fft3D_state& fftstate);
	
    private:

	void setup_remaps();
	void setup_plans();
	
#ifdef IRIS_CUDA
	FFTW_(plan) m_fw_plan1;
	FFTW_(plan) m_fw_plan2;
	FFTW_(plan) m_bk_plan1;
	FFTW_(plan) m_bk_plan2;
#endif

	class remap_gpu *m_fw_remap;  // forward: X[YZ] -> YZ[X]
	class remap_gpu *m_bk_remap;  // backward: YZ[X] -> X[YZ]
	
    };
}

#endif
