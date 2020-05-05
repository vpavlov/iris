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
#ifndef __IRIS_FFT_PLANE_H__
#define __IRIS_FFT_PLANE_H__

#include "fft_base.h"

namespace ORG_NCSA_IRIS {

    class fft_plane : public fft_base {
	
    public:
	fft_plane(class iris *obj, const char *in_name, bool in_use_collective);
	virtual ~fft_plane();
	
	virtual iris_real *compute_fw(iris_real *src, iris_real *dest);
	virtual void compute_bk(iris_real *src, iris_real *dest);

    private:

	void setup_remaps();
	void setup_plans();
	
#ifdef FFT_FFTW
	FFTW_(plan) m_forward_plan1;  // 2D FFT plan for YZ transform
	FFTW_(plan) m_forward_plan2;  // 1D FFT plan for X transform
	FFTW_(plan) m_backward_plan1;  // 2D FFT plan for YZ transform
	FFTW_(plan) m_backward_plan2;  // 1D FFT plan for X transform
#endif
	class remap *m_remap1;
	class remap *m_remap2;
	
    };
}

#endif
