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
#ifndef __IRIS_FFT_BASE_H__
#define __IRIS_FFT_BASE_H__

#include "state_accessor.h"
#include "timer.h"

#ifdef FFT_FFTW

#include "fftw3.h"

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

    class fft_base : protected state_accessor {

    public:
	fft_base(class iris *obj, const char *in_name, bool in_use_collective);
	~fft_base();

	virtual iris_real *compute_fw(iris_real *src, iris_real *dest) = 0;
	virtual void compute_bk(iris_real *src, iris_real *dest) = 0;

	int get_count() { return m_count; };
	
    protected:
	const char *m_name;
	bool m_use_collective;

	int m_count;               // total number of items to do FFT on (own mesh)
	
    };
    
}

#endif
