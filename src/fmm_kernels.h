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
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//==============================================================================
#ifndef __IRIS_FMM_KERNELS_H__
#define __IRIS_FMM_KERNELS_H__

#ifdef IRIS_CUDA
#include <thrust/complex.h>
#else
#include <complex>
#endif
#include "real.h"

#ifndef IRIS_CUDA_DEVICE_HOST
#define IRIS_CUDA_DEVICE_HOST
#endif

#ifndef IRIS_CUDA_DEVICE
#define IRIS_CUDA_DEVICE
#endif

#ifndef IRIS_CUDA_HOST
#define IRIS_CUDA_HOST
#endif

#ifdef IRIS_CUDA
using namespace thrust;
#else
using namespace std;
#endif

namespace ORG_NCSA_IRIS {

    //
    // Indexing in a triangular array of mutipoles. It is "triangular", because
    // we only need e.g. m>0, since for m<0 we have Y_l^m = (-1)^m Y_l^m*
    // (* = complex conjugate)
    //
    // Each multipole entry is a complex number, this returns the index for real part
    //
    // Indexing goes like this:
    //
    // l = 0     : 
    //   m = 0   : 0 = real, 1 = imag
    // l = 1
    //   m = 0   : 2 = real, 3 = imag
    //   m = 1   : 4 = real, 5 = imag
    // l = 2
    //   m = 0   : 6 = real, 7 = imag
    //   m = 1   : 8 = real, 9 = imag
    //   m = 2   : 10 = real, 11 = imag
    // ...etc.
    //
    IRIS_CUDA_DEVICE_HOST int multipole_index(int l, int m);
    IRIS_CUDA_DEVICE_HOST void multipole_get(iris_real *M, int l, int m, iris_real *out_re, iris_real *out_im);
    IRIS_CUDA_DEVICE_HOST void multipole_add(iris_real *M, int l, int m, complex<iris_real> &val);
    
    IRIS_CUDA_DEVICE_HOST void p2m(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_M);
    IRIS_CUDA_DEVICE_HOST void p2l(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_L);
    void h_p2l(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *out_L);
    IRIS_CUDA_DEVICE_HOST void m2m(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M, iris_real *out_M, iris_real *in_scratch);
    IRIS_CUDA_DEVICE_HOST void m2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M, iris_real *out_L, iris_real *in_scratch);
    void h_m2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_M, iris_real *out_L, iris_real *in_scratch);
    IRIS_CUDA_DEVICE_HOST void l2l(int order, iris_real x, iris_real y, iris_real z, iris_real *in_L, iris_real *out_L, iris_real *in_scratch);
    IRIS_CUDA_DEVICE_HOST void l2p(int order, iris_real x, iris_real y, iris_real z, iris_real q, iris_real *in_L, iris_real *in_scratch,
				   iris_real *out_phi, iris_real *out_Ex, iris_real *out_Ey, iris_real *out_Ez);

}

#endif
