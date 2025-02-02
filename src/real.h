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
#ifndef __IRIS_REAL_H__
#define __IRIS_REAL_H__

#if IRIS_DOUBLE

typedef double iris_real;
#define IRIS_REAL MPI_DOUBLE
#define log_fn  log
#define fabs_fn fabs
#define sqrt_fn sqrt
#define pow_fn  pow
#define exp_fn  exp
#define atan_fn atan2
#define cos_fn cos
#define sin_fn sin
#define __rsqrt __drsqrt_rn
#define __fma   __fma_rn
#define iris_real3 double3
#define iris_real4 double4
#define make_iris_real3 make_double3
#define make_iris_real4 make_double4

#else

typedef float iris_real;
#define IRIS_REAL MPI_FLOAT
#define log_fn  logf
#define fabs_fn fabsf
#define sqrt_fn sqrtf
#define pow_fn  powf
#define exp_fn  expf
#define atan_fn atan2f
#define cos_fn cosf
#define sin_fn sinf
#define __rsqrt __frsqrt_rn
#define __fma   __fmaf_rn
#define iris_real3 float3
#define iris_real4 float4
#define make_iris_real3 make_float3
#define make_iris_real4 make_float4

#endif

#endif // __IRIS_REAL_H__
