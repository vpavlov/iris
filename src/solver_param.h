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
#ifndef __IRIS_SOLVER_PARAM_H__
#define __IRIS_SOLVER_PARAM_H__

union solver_param_t
{
    int i;
    iris_real r;
};

#define IRIS_SOLVER_CG_NSIGMAS         0  // total width of Gaussian
#define IRIS_SOLVER_CG_STENCIL_PADE_M  1  // nom order of Pade approximant
#define IRIS_SOLVER_CG_STENCIL_PADE_N  2  // denom order of Pade approximant
#define IRIS_SOLVER_P3M_USE_COLLECTIVE 3  // whether to use collective calls for the FFT3D remap
#define IRIS_SOLVER_FMM_NCRIT          4  // avg. number of particles in cell (determines max level)
#define IRIS_SOLVER_FMM_MAC            5  // FMM: Multipole Acceptance Criteria θ
#define IRIS_SOLVER_FMM_MAC_CORR       6  // FMM: Correction to θ for estimating distance for LET
#define IRIS_SOLVER_FMM_ONE_SIDED      7  // FMM: One-sided communication of LET
#define IRIS_SOLVER_PARAM_CNT          8  // number of different parameters

#endif
