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

#define IRIS_TAG_INTERCOMM_CREATE    1
#define IRIS_TAG_LOCAL_BOXES         2
#define IRIS_TAG_CHARGES             3
#define IRIS_TAG_CHARGES_ACK         4
#define IRIS_TAG_COMMIT_CHARGES      5
#define IRIS_TAG_QUIT                6
#define IRIS_TAG_RHO_HALO            7   // leave room for 6 halo msgs
#define IRIS_TAG_REMAP              13
#define IRIS_TAG_EX_HALO            14   // leave room for 6 halo msgs
#define IRIS_TAG_EY_HALO            20   // leave room for 6 halo msgs
#define IRIS_TAG_EZ_HALO            26   // leave room for 6 halo msgs
#define IRIS_TAG_FORCES             32
#define IRIS_TAG_GET_GLOBAL_ENERGY  33
#define IRIS_TAG_GLOBAL_ENERGY      34
#define IRIS_TAG_GX_HALO            35   // leave room for 6 halo msgs
#define IRIS_TAG_GY_HALO            41   // leave room for 6 halo msgs
#define IRIS_TAG_GZ_HALO            47   // leave room for 6 halo msgs
#define IRIS_TAG_PHI_HALO           53   // leave room for 6 halo msgs
#define IRIS_TAG_P_HALO             59   // leave room for 6 halo msgs
