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
#define IRIS_TAG_RHO_HALO            7   // ..._RHO_HALOx must be consequitive
#define IRIS_TAG_RHO_HALO2           8   
#define IRIS_TAG_RHO_HALO3           9
#define IRIS_TAG_RHO_HALO4          10
#define IRIS_TAG_RHO_HALO5          11
#define IRIS_TAG_RHO_HALO6          12
#define IRIS_TAG_REMAP              13
#define IRIS_TAG_FIELD_HALO         14   // ..._FIELD_HALOx must be consequitive
#define IRIS_TAG_FIELD_HALO2        15   
#define IRIS_TAG_FIELD_HALO3        16
#define IRIS_TAG_FIELD_HALO4        17
#define IRIS_TAG_FIELD_HALO5        18
#define IRIS_TAG_FIELD_HALO6        19
#define IRIS_TAG_FORCES             20
#define IRIS_TAG_GET_GLOBAL_ENERGY  21
#define IRIS_TAG_GLOBAL_ENERGY      22
