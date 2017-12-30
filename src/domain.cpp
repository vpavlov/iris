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
#include <stdexcept>
#include "domain.h"
#include "iris.h"
#include "comm.h"

using namespace ORG_NCSA_IRIS;

domain::domain(iris *obj) : global_state(obj)
{
    iris_real default_box_min[] = {0.0, 0.0, 0.0};
    iris_real default_box_max[] = {1.0, 1.0, 1.0};

    dimensions = 3;
    this->set_box(default_box_min, default_box_max);
}

domain::~domain()
{
}

void domain::set_dimensions(int in_dimensions)
{
    if(in_dimensions != 2 && in_dimensions != 3) {
	throw std::invalid_argument("Unsupported number of dimensions!");
    }

    dimensions = in_dimensions;
}

void domain::set_box(iris_real in_box_min[3], iris_real in_box_max[3])
{
    if(in_box_min[0] >= in_box_max[0] ||
       in_box_min[1] >= in_box_max[1] ||
       in_box_min[2] >= in_box_max[2])
    {
	throw std::domain_error("Invalid bounding box!");
    }

    box_min[0] = in_box_min[0];
    box_min[1] = in_box_min[1];
    box_min[2] = in_box_min[2];

    box_max[0] = in_box_max[0];
    box_max[1] = in_box_max[1];
    box_max[2] = in_box_max[2];

    box_size[0] = box_max[0] - box_min[0];
    box_size[1] = box_max[1] - box_min[1];
    box_size[2] = box_max[2] - box_min[2];
}

void domain::setup_local_box()
{
    iris_real *xsplit = the_comm->xsplit;
    iris_real *ysplit = the_comm->ysplit;
    iris_real *zsplit = the_comm->zsplit;
    int *coords = the_comm->grid_coords;
    int *size = the_comm->grid_size;

    loc_box_min[0] = box_min[0] + box_size[0] * xsplit[coords[0]];
    if(coords[0] < size[0] - 1) {
	loc_box_max[0] = box_min[0] + box_size[0] * xsplit[coords[0] + 1];
    }else {
	loc_box_max[0] = box_max[0];
    }

    loc_box_min[1] = box_min[1] + box_size[1] * ysplit[coords[1]];
    if(coords[1] < size[1] - 1) {
	loc_box_max[1] = box_min[1] + box_size[1] * ysplit[coords[1] + 1];
    }else {
	loc_box_max[1] = box_max[1];
    }

    loc_box_min[2] = box_min[2] + box_size[2] * zsplit[coords[2]];
    if(coords[2] < size[2] - 1) {
	loc_box_max[2] = box_min[2] + box_size[2] * zsplit[coords[2] + 1];
    }else {
	loc_box_max[2] = box_max[2];
    }
}
