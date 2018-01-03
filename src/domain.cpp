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
#include "mesh.h"

using namespace ORG_NCSA_IRIS;

domain::domain(iris *obj) : global_state(obj)
{
    dimensions = 3;
    this->set_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0);

    for(int j=0;j<3;j++) {
	for(int i=0;i<2;i++) {
	    lbox_sides[i][j] = box_sides[i][j];
	}
	lbox_size[j] = box_size[j];
    }
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

void domain::set_box(iris_real x0, iris_real y0, iris_real z0,
		     iris_real x1, iris_real y1, iris_real z1)
{
    if(x0 >= x1 || y0 >= y1 || z0 >= z1)
    {
	throw std::domain_error("Invalid bounding box!");
    }

    box_sides[0][0] = x0;
    box_sides[0][1] = y0;
    box_sides[0][2] = z0;

    box_sides[1][0] = x1;
    box_sides[1][1] = y1;
    box_sides[1][2] = z1;

    box_size[0] = box_sides[1][0] - box_sides[0][0];
    box_size[1] = box_sides[1][1] - box_sides[0][1];
    box_size[2] = box_sides[1][2] - box_sides[0][2];

    if(the_mesh != NULL) {
	the_mesh->box_changed();
    }
}

void domain::setup_local()
{
    iris_real *xsplit = the_comm->xsplit;
    iris_real *ysplit = the_comm->ysplit;
    iris_real *zsplit = the_comm->zsplit;
    int *c = the_comm->grid_coords;
    int *size = the_comm->grid_size;

    lbox_sides[0][0] = box_sides[0][0] + box_size[0] * xsplit[c[0]];
    if(c[0] < size[0] - 1) {
	lbox_sides[1][0] = box_sides[0][0] + box_size[0] * xsplit[c[0] + 1];
    }else {
	lbox_sides[1][0] = box_sides[1][0];
    }

    lbox_sides[0][1] = box_sides[0][1] + box_size[1] * ysplit[c[1]];
    if(c[1] < size[1] - 1) {
	lbox_sides[1][1] = box_sides[0][1] + box_size[1] * ysplit[c[1] + 1];
    }else {
	lbox_sides[1][1] = box_sides[1][1];
    }

    lbox_sides[0][2] = box_sides[0][2] + box_size[2] * zsplit[c[2]];
    if(c[2] < size[2] - 1) {
	lbox_sides[1][2] = box_sides[0][2] + box_size[2] * zsplit[c[2] + 1];
    }else {
	lbox_sides[1][2] = box_sides[1][2];
    }

    lbox_size[0] = lbox_sides[1][0] - lbox_sides[0][0];
    lbox_size[1] = lbox_sides[1][1] - lbox_sides[0][1];
    lbox_size[2] = lbox_sides[1][2] - lbox_sides[0][2];
}
