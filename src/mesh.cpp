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
#include <vector>
#include <tuple>
#include "mesh.h"
#include "comm.h"
#include "memory.h"
#include "domain.h"

using namespace ORG_NCSA_IRIS;

iris_real mesh::__2order[2][2] =
    {{ 1/2.0,  1.0 },
     { 1/2.0, -1.0 }};


iris_real mesh::__3order[3][3] =
	    {{ 1/8.0,  1/2.0,  1/2.0 },
	     { 6/8.0,      0, -2/2.0 },
	     { 1/8.0, -1/2.0,  1/2.0 }};

iris_real mesh::__4order[4][4] = 
	    {{  1/48.0,  1/8.0,  1/4.0,  1/6.0 },
	     { 23/48.0,  5/8.0, -1/4.0, -3/6.0 },
	     { 23/48.0, -5/8.0, -1/4.0,  3/6.0 },
	     {  1/48.0, -1/8.0,  1/4.0, -1/6.0 }};

iris_real mesh::__5order[5][5] = 
	    {{   1/384.0,   1/48.0,   1/16.0,  1/12.0,  1/24.0 },
	     {  76/384.0,  22/48.0,   4/16.0, -2/12.0, -4/24.0 },
	     { 230/384.0,        0, -10/16.0,       0,  6/24.0 },
	     {  76/384.0, -22/48.0,   4/16.0,  2/12.0, -4/24.0 },
	     {   1/384.0,  -1/48.0,   1/16.0, -1/12.0,  1/24.0 }};

iris_real mesh::__6order[6][6] = 
	    {{    1/3840.0,    1/384.0,   1/96.0,   1/48.0,  1/48.0,   1/120.0 },
	     {  237/3840.0,   75/384.0,  21/96.0,   3/48.0, -3/48.0,  -5/120.0 },
	     { 1682/3840.0,  154/384.0, -22/96.0, -14/48.0,  2/48.0,  10/120.0 },
	     { 1682/3840.0, -154/384.0, -22/96.0,  14/48.0,  2/48.0, -10/120.0 },
	     {  237/3840.0,  -75/384.0,  21/96.0,  -3/48.0, -3/48.0,   5/120.0 },
	     {    1/3840.0,   -1/384.0,   1/96.0,  -1/48.0,  1/48.0,  -1/120.0 }};

iris_real mesh::__7order[7][7] = 
	    {{     1/46080.0,     1/3840.0,    1/768.0,   1/288.0,   1/192.0,  1/240.0,   1/720.0 },
	     {   722/46080.0,   236/3840.0,   74/768.0,  20/288.0,   2/192.0, -4/240.0,  -6/720.0 },
	     { 10543/46080.0,  1445/3840.0,   79/768.0, -43/288.0, -17/192.0,  5/240.0,  15/720.0 },
	     { 23548/46080.0,            0, -308/768.0,         0,  28/192.0,        0, -20/720.0 },
	     { 10543/46080.0, -1445/3840.0,   79/768.0,  43/288.0, -17/192.0, -5/240.0,  15/720.0 },
	     {   722/46080.0,  -236/3840.0,   74/768.0, -20/288.0,   2/192.0,  4/240.0,  -6/720.0 },
	     {     1/46080.0,    -1/3840.0,    1/768.0,  -1/288.0,   1/192.0, -1/240.0,   1/720.0 }};

mesh::mesh(iris *obj) : global_state(obj)
{
    __ca_coeff = NULL;

    this->set_order(2);
    this->set_size(1, 1, 1);
    for(int i=0;i<3;i++) {
	lsize[i] = size[i];
	loffset[i] = 0;
    }

    memory::create_3d(rho, lsize[0], lsize[1], lsize[2]);

    for(int i=0;i<lsize[0];i++) {
	for(int j=0;j<lsize[1];j++) {
	    for(int k=0;k<lsize[2];k++) {
		rho[i][j][k] = (iris_real)0.0;
	    }
	}
    }
}

mesh::~mesh()
{
    memory::destroy_3d(rho);
}

void mesh::set_order(int in_order)
{
    if(in_order < 2 || in_order > 7) {
	throw std::invalid_argument("Orders below 2 and above 7 are not supported!");
    }

    order = in_order;
    __left = -(order-1)/2;
    __right = order/2;

    if(order % 2) {
	__center = 0.5;
	__shift1 = 0.0;
    }else {
	__center = 0.0;
	__shift1 = 0.5;
    }

    if(order == 2) {
	__rho_coeff = (iris_real *)__2order;
    }else if(order == 3) {
	__rho_coeff = (iris_real *)__3order;
    }else if(order == 4) {
	__rho_coeff = (iris_real *)__4order;
    }else if(order == 5) {
	__rho_coeff = (iris_real *)__5order;
    }else if(order == 6) {
	__rho_coeff = (iris_real *)__6order;
    }else if(order == 7) {
	__rho_coeff = (iris_real *)__7order;
    }

    if(__ca_coeff != NULL) {
	memory::destroy_2d(__ca_coeff);
    }
    memory::create_2d(__ca_coeff, 3, order);
}

void mesh::set_size(int nx, int ny, int nz)
{
    if(nx <= 0 || ny <= 0 || nz <= 0) {
	throw std::invalid_argument("Invalid mesh size!");
    }

    size[0] = nx;
    size[1] = ny;
    size[2] = nz;

    box_changed();
}

void mesh::box_changed()
{
    hinv[0] = size[0] / the_domain->box_size[0];  // 1/dx
    hinv[1] = size[1] / the_domain->box_size[1];  // 1/dy
    hinv[2] = size[2] / the_domain->box_size[2];  // 1/dz
    hinv3 = hinv[0] * hinv[1] * hinv[2];          // 1/dV
}

void mesh::setup_local()
{
    int *c = the_comm->grid_coords;

    lsize[0] = size[0] / the_comm->grid_size[0];
    lsize[1] = size[1] / the_comm->grid_size[1];
    lsize[2] = size[2] / the_comm->grid_size[2];

    loffset[0] = c[0] * lsize[0];
    loffset[1] = c[1] * lsize[1];
    loffset[2] = c[2] * lsize[2];

    memory::destroy_3d(rho);
    memory::create_3d(rho, lsize[0], lsize[1], lsize[2]);

    for(int i=0;i<lsize[0];i++) {
	for(int j=0;j<lsize[1];j++) {
	    for(int k=0;k<lsize[2];k++) {
		rho[i][j][k] = (iris_real)0.0;
	    }
	}
    }
}

void mesh::__compute_ca_coeff(iris_real dx, iris_real dy, iris_real dz)
{
    iris_real r1, r2, r3;

    for(int i = 0; i < order; i++) {
	r1 = r2 = r3 = (iris_real)0.0;
	
	for(int j = order - 1; j >= 0; j--) {
	    r1 = __rho_coeff[i*order + j] + r1 * dx;
	    r2 = __rho_coeff[i*order + j] + r2 * dy;
	    r3 = __rho_coeff[i*order + j] + r3 * dz;
	}
	
	__ca_coeff[0][i] = r1;
	__ca_coeff[1][i] = r2;
	__ca_coeff[2][i] = r3;
    }
}

void mesh::assign_charges(iris_real **atoms, int natoms)
{
    std::map<int, std::map<std::tuple<int, int, int>, iris_real>> outer;

    for(int i=0;i<natoms;i++) {
	iris_real tx = (atoms[i][0] - the_domain->lbox_sides[0][0]) * hinv[0];
	iris_real ty = (atoms[i][1] - the_domain->lbox_sides[0][1]) * hinv[1];
	iris_real tz = (atoms[i][2] - the_domain->lbox_sides[0][2]) * hinv[2];

	// the number of the cell that is to the "left" of the atom
	int nx = (int) (tx + __center);
	int ny = (int) (ty + __center);
	int nz = (int) (tz + __center);

	// distance (increasing to the left!) from the center of the interpolation grid
	iris_real dx = nx - tx + __shift1;
	iris_real dy = ny - ty + __shift1;
	iris_real dz = nz - tz + __shift1;

	// printf("%g %g %g %g | %d %d %d | %g %g %g\n",
	//        atoms[i][0], atoms[i][1], atoms[i][2], atoms[i][3],
	//        nx, ny, nz,
	//        dx, dy, dz);

	__compute_ca_coeff(dx, dy, dz);

	iris_real t0 = hinv3 * atoms[i][3];  // charge/volume
	for(int x = 0; x < order; x++) {
	    iris_real t1 = t0 * __ca_coeff[0][x];
	    for(int y = 0; y < order; y++) {
		iris_real t2 = t1 * __ca_coeff[1][y];
		for(int z = 0; z < order; z++) {
		    iris_real t3 = t2 * __ca_coeff[2][z];

		    int the_x = nx + x + __left;
		    int the_y = ny + y + __left;
		    int the_z = nz + z + __left;
		    
		    // figure out which neighbour we need to send this
		    // info to (if any)
		    int nidx = 0;
		    int ne_x = the_x;
		    int ne_y = the_y;
		    int ne_z = the_z;

		    if(the_x < 0) {
			nidx += 1;
			ne_x = lsize[0] + 1 + the_x;  // e.g. -1 becomes 128
		    }else if(the_x >= lsize[0]) {
			nidx += 2;
			ne_x = the_x - lsize[0];      // e.g. 128 becomes 0
		    }

		    if(the_y < 0) {
			nidx += 3;
			ne_y = lsize[1] + 1 + the_y;
		    }else if(the_y >= lsize[1]) {
			nidx += 6;
			ne_y = the_y - lsize[1];
		    }

		    if(the_z < 0) {
			nidx += 9;
			ne_z = lsize[2] + 1 + the_z;
		    }else if(the_z >= lsize[2]) {
			nidx += 18;
			ne_z = the_z - lsize[2];
		    }

		    if(the_comm->grid_hood[nidx] != the_comm->iris_rank) {
			std::tuple<int, int, int> entry = std::make_tuple(ne_x, ne_y, ne_z);
			outer[the_comm->grid_hood[nidx]][entry] += t3;
		    }else {
			rho[ne_x][ne_y][ne_z] += t3;
		    }
		}
	    }
	}
    }

    // send out halo elements
    for(auto it = outer.begin(); it != outer.end(); it++) {
    	for(auto it2 = it->second.begin(); it2 != it->second.end(); it2++) {
    	    printf("%d -> %d[%d %d %d]: %g\n",
		   the_comm->iris_rank,
    		   it->first,
    		   std::get<0>(it2->first),
    		   std::get<1>(it2->first),
    		   std::get<2>(it2->first),
    		   it2->second);
    	}
    }
}

void mesh::dump_rho(char *fname)
{
    char values_fname[256];
    char header_fname[256];
    
    strcpy(values_fname, fname);
    strcat(values_fname, ".bov");
    strcpy(header_fname, fname);
    strcat(header_fname, ".bovh");
    
    // 1. write the bov file
    FILE *fp = fopen(values_fname, "wb");
    fwrite(&(rho[0][0][0]), sizeof(iris_real), lsize[0] * lsize[1] * lsize[2], fp);
    fclose(fp);
    
    // 2. write the bov header
    fp = fopen(header_fname, "w");
    fprintf(fp, "TIME: 1.23456\n");
    fprintf(fp, "DATA_FILE: %s\n", values_fname);
    fprintf(fp, "DATA_SIZE: %d %d %d\n", lsize[0], lsize[1], lsize[2]);
    if(sizeof(iris_real) == sizeof(double)) {
	fprintf(fp, "DATA_FORMAT: DOUBLE\n");
    }else {
	fprintf(fp, "DATA_FORMAT: FLOAT\n");
    }
    fprintf(fp, "VARIABLE: RHO\n");
    fprintf(fp, "DATA_ENDIAN: LITTLE\n");
    fprintf(fp, "CENTERING: zonal\n");
    fprintf(fp, "BRICK_ORIGIN: %f %f %f\n",
	    the_domain->lbox_sides[0][0], the_domain->lbox_sides[0][1], the_domain->lbox_sides[0][2]);
    fprintf(fp, "BRICK_SIZE: %f %f %f\n",
	    the_domain->lbox_size[0], the_domain->lbox_size[1], the_domain->lbox_size[2]);
    fclose(fp);
}
