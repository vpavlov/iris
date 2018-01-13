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
#include <string.h>
#include <stdio.h>
#include "iris.h"
#include "mesh.h"
#include "proc_grid.h"
#include "memory.h"
#include "domain.h"
#include "logger.h"
#include "event_codes.h"
#include "event.h"

using namespace ORG_NCSA_IRIS;

mesh::mesh(iris *obj)
    :state_accessor(obj), m_size{0, 0, 0}, m_rho(NULL), m_dirty(true),
    m_initialized(false)
{
}

mesh::~mesh()
{
    if(m_rho != NULL) {
	memory::destroy_3d(m_rho);
    }
}

void mesh::set_size(int nx, int ny, int nz)
{
    if(nx <= 0 || ny <= 0 || nz <= 0) {
	throw std::invalid_argument("Invalid mesh size!");
    }

    m_size[0] = nx;
    m_size[1] = ny;
    m_size[2] = nz;

    m_initialized = true;
    m_dirty = true;
    m_logger->trace("Discretization mesh is %d x %d x %d",
		    m_size[0], m_size[1], m_size[2]);
}


void mesh::commit()
{
    if(!m_domain->m_initialized) {
	throw std::logic_error("mesh commit called, but domain is not initialized!");
    }

    if(!m_initialized) {
	throw std::logic_error("mesh commit called without size being initialized!");
    }

    if(m_dirty) {
	m_hinv[0] = m_size[0] / m_domain->m_global_box.xsize;
	m_hinv[1] = m_size[1] / m_domain->m_global_box.ysize;
	m_hinv[2] = m_size[2] / m_domain->m_global_box.zsize;
	m_h3inv = m_hinv[0] * m_hinv[1] * m_hinv[2];

	m_own_size[0] = m_size[0] / m_proc_grid->m_size[0];
	m_own_size[1] = m_size[1] / m_proc_grid->m_size[1];
	m_own_size[2] = m_size[2] / m_proc_grid->m_size[2];
	
	int *c = m_proc_grid->m_coords;
	m_own_offset[0] = c[0] * m_own_size[0];
	m_own_offset[1] = c[1] * m_own_size[1];
	m_own_offset[2] = c[2] * m_own_size[2];
	
	m_logger->trace("Local discretization mesh is %d x %d x %d starting at [%d, %d, %d]",
			m_own_size[0], m_own_size[1], m_own_size[2],
			m_own_offset[0], m_own_offset[1], m_own_offset[2]);
	m_dirty = false;
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
    for(int i=0;i<m_own_size[2];i++) {
	for(int j=0;j<m_own_size[1];j++) {
	    for(int k=0;k<m_own_size[0];k++) {
		fwrite(&(m_rho[k][j][i]), sizeof(iris_real), 1, fp);
	    }
	}
    }
    fclose(fp);
    
    // 2. write the bov header
    fp = fopen(header_fname, "w");
    fprintf(fp, "TIME: 1.23456\n");
    fprintf(fp, "DATA_FILE: %s\n", values_fname);
    fprintf(fp, "DATA_SIZE: %d %d %d\n", m_own_size[0], m_own_size[1], m_own_size[2]);
    if(sizeof(iris_real) == sizeof(double)) {
	fprintf(fp, "DATA_FORMAT: DOUBLE\n");
    }else {
	fprintf(fp, "DATA_FORMAT: FLOAT\n");
    }
    fprintf(fp, "VARIABLE: RHO\n");
    fprintf(fp, "DATA_ENDIAN: LITTLE\n");
    fprintf(fp, "CENTERING: zonal\n");
    fprintf(fp, "BRICK_ORIGIN: %f %f %f\n",
	    m_domain->m_local_box.xlo, m_domain->m_local_box.ylo, m_domain->m_local_box.zlo);
    fprintf(fp, "BRICK_SIZE: %f %f %f\n",
	    m_domain->m_local_box.xsize, m_domain->m_local_box.ysize, m_domain->m_local_box.zsize);
    fclose(fp);
}
