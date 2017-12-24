// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#include "api.h"
#include "iris.h"
#include "poisson_solver_psm.h"
#include "iris_exception.h"
#include "utils.h"

using namespace ORG_NCSA_IRIS;

iris::iris(MPI_Comm communicator)
{
    m_commrec = new commrec(communicator);
    m_poisson_solver = NULL;
    m_ddx = 0;
    m_ddy = 0;
    m_ddz = 0;
}

iris::~iris()
{
    delete m_commrec;

    if(m_poisson_solver != NULL) {
	delete m_poisson_solver;
    }
}

void iris::select_poisson_solver_method(int method)
{
    if(m_poisson_solver != NULL) {
	delete m_poisson_solver;
    }

    switch(method) {
    case IRIS_POISSON_PSM:
	m_poisson_solver = new poisson_solver_psm();
	break;

    default:
	throw new iris_exception("Unknown solver selected!");
    }
}

void iris::select_dd_conf(int x, int y, int z)
{
    if(x*y*z != m_commrec->get_size()) {
	throw new iris_exception("Bad domain decomposition configuration!");
    }
    m_ddx = x;
    m_ddy = y;
    m_ddz = z;
}

// Call this to apply all configuration, which will make the lib ready to
// perform calculations
void iris::apply_conf()
{
    // first, check if any of the configuration parameters was left
    // unititialized and initialized it to its default value
    if(m_poisson_solver == NULL) {
	select_poisson_solver_method(IRIS_POISSON_PSM);
    }

    if(m_ddx == 0 || m_ddy == 0 || m_ddz == 0) {
	select_dd_conf_auto();
    }
}


void iris::eval_dd_conf(int *factors, int *powers, int count,
			int tx, int ty, int tz,
			int *bestx, int *besty, int *bestz,
			float *best_val)
{
    if(count == 0) {
	float val = m_poisson_solver->eval_dd_conf(tx, ty, tz);
	if(val > *best_val) {
	    *bestx = tx;
	    *besty = ty;
	    *bestz = tz;
	    *best_val = val;
	}
	return;
    }

    for(int x = powers[0]; x>= 0; x--) {
	for(int i = 0; i < x; i++) {
	    tx *= factors[0];
	}

	for(int y = powers[0] - x; y >= 0; y--) {
	    for(int i = 0; i < y; i++) {
		ty *= factors[0];
	    }

	    for(int i = 0; i < powers[0]-x-y; i++) {
		tz *= factors[0];
	    }

	    eval_dd_conf(factors+1, powers+1, count-1, tx, ty, tz,
			 bestx, besty, bestz, best_val);

	    for(int i = 0; i < powers[0]-x-y; i++) {
		tz /= factors[0];
	    }

	    for(int i = 0; i < y; i++) {
		ty /= factors[0];
	    }
	}
        for (int i = 0; i < x; i++)
        {
	    tx /= factors[0];
        }
    }
}

void iris::select_dd_conf_auto()
{
    int *factors;
    int *powers;
    float best = 0.0;
    int count = factorize(m_commrec->get_size(), &factors, &powers);
    
    eval_dd_conf(factors, powers, count, 1, 1, 1,
		 &m_ddx, &m_ddy, &m_ddz, &best);
    
    delete factors;
    delete powers;
}
