// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#ifndef __IRIS_H__
#define __IRIS_H__

#include <mpi.h>
#include "config.h"
#include "commrec.h"
#include "poisson_solver_base.h"

namespace ORG_NCSA_IRIS {

    class iris {

    public:
	
	iris(MPI_Comm comm);
	~iris();

	void select_poisson_solver_method(int method);
	void select_dd_conf(int x, int y, int z);
	void apply_conf();

    private:

	void select_dd_conf_auto();
	void eval_dd_conf(int *factors, int *powers, int count, int tx, int ty, int tz, int *bestx, int *besty, int *bestz, float *best_val);

	commrec *m_commrec;
	poisson_solver_base *m_poisson_solver;
	int m_ddx, m_ddy, m_ddz;  // domain decomposition in each dim
};

}

#endif
