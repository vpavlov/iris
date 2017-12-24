// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#ifndef __POISSON_SOLVER_PSM_H__
#define __POISSON_SOLVER_PSM_H__

#include "config.h"
#include "poisson_solver_base.h"

namespace ORG_NCSA_IRIS {

    class poisson_solver_psm : public poisson_solver_base {
	
    public:
	poisson_solver_psm();
	~poisson_solver_psm();

	virtual iris_real eval_dd_conf(int x, int y, int z);

    };

}
#endif
