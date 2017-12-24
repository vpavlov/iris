// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#ifndef __API_H__
#define __API_H__

#include "config.h"
#include <mpi.h>

typedef void *HIRIS;

// Result codes
#define IRIS_OK                                0
#define IRIS_RESULT_BAD_POISSON_SOLVER_METHOD -1
#define IRIS_RESULT_BAD_DD_CONF               -2
#define IRIS_RESULT_BAD_CONFIGURATION         -3

// Poisson solver method selection (with iris_select_poisson_solver_method)
#define IRIS_POISSON_PSM 1

#ifdef __cplusplus
extern "C" {
#endif

    HIRIS iris_open(MPI_Comm iris_comm);
    int iris_select_poisson_solver_method(HIRIS handle, int method);
    int iris_select_dd_conf(HIRIS handle, int x, int y, int z);
    int iris_apply_conf(HIRIS handle);
    void iris_close(HIRIS handle);
    
#ifdef __cplusplus
}
#endif

#endif  // __API_H__
