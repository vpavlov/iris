// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#include "api.h"
#include "iris.h"
#include "iris_exception.h"

using namespace ORG_NCSA_IRIS;

HIRIS iris_open(MPI_Comm communicator)
{
    HIRIS retval = NULL;

    try {
	retval = (HIRIS) new iris(communicator);
    }catch(iris_exception &e) {
	fprintf(stderr, "IRIS: exception during init: %s", e.message.c_str());
    }

    return retval;
}

int iris_select_poisson_solver_method(HIRIS handle, int method)
{
    iris *obj = (iris *)handle;
    try {
	obj->select_poisson_solver_method(method);
	return IRIS_OK;
    }catch(iris_exception &e) {
	return IRIS_RESULT_BAD_POISSON_SOLVER_METHOD;
    }
}

int iris_select_dd_conf(HIRIS handle, int x, int y, int z)
{
    iris *obj = (iris *)handle;
    try {
	obj->select_dd_conf(x, y, z);
	return IRIS_OK;
    }catch(iris_exception &e) {
	return IRIS_RESULT_BAD_DD_CONF;
    }
}

int iris_apply_conf(HIRIS handle)
{
    iris *obj = (iris *)handle;
    try {
	obj->apply_conf();
	return IRIS_OK;
    }catch(iris_exception &e) {
	return IRIS_RESULT_BAD_CONFIGURATION;
    }
}

void iris_close(HIRIS handle)
{
    iris *obj = (iris *)handle;
    delete obj;
}
