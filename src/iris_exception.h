// -*- c++ -*-
//==============================================================================
// Copyright (c) 2017-2018 NCSA
//
// See the README and LICENSE files in the top-level IRIS directory.
//==============================================================================
#ifndef __IRIS_EXCEPTION_H__
#define __IRIS_EXCEPTION_H__

#include <string>
#include <exception>

namespace ORG_NCSA_IRIS {

class iris_exception : public std::exception
{

public:
    std::string message;

    iris_exception(std::string msg) : message(msg) { };
    ~iris_exception() throw() { };
    
    virtual const char *what() const throw() {
	return message.c_str();
    };
};

}

#endif
