# -*- cmake -*-
#==============================================================================
# IRIS - Long-range Interaction Solver Library
#
# Copyright (c) 2017-2018, the National Center for Supercomputing Applications
#
# Primary authors:
#     Valentin Pavlov <vpavlov@rila.bg>
#     Peicho Petkov <peicho@phys.uni-sofia.bg>
#     Stoyan Markov <markov@acad.bg>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#==============================================================================
cmake_minimum_required(VERSION 2.8)

project(iris)
set(SOVERSION 0)

enable_language(CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(IRIS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(IRIS_LINK_LIBS)
set(INSTALLED_DEFINITIONS "")

file(GLOB IRIS_SOURCES ${IRIS_SOURCE_DIR}/*.cpp)
file(GLOB IRIS_INCLUDES ${IRIS_SOURCE_DIR}/*.h)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(IRIS_DOUBLE "Use double precision" OFF)

if(IRIS_DOUBLE)
  set(IRIS_DOUBLE_VALUE 1)
else()
  set(IRIS_DOUBLE_VALUE 0)
endif()
add_definitions(-DIRIS_DOUBLE=${IRIS_DOUBLE_VALUE})
list(APPEND INSTALLED_DEFINITIONS "-DIRIS_DOUBLE=${IRIS_DOUBLE_VALUE}")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

include(CheckCCompilerFlag)
include(GNUInstallDirs)

find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH})
list(APPEND IRIS_LINK_LIBS ${MPI_CXX_LIBRARIES})

################################################################################
# Find CUDA
################################################################################
option(IRIS_CUDA "Build CUDA version" ON)
if(IRIS_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DIRIS_CUDA)
  #set(CMAKE_CUDA_ARCHITECTURES  61)
  file(GLOB IRIS_SOURCES ${IRIS_SOURCE_DIR}/*.c*)
  include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
  list(REMOVE_ITEM IRIS_SOURCES "${IRIS_SOURCE_DIR}/ses.cpp")
  list(REMOVE_ITEM IRIS_SOURCES "${IRIS_SOURCE_DIR}/point.cpp")
  list(REMOVE_ITEM IRIS_SOURCES "${IRIS_SOURCE_DIR}/sphere.cpp")
  list(REMOVE_ITEM IRIS_SOURCES "${IRIS_SOURCE_DIR}/fmm_cell.cpp")
  list(REMOVE_ITEM IRIS_SOURCES "${IRIS_SOURCE_DIR}/fmm_kernels.cpp")
endif()

option(IRIS_OPENMP "Build OpenMP version" ON)
if(IRIS_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(IRIS_MEMALIGN "64" CACHE STRING "Memory alignment")
add_definitions(-DIRIS_MEMALIGN=${IRIS_MEMALIGN})
list(APPEND INSTALLED_DEFINITIONS "-DIRIS_MEMALIGN=${IRIS_MEMALIGN}")

set(IRIS_SUFFIX "" CACHE STRING "Suffix to append to libiris")

option(CMAKE_VERBOSE_MAKEFILE "Verbose makefile" OFF)

###
### We need an FFT library...
###
set(FFT "FFTW" CACHE STRING "FFT Library")
set_property(CACHE FFT PROPERTY STRINGS FFTW3 MKL)

if(IRIS_DOUBLE)
  if(IRIS_OPENMP)
    find_package(${FFT} REQUIRED COMPONENTS OMP)
  else()
    find_package(${FFT} REQUIRED)
  endif()
else()
  if(IRIS_OPENMP)
    find_package(${FFT} REQUIRED COMPONENTS OMP SIMPLE)
  else()
    find_package(${FFT} REQUIRED COMPONENTS SIMPLE)
  endif()
endif()


add_definitions(-DFFT_${FFT})
include_directories(${${FFT}_INCLUDE_DIRS})
list(APPEND IRIS_LINK_LIBS ${${FFT}_LIBRARIES})

###
### Needed standard libaries, etc.
###
include(CheckIncludeFile)
include(CheckFunctionExists)
include(CheckLibraryExists)

foreach(FUNC posix_memalign)
  check_function_exists(${FUNC} FOUND_${FUNC})
endforeach(FUNC)

list(APPEND IRIS_LINK_LIBS "m")

include_directories(${IRIS_SOURCE_DIR})
add_library(iris ${IRIS_SOURCES})
target_link_libraries(iris ${IRIS_LINK_LIBS})
set_target_properties(iris PROPERTIES OUTPUT_NAME iris${IRIS_SUFFIX})
if(IRIS_CUDA)
  set_target_properties(iris PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
if(BUILD_SHARED_LIBS)
  set_target_properties(iris PROPERTIES SOVERSION ${SOVERSION})
endif()

# add_executable(nacl ${IRIS_SOURCE_DIR}/../examples/nacl.cpp)
# target_link_libraries(nacl iris)

add_executable(bob ${IRIS_SOURCE_DIR}/../examples/bob.cpp)
target_link_libraries(bob iris)

add_executable(bob-ref ${IRIS_SOURCE_DIR}/../examples/bob-ref.cpp)
target_link_libraries(bob-ref iris)

if(IRIS_CUDA)
  cuda_add_executable(sort-test ${IRIS_SOURCE_DIR}/../examples/sort-test.cu)
  target_link_libraries(sort-test iris)
endif()

# add_executable(ses-test ${IRIS_SOURCE_DIR}/../examples/ses-test.cpp)
# target_link_libraries(ses-test iris)

# add_executable(poisson ${IRIS_SOURCE_DIR}/../examples/poisson.cpp)
# target_link_libraries(poisson iris)

# add_executable(stencils ${IRIS_SOURCE_DIR}/../examples/stencils.cpp)
# target_link_libraries(stencils iris)

# add_executable(gro-charges ${IRIS_SOURCE_DIR}/../examples/gro-charges.cpp)
# target_link_libraries(gro-charges iris)

# Handle pkgconfig
set(PKG_CFLAGS "")
foreach (_def ${INSTALLED_DEFINITIONS})
  set(PKG_CFLAGS "${PKG_CFLAGS} ${_def}")
endforeach()
configure_file(libiris.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libiris${IRIS_SUFFIX}.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libiris${IRIS_SUFFIX}.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig COMPONENT development)

install(TARGETS iris LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

foreach (_inc ${IRIS_INCLUDES})
  install(FILES ${_inc} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
endforeach()

string(TOUPPER "${CMAKE_BUILD_TYPE}" BTYPE)
