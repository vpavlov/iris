#
cmake_minimum_required(VERSION 3.1)

project(iris)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)  # we require C++ 11

set(IRIS_VERSION_MAJOR 0)
set(IRIS_VERSION_MINOR 1)
set(IRIS_VERSION "${IRIS_VERSION_MAJOR}.${IRIS_VERSION_MINOR}")
math(EXPR IRIS_VERSION_NUMERIC "${IRIS_VERSION_MAJOR}*10000 + ${IRIS_VERSION_MINOR}")

set(IRIS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(IRIS_LINK_LIBS)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})  # is this necessary?

file(GLOB IRIS_SOURCES ${IRIS_SOURCE_DIR}/*.cpp)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(IRIS_DATA_PRECISION "single" CACHE STRING "Data precision. Allowed values: single double")
set_property(CACHE IRIS_DATA_PRECISION PROPERTY STRINGS single double)

if(IRIS_DATA_PRECISION STREQUAL "single")
  set(IRIS_DATA_TYPE float)
  set(IRIS_IS_FLOAT 1)
  set(IRIS_IS_DOUBLE 0)
else()
  set(IRIS_DATA_TYPE double)
  set(IRIS_IS_FLOAT 0)
  set(IRIS_IS_DOUBLE 1)
endif()

add_definitions(-DIRIS_DATA_TYPE=${IRIS_DATA_TYPE})
add_definitions(-DIRIS_IS_FLOAT=${IRIS_IS_FLOAT})
add_definitions(-DIRIS_IS_DOUBLE=${IRIS_IS_DOUBLE})

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

include(CheckCCompilerFlag)
include(GNUInstallDirs)

find_package(MPI REQUIRED)
include_directories(${MPI_C_INCLUDE_PATH})
list(APPEND IRIS_LINK_LIBS ${MPI_CXX_LIBRARIES})

option(IRIS_OPENMP "Build OpenMP version" ON)
if(IRIS_OPENMP)
  find_package(OpenMP REQUIRED)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(IRIS_MEMALIGN "64" CACHE STRING "Memory alignment")
add_definitions(-DIRIS_MEMALIGN=${IRIS_MEMALIGN})

set(IRIS_SUFFIX "" CACHE STRING "Suffix to append to libiris")

option(CMAKE_VERBOSE_MAKEFILE "Verbose makefile" OFF)

###
### We need an FFT library...
###
set(FFT "FFTW3" CACHE STRING "FFT Library")
set_property(CACHE FFT PROPERTY STRINGS FFTW3 MKL)
find_package(${FFT} REQUIRED)
add_definitions(-DFFT_${FFT})
include_directories(${${FFT}_INCLUDE_DIRS})
list(APPEND IRIS_LINK_LIBS ${${FFT}_LIBRARIES})

###
### Needed standard libaries, etc.
###
include(CheckIncludeFile)
foreach(HEADER math.h)
  check_include_file(${HEADER} FOUND_${HEADER})
  if(NOT FOUND_${HEADER})
    message(FATAL_ERROR "Could not find needed header - ${HEADER}")
  endif(NOT FOUND_${HEADER})
endforeach(HEADER)

set(MATH_LIBRARIES "m" CACHE STRING "math library")
mark_as_advanced( MATH_LIBRARIES )
include(CheckLibraryExists)
foreach(FUNC sin cos)
  check_library_exists(${MATH_LIBRARIES} ${FUNC} "" FOUND_${FUNC}_${MATH_LIBRARIES})
  if(NOT FOUND_${FUNC}_${MATH_LIBRARIES})
    message(FATAL_ERROR "Could not find needed math function - ${FUNC}")
  endif(NOT FOUND_${FUNC}_${MATH_LIBRARIES})
endforeach(FUNC)
list(APPEND IRIS_LINK_LIBS ${MATH_LIBRARIES})
configure_file(config.h.in ${IRIS_SOURCE_DIR}/config.h)
include_directories(${IRIS_SOURCE_DIR})
add_library(iris ${IRIS_SOURCES})
target_link_libraries(iris ${IRIS_LINK_LIBS})
set_target_properties(iris PROPERTIES OUTPUT_NAME iris${IRIS_SUFFIX})

if(BUILD_SHARED_LIBS)
  set_target_properties(iris PROPERTIES SOVERSION ${IRIS_VERSION})
  configure_file(libiris.pc.in ${CMAKE_CURRENT_BINARY_DIR}/libiris${IRIS_SUFFIX}.pc @ONLY)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libiris${IRIS_SUFFIX}.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()

install(TARGETS iris LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${IRIS_SOURCE_DIR}/config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)
install(FILES ${IRIS_SOURCE_DIR}/api.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/iris)

string(TOUPPER "${CMAKE_BUILD_TYPE}" BTYPE)
message(STATUS "<<< Build configuration >>>
   Version          ${IRIS_VERSION_STRING}
   Build type       ${CMAKE_BUILD_TYPE}
   Precision        ${IRIS_DATA_PRECISION}
   Install path     ${CMAKE_INSTALL_PREFIX}
   Compilers and Flags:
   C++ Compiler     ${CMAKE_CXX_COMPILER}
       Type         ${CMAKE_CXX_COMPILER_ID}
   C++ Flags        ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BTYPE}}")
if(BUILD_SHARED_LIBS)
  message(STATUS "Shared libries  ${CMAKE_SHARED_LINKER_FLAGS}")
else()
  message(STATUS "Static libries  ${CMAKE_STATIC_LINKER_FLAGS}")
endif()
message(STATUS "Link libraries: ${IRIS_LINK_LIBS}")
