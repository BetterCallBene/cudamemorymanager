#
# Project
#
project(apsw_mod_trajectory_planning LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.5)
set(PROJECT_PREFIX "apsw")
set(MODULE_PREFIX "TraPla")
#
# Optimization
#
OPTION(OPTIMIZATION "Optimizing Trajectory Planning" ON)
if(NOT OPTIMIZATION)
    STRING(REPLACE "/O2" "/Od" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    set(CUDA_NVCC_FLAGS_DEBUG "--device-debug") 
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "--device-debug")
endif(OPTIMIZATION)

# Enable constexpr for WINDOOF
if (WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}--expt-relaxed-constexpr")
endif (WIN32)
#
# Common Settings
#
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_definitions(-DENABLE_FRIEND_TESTING)
if(WIN32)
add_definitions(-DNOMINMAX)
endif(WIN32)
#
# Support Conan.io
#
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()
#
# Import SEP framework
#
find_package(sep_framework REQUIRED)
#
# 3rdParty: GTest, OXF
#
find_package(GTest REQUIRED)
#
# Setup testing
#
include(CTest)
#
# Logger
#
OPTION(RTE_LOGGER "Using RTE Logger" OFF)
if(RTE_LOGGER)
    add_definitions(-DRTE_LOGGER)
endif(RTE_LOGGER)

OPTION(LOCAL_LOGGER_WINDOWS "Using Local Logger for Windows" OFF)
if(LOCAL_LOGGER_WINDOWS)
    add_definitions(-DLOCAL_LOGGER_WINDOWS)
endif(LOCAL_LOGGER_WINDOWS)

OPTION(LOCAL_LOGGER_LINUX "Using Local Logger for Linux" OFF)
if(LOCAL_LOGGER_LINUX)
    add_definitions(-DLOCAL_LOGGER_LINUX)
endif(LOCAL_LOGGER_LINUX)

OPTION(LOCAL_LOGGER "Using LOCAL Logger" OFF)
if(LOCAL_LOGGER)
    add_definitions(-DLOCAL_LOGGER)
endif(LOCAL_LOGGER)

OPTION(FILE_LOGGER "Using FILE Logger" OFF)
if(FILE_LOGGER)
    add_definitions(-DFILE_LOGGER)
endif(FILE_LOGGER)

OPTION(PROFILING "Do Profiling" OFF)
if(PROFILING)
    add_definitions(-DPROFILING)
endif(PROFILING)
#
# Libraries
#
add_subdirectory(libraries)
#
# Exports
#
add_subdirectory(exports)
#
# Tests
#
add_subdirectory(tests)
#
# Doxygen Documentation
#
option(DOCS "Generating Doxygen Documentation" OFF)
if(DOCS)
    add_subdirectory(docs)
endif(DOCS)

#
# install
#
install(
    FILES README.md
    DESTINATION .
)
