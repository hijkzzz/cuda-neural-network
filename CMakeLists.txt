cmake_minimum_required (VERSION 3.8)
project(neural_network CXX CUDA)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd 4819")
endif()

# find cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=\"sm_61,compute_61\"")
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
endif()

# add source
file(GLOB HEADERS ./src/cuda/*.cuh)
file(GLOB SOURCES ./src/cuda/*.cu)
file(GLOB PYBIND11_SOURCES ./src/python/*.cpp)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES} ${PYBIND11_SOURCES})

include_directories(./src/cuda)

# add cuda library
cuda_add_library(cu STATIC ${SOURCES})

# option
option(WRAP_LIB "wrap library" OFF)
option(UNIT_TEST "unit test" ON)

# wrap library
if(WRAP_LIB)
    # find pybind11
    add_subdirectory(./third_parts/pybind11)
    include_directories(${PYBIND11_INCLUDE_DIR})
    if(CMAKE_COMPILER_IS_GNUCXX)
	    set(PYBIND11_CPP_STANDARD "-std=c++14")
    endif()

	pybind11_add_module(neural_network MODULE ${PYBIND11_SOURCES})
	target_link_libraries(neural_network cu)
endif()

# unit test
if(UNIT_TEST)
    # find google test
    add_subdirectory(./third_parts/googletest)
    add_subdirectory(./test/cuda)
endif()