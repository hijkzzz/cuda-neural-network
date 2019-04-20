cmake_minimum_required (VERSION 3.8)
project(neural_network LANGUAGES CXX CUDA)

# find cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61\'")

# find pybind11
add_subdirectory(./src/pybind11)
set(PYBIND11_CPP_STANDARD -std=c++14)

# add source
file(GLOB_RECURSE HEADERS ./src/cuda/*.cuh)
file(GLOB_RECURSE SOURCES ./src/cuda/*.cu)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES})

include_directories(./src/cuda)

# add library
cuda_add_library(cu STATIC ${SOURCES})

# option
option(WRAP_LIB "wrap library" OFF)
option(UNIT_TEST "unit test" ON)

# wrap library
if(WRAP_LIB)
	pybind11_add_module(neural_network ./src/python/bind.cpp)
	target_link_libraries(neural_network cu)
endif()

# unit test
if(UNIT_TEST)
	# tensor
	add_executable(tensor_test ./test/tensor_test.cpp)
	target_link_libraries(tensor_test cu)

	# neural network
	add_executable(nn_test ./test/nn_test.cpp)
	target_link_libraries(nn_test cu)

	# sgd
	add_executable(sgd_test ./test/sgd_test.cpp)
	target_link_libraries(sgd_test cu)

	# minist
	add_executable(minist_test ./test/minist_test.cpp)
	target_link_libraries(minist_test cu)
endif()