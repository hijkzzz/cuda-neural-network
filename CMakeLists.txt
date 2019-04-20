cmake_minimum_required (VERSION 3.8)
project(neural_network LANGUAGES CXX CUDA)

# find cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61\'")

# find python
find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

# add source
file(GLOB_RECURSE HEADERS ./src/*.cuh ./src/*.h)
file(GLOB_RECURSE SOURCES ./src/*.cu ./src/*.cpp)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES})

include_directories(./src/*)

# add library
cuda_add_library(cu STATIC ${SOURCES})

# option
option(WRAP_LIB "wrap library" OFF)
option(UNIT_TEST "unit test" ON)

# wrap library
if(WRAP_LIB)
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