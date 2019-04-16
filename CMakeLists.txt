cmake_minimum_required (VERSION 3.0)
project(neural_network LANGUAGES CXX CUDA)

# find cuda
find_package(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_61,code=sm_61")

# add directories
file(GLOB HEADERS ./src/*.cuh)
file(GLOB SOURCES ./src/*.cu)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES})

# option
option(WRAP_LIB "wrap library" OFF)
option(UNIT_TEST "unit test" ON)

# wrap library
if(WRAP_LIB)

endif()

# unit test
if(UNIT_TEST)
	cuda_add_library(neural_network SHARED ${SOURCES} ${HEADERS})

	add_executable(minist_test ./test/minist_test.cpp)
	target_link_libraries(minist_test neural_network)
endif()