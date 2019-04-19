cmake_minimum_required (VERSION 3.8)
project(neural_network LANGUAGES CXX CUDA)

# find cuda
find_package(CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -gencode arch=compute_61,code=sm_61")

# find swig
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# find python
find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

# add directories
file(GLOB HEADERS ./src/*.cuh ./src/*.h)
file(GLOB SOURCES ./src/*.cu ./src/*.cpp)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES})

# option
option(WRAP_LIB "wrap library" OFF)
option(UNIT_TEST "unit test" ON)

# add library
cuda_add_library(cu STATIC ${SOURCES} ${HEADERS})

# wrap library
if(WRAP_LIB)
    set_property(SOURCE ./src/neural_network.i PROPERTY CPLUSPLUS ON)
    swig_add_library(neural_network TYPE SHARED LANGUAGE python SOURCES ./src/neural_network.i)
    swig_link_libraries(neural_network cu ${PYTHON_LIBRARIES})
endif()

# unit test
if(UNIT_TEST)
	add_executable(tensor_test ./test/tensor_test.cpp)
	target_link_libraries(tensor_test cu)

	add_executable(conv_test ./test/conv_test.cpp)
	target_link_libraries(conv_test cu)

	add_executable(fc_test ./test/fc_test.cpp)
	target_link_libraries(fc_test cu)

	add_executable(pool_test ./test/pool_test.cpp)
	target_link_libraries(pool_test cu)

	add_executable(activation_test ./test/activation_test.cpp)
	target_link_libraries(activation_test cu)

	add_executable(minist_test ./test/minist_test.cpp)
	target_link_libraries(minist_test cu)
endif()