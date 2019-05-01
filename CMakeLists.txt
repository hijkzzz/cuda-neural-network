cmake_minimum_required (VERSION 3.8)
project(neural_network LANGUAGES CUDA CXX)

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd 4819 /FS")
endif()

# find cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_61,code=\"sm_61,compute_61\"")
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
endif()

# add source
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cuh)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu)
file(GLOB PYBIND11_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/python/*.cpp)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES} ${PYBIND11_SOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# add cuda library
add_library(cu STATIC ${SOURCES})
target_link_libraries(cu ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# option
option(WRAP_LIB "wrap library" OFF)
option(UNIT_TEST "unit test" ON)

# wrap library
if(WRAP_LIB)
    # find pybind11
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_parts/pybind11)
    include_directories(${PYBIND11_INCLUDE_DIR})
    if(CMAKE_COMPILER_IS_GNUCXX)
	    set(PYBIND11_CPP_STANDARD "-std=c++14")
    endif()

	pybind11_add_module(neural_network SHARED ${PYBIND11_SOURCES})
	target_link_libraries(neural_network cu)
endif()

# unit test
if(UNIT_TEST)
    # find google test
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_parts/googletest)

    file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda/*.cpp)
    add_executable(unit_tests ${TEST_SRCS})
    target_link_libraries(unit_tests cu gtest gtest_main)
    add_test(NAME unit_tests COMMAND unit_tests)
endif()