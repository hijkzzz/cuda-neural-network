cmake_minimum_required (VERSION 3.8)
project(neural_network LANGUAGES CUDA CXX)

# option
option(MINIST "minst demo" ON)
option(WRAP_LIB "pybind11" OFF)
option(UNIT_TEST "googletest" ON)

# find cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
 
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
endif()

if (MSVC)
    # warning
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/wd 4819\"")

    # static runtime
    # set(CompilerFlags
    #     CMAKE_CXX_FLAGS
    #     CMAKE_CXX_FLAGS_DEBUG
    #     CMAKE_CXX_FLAGS_RELEASE
    #    )
    # foreach(CompilerFlag ${CompilerFlags})
    #   string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    # endforeach()
endif()

# add cuda source
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cuh)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu)
file(GLOB PYBIND11_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/python/*.cpp)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES} ${PYBIND11_SOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# add cuda library
add_library(cu STATIC ${SOURCES})
target_link_libraries(cu PUBLIC cuda)

# minist demo
if (MINIST) 
    set(MINIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/minist)
    file(GLOB MINIST_SRCS ${MINIST_DIR}/*.cu)

    add_executable(minist ${MINIST_SRCS})
    target_include_directories(minist PRIVATE ${MINIST_DIR})
    target_link_libraries(minist cu)
endif()

# wrap library
if(WRAP_LIB)
    # find pybind11
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_parts/pybind11)
    include_directories(${PYBIND11_INCLUDE_DIR})
    if(CMAKE_COMPILER_IS_GNUCXX)
	    set(PYBIND11_CPP_STANDARD "-std=c++14")
    endif()

	pybind11_add_module(neural_network SHARED ${PYBIND11_SOURCES})
	target_link_libraries(neural_network PUBLIC cu)
endif()

# unit test
if(UNIT_TEST)
    # find google test
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_parts/googletest)

    set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda)
    file(GLOB TEST_SRCS ${TEST_DIR}/*.cu)

    add_executable(unit_tests ${TEST_SRCS})
    target_include_directories(unit_tests PRIVATE ${TEST_DIR})
    target_link_libraries(unit_tests cu gtest_main)

    add_test(NAME unit_tests COMMAND unit_tests)
endif()