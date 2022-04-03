cmake_minimum_required (VERSION 3.8)
project(neural_network LANGUAGES CUDA CXX)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-DDEBUG)
endif()

# option
option(MNIST "mnist demo" ON)
option(UNIT_TEST "googletest" OFF)

# find cuda
find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
# set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")
 
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++14")
endif()

if (MSVC)
    # warning
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/wd 4819\"")

    if (NOT CMAKE_BUILD_TYPE STREQUAL Debug)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")
    endif()
endif()

# add cuda source
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cuh)
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu)

source_group("Include" FILES ${HEADERS})
source_group("Source" FILES ${SOURCES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# add cuda library
add_library(cu STATIC ${SOURCES})
target_link_libraries(cu PUBLIC cuda)

# mnist demo
if (MNIST) 
    set(MINIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/mnist)
    file(GLOB MINIST_SRCS ${MINIST_DIR}/*.cu)

    add_executable(mnist ${MINIST_SRCS})
    target_include_directories(mnist PRIVATE ${MINIST_DIR})
    target_link_libraries(mnist cu)
endif()

# unit test
if (UNIT_TEST)
    # find google test
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_parts/googletest)

    set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test/cuda)
    file(GLOB TEST_SRCS ${TEST_DIR}/*.cu)

    add_executable(unit_tests ${TEST_SRCS})
    target_include_directories(unit_tests PRIVATE ${TEST_DIR})
    target_link_libraries(unit_tests cu gtest_main)

    add_test(NAME unit_tests COMMAND unit_tests)
endif()