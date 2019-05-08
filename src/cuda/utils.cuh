#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

#define CHECK_EQ(val1, val2, message)                              \
  do {                                                             \
    if (val1 != val2) {                                            \
      std::cerr << __FILE__ << "(" << __LINE__ << "): " << message \
                << std::endl;                                      \
      exit(1);                                                     \
    }                                                              \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition)                                \
  do {                                                       \
    cudaError_t error = condition;                           \
    CHECK_EQ(error, cudaSuccess, cudaGetErrorString(error)); \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

template <class T>
void stl_clear_object(T *obj) {
  T tmp;
  tmp.swap(*obj);
  // Sometimes "T tmp" allocates objects with memory (arena implementation?).
  // Hence using additional reserve(0) even if it doesn't always work.
  obj->reserve(0);
}

__host__ __device__ void index2loc(int index, const int *shape, int dims,
                                   int *loc);
__host__ __device__ int loc2index(const int *loc, const int *shape, int dims);
__host__ __device__ void swap(int &a, int &b);
