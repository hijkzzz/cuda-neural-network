#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 256
#define TILE_SIZE 16

#define CHECK_EQ(val1, val2, message) \
  do {                                \
    if (val1 != val2) throw message;  \
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
inline __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

__device__ void index2loc(unsigned int index, const unsigned int *shape,
                          unsigned int dims, unsigned int *loc) {
  for (unsigned int i = dims - 1; i >= 0; i--) {
    loc[i] = index % shape[i];
    index /= shape[i];
  }
}

__device__ unsigned int loc2index(const unsigned int *loc,
                                  const unsigned int *shape,
                                  unsigned int dims) {
  unsigned int index = 0;
  unsigned int base = 1;
  for (unsigned int i = dims - 1; i >= 0; i--) {
    index += base * loc[i];
    base *= shape[i];
  }
  return index;
}
