#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>

#define BLOCK_SIZE 512
#define TILE_SIZE 16

#define CHECK_EQ(val1, val2, message)                              \
  do {                                                             \
    if (val1 != val2) {                                            \
      std::cout << __FILE__ << "(" << __LINE__ << "): " << message \
                << std::endl;                                      \
      throw std::runtime_error(message);                           \
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

inline __host__ __device__ void index2loc(int index, const int *shape, int dims,
                                          int *loc) {
  for (int i = dims - 1; i >= 0; i--) {
    loc[i] = index % shape[i];
    index /= shape[i];
  }
}

inline __host__ __device__ int loc2index(const int *loc, const int *shape,
                                         int dims) {
  int index = 0;
  int base = 1;
  for (int i = dims - 1; i >= 0; i--) {
    index += base * loc[i];
    base *= shape[i];
  }
  return index;
}

inline __host__ __device__ void swap(int &a, int &b) {
  int temp = a;
  a = b;
  b = temp;
}

inline __host__ __device__ void swap(float &a, float &b) {
  float temp = a;
  a = b;
  b = temp;
}

template <class T>
void stl_clear_object(T *obj) {
  T tmp;
  tmp.swap(*obj);
  // Sometimes "T tmp" allocates objects with memory (arena implementation?).
  // Hence using additional reserve(0) even if it doesn't always work.
  obj->reserve(0);
}