#pragma once

#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define TILED_SIZE 16

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition)                                                  \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error);          \
  } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

template <class T> inline __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

__device__ void index2loc(std::size_t index, std::size_t *shape,
                          std::size_t dims, std::size_t *loc) {
  for (std::size_t i = dims - 1; i >= 0; i--) {
    loc[i] = index % shape[i];
    index /= shape[i];
  }
}

__device__ std::size_t loc2index(std::size_t *loc, std::size_t *shape,
                                 std::size_t dims) {
  std::size_t index = 0;
  std::size_t base = 1;
  for (std::size_t i = dims - 1; i >= 0; i--) {
    index += base * loc[i];
    base *= shape[i];
  }
  return index;
}
