#pragma once

#define BLOCK_SIZE 256
#define TILED_SIZE 16

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

template <class T> __device__ void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}