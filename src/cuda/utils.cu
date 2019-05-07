#include <utils.cuh>

__host__ __device__ void index2loc(int index, const int *shape, int dims,
                                   int *loc) {
  for (int i = dims - 1; i >= 0; i--) {
    loc[i] = index % shape[i];
    index /= shape[i];
  }
}

__host__ __device__ int loc2index(const int *loc, const int *shape, int dims) {
  int index = 0;
  int base = 1;
  for (int i = dims - 1; i >= 0; i--) {
    index += base * loc[i];
    base *= shape[i];
  }
  return index;
}

__host__ __device__ void swap(int &a, int &b) {
  int temp = a;
  a = b;
  b = temp;
}