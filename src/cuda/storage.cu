#include <storage.cuh>
#include <utils.h>

#include <exception>
#include <cmath>

#include <cuda_runtime.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

Storage::Storage(thrust::device_vector<std::size_t> shape) : shape(shape) {
  std::size_t size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (std::size_t)1,
                     thrust::multiplies<std::size_t>());
  this->data.resize(size);
}

Storage::Storage(thrust::device_vector<std::size_t> shape,
                 thrust::device_vector<float> &&data)
    : shape(shape) {
  std::size_t size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (std::size_t)1,
                     thrust::multiplies<std::size_t>());
  if (data.size() != size) {
    throw "data size != shape";
  }

  this->data = std::move(data);
}

void Storage::fill(float val) {
  thrust::fill(this->data.begin(), this->data.end(), val);
}

void Storage::xavier(size_t in_size, size_t out_size) {
  float *a_ptr = thrust::raw_pointer_cast(this->data.data());
  std::size_t size = this->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  float scale =
      std::sqrt((double)6) / std::sqrt(std::static_cast<double>(in_size) + out_size);
  storage_xavier<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, size, scale);
}

__global__ storage_uniform(float *a, std::size_t size, float scale) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curandState s;
    curand_init(1234, index, 0, &s);
    a[index] = (curand_uniform(&s) * 2 - 1) * scale;
  }
}