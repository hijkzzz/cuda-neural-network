#include <storage.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>

#include <cmath>
#include <exception>

Storage::Storage(std::vector<unsigned int> shape, float value = 0)
    : shape(shape) {
  unsigned int size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (unsigned int)1,
                     thrust::multiplies<unsigned int>());
  this->data.resize(size, value);
}

Storage::Storage(thrust::host_vector<unsigned int> shape, float value = 0)
    : shape(shape) {
  unsigned int size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (unsigned int)1,
                     thrust::multiplies<unsigned int>());
  this->data.resize(size, value);
}

Storage::Storage(thrust::host_vector<unsigned int> shape,
                 thrust::device_vector<float> &&data)
    : shape(shape) {
  this->data = std::move(data);
  this->check_size();
}

Storage::Storage(std::vector<unsigned int> shape,
                 thrust::device_vector<unsigned int>::const_iterator begin,
                 thrust::device_vector<unsigned int>::const_iterator end)
    : shape(shape), data(begin, end) {
  this->check_size();
}

void Storage::check_size() {
  unsigned int size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (unsigned int)1,
                     thrust::multiplies<unsigned int>());
  CHECK_EQ(size, this->data.size(), "error size");
}

__global__ void storage_xavier(float *a, unsigned int size, float scale) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curandState s;
    curand_init(1234, index, 0, &s);
    a[index] = (curand_uniform(&s) * 2 - 1) * scale;
  }
}

void Storage::reshape(std::vector<unsigned int> shape) {
  this->shape.assign(shape.begin(), shape.end());
  this->check_size();
}

void Storage::xavier(size_t in_size, size_t out_size) {
  float *a_ptr = thrust::raw_pointer_cast(this->data.data());
  unsigned int size = this->data.size();
  unsigned int block_size = ceil((float)(size) / BLOCK_SIZE);
  float scale = std::sqrt((double)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<block_size, BLOCK_SIZE>>>(a_ptr, size, scale);

  CUDA_POST_KERNEL_CHECK;
}