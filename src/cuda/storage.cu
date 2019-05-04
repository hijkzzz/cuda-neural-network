#include <storage.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>

#include <cmath>
#include <exception>

Storage::Storage(std::vector<int> shape, float value)
    : shape(shape.begin(), shape.end()) {
  int size = thrust::reduce(this->shape.begin(), this->shape.end(), (int)1,
                            thrust::multiplies<int>());
  this->data.resize(size, value);
}

Storage::Storage(std::vector<int> shape, const std::vector<float> &data)
    : shape(shape), data(data.begin(), data.end()) {
  this->check_size();
}

Storage::Storage(thrust::device_vector<int> shape, float value) : shape(shape) {
  int size = thrust::reduce(this->shape.begin(), this->shape.end(), (int)1,
                            thrust::multiplies<int>());
  this->data.resize(size, value);
}

Storage::Storage(thrust::device_vector<int> shape,
                 thrust::device_vector<float> &&data)
    : shape(shape) {
  this->data = std::move(data);
  this->check_size();
}

Storage::Storage(std::vector<int> shape,
                 thrust::device_vector<float>::const_iterator begin,
                 thrust::device_vector<float>::const_iterator end)
    : shape(shape.begin(), shape.end()), data(begin, end) {
  this->check_size();
}

void Storage::check_size() {
  int size = thrust::reduce(this->shape.begin(), this->shape.end(), (int)1,
                            thrust::multiplies<int>());
  CHECK_EQ(size, this->data.size(), "Storage: error size");
}

void Storage::reshape(std::vector<int> shape) {
  this->shape.assign(shape.begin(), shape.end());
  this->check_size();
}

std::vector<int> Storage::get_shape() {
  return std::vector<int>(this->shape.begin(), this->shape.end());
}

std::vector<float> Storage::get_data() {
  return std::vector<float>(this->data.begin(), this->data.end());
}

__global__ void storage_xavier(float *a, int size, float scale) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curandState s;
    curand_init(1234, index, 0, &s);
    a[index] = (curand_uniform(&s) * 2 - 1) * scale;
  }
}

void Storage::xavier(size_t in_size, size_t out_size) {
  float *a_ptr = thrust::raw_pointer_cast(this->data.data());
  int size = this->data.size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  float scale = std::sqrt((double)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale);

  CUDA_POST_KERNEL_CHECK;
}