#include <storage.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>

#include <cmath>
#include <exception>

Storage::Storage() {}

Storage::Storage(const std::vector<int> &_shape, float value)
    : shape(_shape.begin(), _shape.end()) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  this->data.resize(size, value);
}

Storage::Storage(const std::vector<int> &_shape,
                 const std::vector<float> &_data)
    : shape(_shape.begin(), _shape.end()), data(_data.begin(), _data.end()) {
  this->check_size();
}

Storage::Storage(const std::vector<int> &_shape,
                 thrust::device_vector<float>::const_iterator begin,
                 thrust::device_vector<float>::const_iterator end)
    : shape(_shape.begin(), _shape.end()), data(begin, end) {
  this->check_size();
}

Storage::Storage(const thrust::device_vector<int> &_shape, float value)
    : shape(_shape.begin(), _shape.end()) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  this->data.resize(size, value);
}

Storage::Storage(const thrust::device_vector<int> &_shape,
                 const thrust::device_vector<float> &_data)
    : shape(_shape.begin(), _shape.end()), data(_data.begin(), _data.end()) {
  this->check_size();
}

Storage::Storage(const thrust::device_vector<int> &_shape,
                 thrust::device_vector<float>::const_iterator begin,
                 thrust::device_vector<float>::const_iterator end)
    : shape(_shape.begin(), _shape.end()), data(begin, end) {
  this->check_size();
}

Storage::Storage(const Storage &other) {
  this->shape.resize(other.shape.size());
  thrust::copy(other.shape.begin(), other.shape.end(), this->shape.begin());

  this->data.resize(other.data.size());
  thrust::copy(other.data.begin(), other.data.end(), this->data.begin());
}

Storage &Storage::operator=(const Storage &other) {
  if (this != &other) {
    this->shape.resize(other.shape.size());
    thrust::copy(other.shape.begin(), other.shape.end(), this->shape.begin());

    this->data.resize(other.data.size());
    thrust::copy(other.data.begin(), other.data.end(), this->data.begin());
  }

  return *this;
}

Storage::Storage(Storage &&other) {
  this->data.swap(other.data);
  this->shape.swap(other.shape);
}

Storage &Storage::operator=(Storage &&other) {
  if (this != &other) {
    this->data.swap(other.data);
    this->shape.swap(other.shape);
  }
  return *this;
}

void Storage::check_size() {
  CHECK_EQ(true, this->shape.size() >= 2, "Storage: error, shape.size() < 2");
  int size = thrust::reduce(this->shape.begin(), this->shape.end(), (int)1,
                            thrust::multiplies<int>());
  CHECK_EQ(size, this->data.size(), "Storage: error size");
}

void Storage::reshape(const std::vector<int> &_shape) {
  this->shape.assign(_shape.begin(), _shape.end());
  this->check_size();
}

void Storage::reshape(const thrust::device_vector<int> &_shape) {
  this->shape.assign(_shape.begin(), _shape.end());
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

  float scale = std::sqrt((float)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale);

  CUDA_POST_KERNEL_CHECK;
}