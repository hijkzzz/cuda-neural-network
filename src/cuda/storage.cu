#include <storage.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cmath>

Storage::Storage(const std::vector<int> &_shape) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  this->data.resize(size);
}

Storage::Storage(const std::vector<int> &_shape, float value) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  this->data.resize(size, value);
}

Storage::Storage(const std::vector<int> &_shape,
                 const std::vector<float> &_data)
    : shape(_shape), data(_data.begin(), _data.end()) {
  this->check_size();
}

Storage::Storage(const std::vector<int> &_shape,
                 thrust::device_vector<float>::const_iterator begin,
                 thrust::device_vector<float>::const_iterator end)
    : shape(_shape), data(begin, end) {
  this->check_size();
}

Storage::Storage(const Storage &other) {
  this->shape = other.shape;

  this->data.resize(other.data.size());
  thrust::copy(other.data.begin(), other.data.end(), this->data.begin());
}

Storage &Storage::operator=(const Storage &other) {
  if (this != &other) {
    this->shape = other.shape;

    this->data.resize(other.data.size());
    thrust::copy(other.data.begin(), other.data.end(), this->data.begin());
  }

  return *this;
}

Storage::Storage(Storage &&other) {
  this->shape = std::move(other.shape);
  this->data = std::move(other.data);
}

Storage &Storage::operator=(Storage &&other) {
  if (this != &other) {
    this->shape = std::move(other.shape);
    this->data = std::move(other.data);
  }
  return *this;
}

void Storage::reshape(const std::vector<int> &_shape) {
  this->shape = _shape;
  this->check_size();
}

void Storage::resize(const std::vector<int> &_shape) {
  this->shape = _shape;

  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  if (size != this->data.size()) this->data.resize(size);
  this->check_size();
}

__global__ void storage_xavier(float *a, int size, float scale,
                               curandState *cs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curand_init(1234, index, 0, &cs[index]);
    a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
  }
}

void Storage::xavier(size_t in_size, size_t out_size) {
  float *a_ptr = RAW_PTR(this->data);
  int size = this->data.size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  thrust::device_vector<curandState> cs(size);
  curandState *cs_ptr = RAW_PTR(cs);
  float scale = std::sqrt((float)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale, cs_ptr);

  CUDA_POST_KERNEL_CHECK;
}

void Storage::check_size() {
  CHECK_EQ(true, this->shape.size() >= 2,
           "Storage: shape error, shape.size() < 2");
  int size = 1;
  for (int i = 0; i < this->shape.size(); i++) {
    size *= this->shape[i];
  }
  CHECK_EQ(size, this->data.size(), "Storage: size error");
}