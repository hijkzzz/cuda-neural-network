#include <storage.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>

#include <cmath>
#include <exception>

Storage::Storage(const std::vector<int> &_shape, float value) {
  this->shape.resize(_shape.size());
  thrust::copy(_shape.begin(), _shape.end(), this->shape.begin());

  int size = thrust::reduce(this->shape.begin(), this->shape.end(), (int)1,
                            thrust::multiplies<int>());
  this->data.resize(size);
  thrust::fill(this->data.begin(), this->data.end(), value);
}

Storage::Storage(const std::vector<int> &_shape,
                 const std::vector<float> &_data) {
  this->shape.resize(_shape.size());
  thrust::copy(_shape.begin(), _shape.end(), this->shape.begin());

  this->data.resize(_data.size());
  thrust::copy(_data.begin(), _data.end(), this->data.begin());

  this->check_size();
}

Storage::Storage(const std::vector<int> &_shape,
                 thrust::device_vector<float>::const_iterator begin,
                 thrust::device_vector<float>::const_iterator end) {
  this->shape.resize(_shape.size());
  thrust::copy(_shape.begin(), _shape.end(), this->shape.begin());

  this->data.resize(end - begin);
  thrust::copy(begin, end, this->data.begin());
  this->check_size();
}

Storage::Storage(const thrust::device_vector<int> &_shape, float value) {
  this->shape.resize(_shape.size());
  thrust::copy(_shape.begin(), _shape.end(), this->shape.begin());

  int size = thrust::reduce(this->shape.begin(), this->shape.end(), (int)1,
                            thrust::multiplies<int>());
  this->data.resize(size);
  thrust::fill(this->data.begin(), this->data.end(), value);
}

Storage::Storage(const thrust::device_vector<int> &_shape,
                 thrust::device_vector<float> &&_data) {
  this->shape.resize(_shape.size());
  thrust::copy(_shape.begin(), _shape.end(), this->shape.begin());

  this->data = std::move(_data);
  this->check_size();
}

Storage::Storage(const Storage &other) {
  this->data.resize(other.data.size());
  thrust::copy(other.data.begin(), other.data.end(), this->data.begin());

  this->shape.resize(other.shape.size());
  thrust::copy(other.shape.begin(), other.shape.end(), this->shape.begin());
}

Storage &Storage::operator=(const Storage &other) {
  if (this != &other) {
    this->data.resize(other.data.size());
    thrust::copy(other.data.begin(), other.data.end(), this->data.begin());

    this->shape.resize(other.shape.size());
    thrust::copy(other.shape.begin(), other.shape.end(), this->shape.begin());
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

Storage::~Storage() {
  //stl_clear_object(&this->data);
  //stl_clear_object(&this->shape);
}

void Storage::check_size() {
  CHECK_EQ(true, this->shape.size() >= 2, "Storage: error, shape.size() < 2");
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

  float scale = std::sqrt((float)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<grid_size, BLOCK_SIZE>>>(a_ptr, size, scale);

  CUDA_POST_KERNEL_CHECK;
}