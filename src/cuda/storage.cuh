#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

class Storage {
 public:
  Storage();
  // std::vector
  Storage(const std::vector<int> &_shape, float value = 0);
  Storage(const std::vector<int> &_shape, const std::vector<float> &_data);
  Storage(const std::vector<int> &_shape,
          thrust::device_vector<float>::const_iterator begin,
          thrust::device_vector<float>::const_iterator end);

  // thrust::device_vector
  Storage(const thrust::device_vector<int> &_shape, float value = 0);
  Storage(const thrust::device_vector<int> &_shape,
          const thrust::device_vector<float> &_data);
  Storage(const thrust::device_vector<int> &_shape,
          thrust::device_vector<float>::const_iterator begin,
          thrust::device_vector<float>::const_iterator end);

  // copy/move
  Storage(const Storage &other);
  Storage &operator=(const Storage &other);
  Storage(Storage &&other);
  Storage &operator=(Storage &&other);

  void reshape(const std::vector<int> &_shape);
  void reshape(const thrust::device_vector<int> &_shape);
  void xavier(size_t in_size, size_t out_size);

  // copy to host
  std::vector<int> get_shape();
  std::vector<float> get_data();

  // raw data
  thrust::device_vector<float> data;
  thrust::device_vector<int> shape;

 private:
  void check_size();
};