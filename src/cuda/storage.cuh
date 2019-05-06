#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

class Storage {
 public:
  explicit Storage(const std::vector<int> &_shape, float value = 0);
  explicit Storage(const std::vector<int> &_shape,
                   const std::vector<float> &_data);
  explicit Storage(const std::vector<int> &_shape,
                   thrust::device_vector<float>::const_iterator begin,
                   thrust::device_vector<float>::const_iterator end);

  explicit Storage(const thrust::device_vector<int> &_shape, float value = 0);
  explicit Storage(const thrust::device_vector<int> &_shape,
                   thrust::device_vector<float> &&_data);

  Storage(const Storage& other);
  Storage& operator=(const Storage& other);
  Storage(Storage&& other);
  Storage& operator=(Storage&& other);
  ~Storage();

  void xavier(size_t in_size, size_t out_size);
  void reshape(std::vector<int> shape);

  std::vector<int> get_shape();
  std::vector<float> get_data();

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<int> shape;

 private:
  void check_size();
};