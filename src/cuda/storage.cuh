#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

class Storage {
 public:
  explicit Storage(std::vector<int> shape, float value = 0);
  explicit Storage(std::vector<int> shape, const std::vector<float> &data);
  explicit Storage(thrust::device_vector<int> shape, float value = 0);
  explicit Storage(thrust::device_vector<int> shape,
                   thrust::device_vector<float> &&data);
  explicit Storage(std::vector<int> shape,
                   thrust::device_vector<float>::const_iterator begin,
                   thrust::device_vector<float>::const_iterator end);

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