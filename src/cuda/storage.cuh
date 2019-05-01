#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <vector>
#include <iterator>

class Storage {
 public:
  explicit Storage(std::vector<int> shape, float value = 0);
  explicit Storage(thrust::device_vector<int> shape, float value = 0);
  explicit Storage(thrust::device_vector<int> shape,
                   thrust::device_vector<float> &&data);
  explicit Storage(std::vector<int> shape,
                   thrust::device_vector<float>::const_iterator begin,
                   thrust::device_vector<float>::const_iterator end);

  void check_size();

  void reshape(std::vector<int> shape);
  void xavier(size_t in_size, size_t out_size);

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<int> shape;
};