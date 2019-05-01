#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <initializer_list>
#include <iterator>

class Storage {
 public:
  explicit Storage(std::initializer_list<int> shape, float value = 0);
  explicit Storage(thrust::host_vector<int> shape, float value = 0);
  explicit Storage(thrust::host_vector<int> shape,
                   thrust::device_vector<float> &&data);
  explicit Storage(std::initializer_list<int> shape,
                   thrust::device_vector<float>::const_iterator begin,
                   thrust::device_vector<float>::const_iterator end);

  void check_size();

  void reshape(std::initializer_list<int> shape);
  void xavier(size_t in_size, size_t out_size);

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<int> shape;
};