#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iterator>

class Storage {
 public:
  explicit Storage(std::vector<unsigned int> shape, float value = 0);
  explicit Storage(thrust::host_vector<unsigned int> shape, float value = 0);
  explicit Storage(thrust::host_vector<unsigned int> shape,
                   thrust::device_vector<float> &&data);
  explicit Storage(std::vector<unsigned int> shape,
                   thrust::device_vector<unsigned int>::const_iterator begin,
                   thrust::device_vector<unsigned int>::const_iterator end);

  void check_size();

  void reshape(std::vector<unsigned int> shape);
  void xavier(size_t in_size, size_t out_size);

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<unsigned int> shape;
};