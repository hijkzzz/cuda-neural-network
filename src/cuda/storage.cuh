#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Storage {
 public:
  explicit Storage(thrust::host_vector<unsigned int> shape, float value = 0);
  explicit Storage(std::initializer_list<unsigned int> shape, float value = 0);
  explicit Storage(thrust::host_vector<unsigned int> shape,
                   thrust::device_vector<float> &&data);

  void xavier(size_t in_size, size_t out_size);

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<unsigned int> shape;
};