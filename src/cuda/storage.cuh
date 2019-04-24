#pragma once

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class Storage {
public:
  Storage(thrust::host_vector<std::size_t> shape, float value = 0);
  Storage(thrust::host_vector<std::size_t> shape,
          thrust::device_vector<float> &&data);

  void xavier(size_t in_size, size_t out_size);

  // data
  thrust::device_vector<float> data;
  thrust::device_vector<std::size_t> shape;
};

__global__ storage_xavier(float *a, std::size_t size, size_t in_size, size_t out_size);