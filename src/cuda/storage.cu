#include <storage.cuh>
#include <utils.cuh>

#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>

#include <cmath>
#include <exception>

Storage::Storage(thrust::host_vector<std::size_t> shape, float value = 0)
    : shape(shape) {
  std::size_t size = thrust::reduce(shape.begin(), shape.end(), (std::size_t)1,
                                    thrust::multiplies<std::size_t>());
  this->data.resize(size, value);
}

Storage::Storage(std::initializer_list<std::size_t> shape, float value) {
  this->shape = thrust::device_vector<std::size_t>(shape.begin(), shape.end());
  std::size_t size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (std::size_t)1,
                     thrust::multiplies<std::size_t>());
  this->data.resize(size, value);
}

Storage::Storage(thrust::host_vector<std::size_t> shape,
                 thrust::device_vector<float> &&data)
    : shape(shape) {
  std::size_t size =
      thrust::reduce(this->shape.begin(), this->shape.end(), (std::size_t)1,
                     thrust::multiplies<std::size_t>());
  CHECK_EQ(size, data.size(), "Storage: error size");
  this->data = std::move(data);
}

__global__ void storage_xavier(float *a, std::size_t size, float scale) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curandState s;
    curand_init(1234, index, 0, &s);
    a[index] = (curand_uniform(&s) * 2 - 1) * scale;
  }
}

void Storage::xavier(size_t in_size, size_t out_size) {
  float *a_ptr = thrust::raw_pointer_cast(this->data.data());
  std::size_t size = this->data.size();
  std::size_t block_size = ceil((float)(size) / BLOCK_SIZE);
  float scale = std::sqrt((double)6) / std::sqrt((float)(in_size) + out_size);
  storage_xavier<<<block_size, BLOCK_SIZE>>>(a_ptr, size, scale);

  CUDA_POST_KERNEL_CHECK;
}