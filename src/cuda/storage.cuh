#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iterator>
#include <vector>

class Storage {
 public:
  explicit Storage(const std::vector<int> &_shape);
  explicit Storage(const std::vector<int> &_shape, float value);
  explicit Storage(const std::vector<int> &_shape, const std::vector<float> &_data);

  // copy/move
  Storage(const Storage &other);
  Storage &operator=(const Storage &other);
  Storage(Storage &&other);
  Storage &operator=(Storage &&other);

  void reshape(const std::vector<int> &_shape);
  void resize(const std::vector<int> &_shape);
  void xavier(size_t in_size, size_t out_size);

  // get
  std::vector<int> &get_shape() { return this->shape; };
  const std::vector<int> &get_shape() const { return this->shape; };
  thrust::device_vector<float> &get_data() { return this->data; };
  const thrust::device_vector<float> &get_data() const { return this->data; };

 private:
  void check_size();  // check data/shape size

  thrust::device_vector<float> data;
  std::vector<int> shape;
};